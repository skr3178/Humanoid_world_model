#!/bin/bash
# =============================================================================
# Automated Setup Script for Humanoid World Model on New GPU Server
# Target GPU: RTX 4090 (24GB) or similar
# =============================================================================
# This script will:
# 1. Check system prerequisites (conda, git, CUDA, huggingface-cli)
# 2. Create project structure
# 3. Download 1x-technologies/world_model_tokenized_data v2.0 dataset from HuggingFace
# 4. Download build folder from GitHub repository
# 5. Download and setup Cosmos Tokenizer from HuggingFace and GitHub
# 6. Create and configure conda environment 'cosmos-tokenizer'
# 7. Install all dependencies
# 8. Verify the installation
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Configuration
# =============================================================================

# Prompt for base directory
echo ""
log_info "=========================================="
log_info "Humanoid World Model - Server Setup"
log_info "=========================================="
echo ""

read -p "Enter the base directory for installation (default: ~/humanoid_wm): " BASE_DIR
BASE_DIR=${BASE_DIR:-~/humanoid_wm}
BASE_DIR=$(eval echo "$BASE_DIR")  # Expand ~ to home directory

log_info "Installation directory: $BASE_DIR"
echo ""

# Ask about checkpoint download
read -p "Download existing checkpoints from GitHub? (y/n, default: y): " DOWNLOAD_CHECKPOINTS
DOWNLOAD_CHECKPOINTS=${DOWNLOAD_CHECKPOINTS:-y}

# GitHub repository
GITHUB_REPO="https://github.com/skr3178/Humanoid_world_model"

# Conda environment name
CONDA_ENV_NAME="cosmos-tokenizer"

# =============================================================================
# Phase 1: Check System Prerequisites
# =============================================================================

log_info "Phase 1: Checking system prerequisites..."

# Check for conda
if ! command -v conda &> /dev/null; then
    log_error "conda not found. Please install Miniconda or Anaconda first."
    log_info "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
log_success "conda found: $(conda --version)"

# Check for git
if ! command -v git &> /dev/null; then
    log_error "git not found. Please install git first."
    exit 1
fi
log_success "git found: $(git --version)"

# Check for git-lfs (optional but recommended)
if ! command -v git-lfs &> /dev/null; then
    log_warning "git-lfs not found. Large files may not download properly."
    log_info "Consider installing: sudo apt-get install git-lfs (Ubuntu/Debian)"
else
    log_success "git-lfs found: $(git-lfs --version)"
fi

# Check for CUDA (optional check)
if command -v nvcc &> /dev/null; then
    log_success "CUDA found: $(nvcc --version | grep release)"
elif command -v nvidia-smi &> /dev/null; then
    log_success "NVIDIA driver found: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
    # Check GPU availability
    if nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
        log_info "GPU detected: $GPU_NAME ($GPU_COUNT GPU(s), ${GPU_MEMORY}MB total)"
    fi
else
    log_warning "CUDA/NVIDIA driver not detected. GPU training may not work."
fi

# Check Python version (if available)
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        log_success "Python version OK: $PYTHON_VERSION"
    else
        log_warning "Python version $PYTHON_VERSION found. Python 3.10+ recommended."
    fi
fi

# Check disk space (estimate ~150GB needed for dataset + models + environment)
log_info "Checking available disk space..."
AVAILABLE_SPACE=$(df -BG "$BASE_DIR" 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//')
if [ -n "$AVAILABLE_SPACE" ] && [ "$AVAILABLE_SPACE" -lt 150 ]; then
    log_warning "Available disk space: ${AVAILABLE_SPACE}GB (recommended: 150GB+)"
    read -p "Continue anyway? (y/n): " CONTINUE_LOW_SPACE
    if [[ ! "$CONTINUE_LOW_SPACE" =~ ^[Yy]$ ]]; then
        log_error "Setup cancelled due to insufficient disk space"
        exit 1
    fi
else
    log_success "Available disk space: ${AVAILABLE_SPACE}GB"
fi

echo ""

# =============================================================================
# Phase 2: Create Project Structure
# =============================================================================

log_info "Phase 2: Creating project structure..."

mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

log_success "Created base directory: $BASE_DIR"
echo ""

# =============================================================================
# Phase 3: Download 1xGPT v2.0 Dataset from HuggingFace
# =============================================================================

log_info "Phase 3: Downloading 1x-technologies/world_model_tokenized_data v2.0 dataset..."
log_info "Downloading only train_v2.0, val_v2.0, and test_v2.0 directories..."
log_info "This may take a while (~100GB download)..."

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    log_warning "huggingface-cli not found. Installing..."
    pip install --upgrade huggingface-hub
fi

# Check HuggingFace authentication (optional, but may be needed for some repos)
log_info "Checking HuggingFace authentication..."
if huggingface-cli whoami &> /dev/null; then
    HF_USER=$(huggingface-cli whoami 2>/dev/null)
    log_success "HuggingFace authenticated as: $HF_USER"
else
    log_warning "Not logged into HuggingFace. Some downloads may require authentication."
    log_info "To login, run: huggingface-cli login"
    log_info "Or set HF_TOKEN environment variable"
    read -p "Continue without HuggingFace login? (y/n, default: y): " CONTINUE_NO_HF
    CONTINUE_NO_HF=${CONTINUE_NO_HF:-y}
    if [[ ! "$CONTINUE_NO_HF" =~ ^[Yy]$ ]]; then
        log_info "Please login to HuggingFace and run this script again."
        exit 0
    fi
fi

# Create 1xgpt directory structure
mkdir -p 1xgpt/data

# Download train_v2.0 (recursively download all files in the directory)
log_info "Downloading train_v2.0..."
huggingface-cli download 1x-technologies/world_model_tokenized_data \
    --repo-type dataset \
    --local-dir 1xgpt/data \
    --include "train_v2.0/**"

# Download val_v2.0 (recursively download all files in the directory)
log_info "Downloading val_v2.0..."
huggingface-cli download 1x-technologies/world_model_tokenized_data \
    --repo-type dataset \
    --local-dir 1xgpt/data \
    --include "val_v2.0/**"

# Download test_v2.0 (if available, recursively download all files)
log_info "Downloading test_v2.0 (if available)..."
if huggingface-cli download 1x-technologies/world_model_tokenized_data \
    --repo-type dataset \
    --local-dir 1xgpt/data \
    --include "test_v2.0/**" 2>/dev/null; then
    log_success "test_v2.0 downloaded successfully"
else
    log_warning "test_v2.0 not available or download failed (this is optional)"
fi

# Verify download
log_info "Verifying downloaded datasets..."
if [ -d "1xgpt/data/train_v2.0" ] && [ -d "1xgpt/data/val_v2.0" ]; then
    log_success "Required datasets downloaded successfully!"
    
    # Count files in each directory (check for .tar files or metadata.json)
    train_files=$(find 1xgpt/data/train_v2.0 -type f 2>/dev/null | wc -l)
    val_files=$(find 1xgpt/data/val_v2.0 -type f 2>/dev/null | wc -l)
    
    log_info "Training dataset: $train_files files in train_v2.0/"
    log_info "Validation dataset: $val_files files in val_v2.0/"
    
    if [ -d "1xgpt/data/test_v2.0" ]; then
        test_files=$(find 1xgpt/data/test_v2.0 -type f 2>/dev/null | wc -l)
        log_info "Test dataset: $test_files files in test_v2.0/"
    else
        log_info "Test dataset: not downloaded (optional)"
    fi
    
    # Check for metadata files to ensure structure is correct
    if [ -f "1xgpt/data/train_v2.0/metadata.json" ] && [ -f "1xgpt/data/val_v2.0/metadata.json" ]; then
        log_success "Dataset structure verified (metadata.json found)"
        # Check if directories have content
        if [ "$train_files" -eq 0 ] || [ "$val_files" -eq 0 ]; then
            log_warning "Dataset directories exist but appear empty. Download may have failed."
        fi
    else
        log_warning "metadata.json not found - dataset structure may be different"
        log_info "Checking for alternative structure (videos/, robot_states/ directories)..."
        if [ -d "1xgpt/data/train_v2.0/videos" ] || [ -d "1xgpt/data/train_v2.0/robot_states" ]; then
            log_success "Alternative dataset structure detected"
        fi
    fi
else
    log_error "Dataset download failed or incomplete."
    log_error "Required directories train_v2.0 and val_v2.0 must exist."
    exit 1
fi

echo ""

# =============================================================================
# Phase 4: Download Build Folder from GitHub
# =============================================================================

log_info "Phase 4: Downloading build folder from GitHub..."

# Clone repository to temporary location
TEMP_REPO="temp_humanoid_repo"
log_info "Cloning repository: $GITHUB_REPO"

if [ -d "$TEMP_REPO" ]; then
    rm -rf "$TEMP_REPO"
fi

git clone "$GITHUB_REPO" "$TEMP_REPO"

# Copy build folder
if [ -d "$TEMP_REPO/build" ]; then
    log_info "Copying build folder..."
    cp -r "$TEMP_REPO/build" .
    log_success "Build folder copied successfully!"
else
    log_error "Build folder not found in repository."
    exit 1
fi

# Copy other essential files if they exist
for file in requirements.txt setup.py README.md; do
    if [ -f "$TEMP_REPO/$file" ]; then
        cp "$TEMP_REPO/$file" .
        log_info "Copied: $file"
    fi
done

# Copy masked_hwm and data directories if they exist
for dir in masked_hwm data; do
    if [ -d "$TEMP_REPO/$dir" ]; then
        cp -r "$TEMP_REPO/$dir" .
        log_info "Copied directory: $dir"
    fi
done

# Download checkpoints if requested
if [[ "$DOWNLOAD_CHECKPOINTS" =~ ^[Yy]$ ]]; then
    log_info "Checking for checkpoint folders in repository..."

    for checkpoint_dir in checkpoints_10pct_v2_12gb checkpoints_full_v2_12gb checkpoints_test_subset; do
        if [ -d "$TEMP_REPO/build/$checkpoint_dir" ]; then
            log_info "Downloading: $checkpoint_dir (this may take a while)..."
            cp -r "$TEMP_REPO/build/$checkpoint_dir" build/
            log_success "Downloaded: $checkpoint_dir"
        else
            log_warning "Checkpoint folder not found: $checkpoint_dir"
        fi
    done
fi

# Clean up temporary repository
rm -rf "$TEMP_REPO"
log_success "GitHub repository downloaded and extracted!"

echo ""

# =============================================================================
# Phase 5: Setup Cosmos Tokenizer
# =============================================================================

log_info "Phase 5: Setting up Cosmos Tokenizer..."

# Clone Cosmos-Tokenizer repository
log_info "Cloning NVIDIA/Cosmos-Tokenizer from GitHub..."
if [ -d "Cosmos-Tokenizer" ]; then
    log_warning "Cosmos-Tokenizer directory already exists. Skipping clone."
else
    git clone https://github.com/NVIDIA/Cosmos-Tokenizer.git
    log_success "Cosmos-Tokenizer repository cloned!"
fi

# Download Cosmos tokenizer models from HuggingFace
log_info "Downloading Cosmos-0.1-Tokenizer-DV8x8x8 (DV 8x8x8) models from HuggingFace..."
log_info "This includes encoder.jit, decoder.jit, and autoencoder.jit files..."
mkdir -p cosmos_tokenizer

huggingface-cli download nvidia/Cosmos-0.1-Tokenizer-DV8x8x8 \
    --local-dir cosmos_tokenizer/

# Verify tokenizer download
log_info "Verifying Cosmos Tokenizer DV 8x8x8 checkpoint files..."
if [ -f "cosmos_tokenizer/encoder.jit" ] && [ -f "cosmos_tokenizer/decoder.jit" ]; then
    log_success "Cosmos tokenizer DV 8x8x8 checkpoints downloaded successfully!"
    log_info "  - encoder.jit: $(du -h cosmos_tokenizer/encoder.jit 2>/dev/null | cut -f1 || echo 'found')"
    log_info "  - decoder.jit: $(du -h cosmos_tokenizer/decoder.jit 2>/dev/null | cut -f1 || echo 'found')"
    if [ -f "cosmos_tokenizer/autoencoder.jit" ]; then
        log_info "  - autoencoder.jit: $(du -h cosmos_tokenizer/autoencoder.jit 2>/dev/null | cut -f1 || echo 'found')"
    fi
else
    log_error "Cosmos tokenizer DV 8x8x8 model download failed."
    log_error "Required files encoder.jit and decoder.jit not found in cosmos_tokenizer/"
    exit 1
fi

echo ""

# =============================================================================
# Phase 6: Create Conda Environment
# =============================================================================

log_info "Phase 6: Creating conda environment '$CONDA_ENV_NAME'..."

# Check if environment already exists
ENV_EXISTS=false
if conda env list | grep -q "^$CONDA_ENV_NAME "; then
    log_warning "Conda environment '$CONDA_ENV_NAME' already exists."
    read -p "Remove and recreate? (y/n, default: n): " RECREATE_ENV
    RECREATE_ENV=${RECREATE_ENV:-n}
    if [[ "$RECREATE_ENV" =~ ^[Yy]$ ]]; then
        log_info "Removing existing environment..."
        conda env remove -n "$CONDA_ENV_NAME" -y
        ENV_EXISTS=false
    else
        log_info "Using existing environment. Will update packages if needed."
        ENV_EXISTS=true
    fi
fi

if [ "$ENV_EXISTS" = "false" ]; then
    # Create environment from environment.yml if it exists, otherwise create basic environment
    ENV_YML=""
    if [ -f "environment.yml" ]; then
        ENV_YML="environment.yml"
    elif [ -f "build/environment.yml" ]; then
        ENV_YML="build/environment.yml"
    fi
    
    if [ -n "$ENV_YML" ]; then
        log_info "Creating environment from $ENV_YML..."
        # Update the name in environment.yml to match CONDA_ENV_NAME if needed
        if grep -q "^name:" "$ENV_YML"; then
            # Temporarily update the name in environment.yml
            sed -i.bak "s/^name:.*/name: $CONDA_ENV_NAME/" "$ENV_YML"
            conda env create -f "$ENV_YML"
            CREATE_STATUS=$?
            # Restore original if backup exists
            if [ -f "$ENV_YML.bak" ]; then
                mv "$ENV_YML.bak" "$ENV_YML"
            fi
        else
            conda env create -f "$ENV_YML" -n "$CONDA_ENV_NAME"
            CREATE_STATUS=$?
        fi
        
        if [ $CREATE_STATUS -eq 0 ]; then
            log_success "Conda environment created from $ENV_YML!"
        else
            log_error "Failed to create environment from $ENV_YML"
            exit 1
        fi
    else
        log_info "environment.yml not found. Creating new conda environment with Python 3.10..."
        conda create -n "$CONDA_ENV_NAME" python=3.10 -y
        if [ $? -eq 0 ]; then
            log_success "Conda environment created!"
        else
            log_error "Failed to create conda environment"
            exit 1
        fi
    fi
fi

echo ""

# =============================================================================
# Phase 7: Install Dependencies
# =============================================================================

log_info "Phase 7: Installing dependencies..."

# Activate environment and install packages
log_info "Activating conda environment..."

# Initialize conda (try multiple methods)
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    # Try eval method as fallback
    eval "$(conda shell.bash hook)" 2>/dev/null || {
        log_error "Failed to initialize conda. Please ensure conda is installed and in PATH."
        exit 1
    }
fi

conda activate "$CONDA_ENV_NAME"

if [ $? -ne 0 ]; then
    log_error "Failed to activate conda environment '$CONDA_ENV_NAME'"
    log_error "Make sure the environment was created successfully in Phase 6"
    exit 1
fi

log_success "Environment activated: $CONDA_ENV_NAME"

# Upgrade pip first
log_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
log_info "Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

if [ $? -ne 0 ]; then
    log_error "Failed to install PyTorch"
    exit 1
fi
log_success "PyTorch installed successfully"

# Install project requirements.txt files first (before other packages)
log_info "Installing packages from requirements.txt files..."
if [ -f "build/requirements.txt" ]; then
    log_info "Installing from build/requirements.txt..."
    pip install -r build/requirements.txt
    if [ $? -ne 0 ]; then
        log_warning "Some packages from build/requirements.txt may have failed to install"
    else
        log_success "build/requirements.txt packages installed"
    fi
elif [ -f "requirements.txt" ]; then
    log_info "Installing from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        log_warning "Some packages from requirements.txt may have failed to install"
    else
        log_success "requirements.txt packages installed"
    fi
else
    log_warning "No requirements.txt found in root or build directory"
fi

# Install Cosmos Tokenizer dependencies (from Cosmos-Tokenizer repo)
log_info "Installing Cosmos Tokenizer dependencies from Cosmos-Tokenizer/requirements.txt..."
if [ -f "Cosmos-Tokenizer/requirements.txt" ]; then
    pip install -r Cosmos-Tokenizer/requirements.txt
    if [ $? -ne 0 ]; then
        log_warning "Some Cosmos Tokenizer dependencies may have failed to install"
    else
        log_success "Cosmos Tokenizer dependencies installed"
    fi
else
    log_warning "Cosmos-Tokenizer/requirements.txt not found, skipping..."
fi

# Install Cosmos Tokenizer package in editable mode
log_info "Installing cosmos_tokenizer package from Cosmos-Tokenizer..."
if [ -d "Cosmos-Tokenizer" ]; then
    pip install -e ./Cosmos-Tokenizer/
    if [ $? -eq 0 ]; then
        log_success "cosmos_tokenizer package installed successfully"
    else
        log_error "Failed to install cosmos_tokenizer package"
        exit 1
    fi
else
    log_error "Cosmos-Tokenizer directory not found!"
    exit 1
fi

# Install xformers (memory-efficient attention)
log_info "Installing xformers (memory-efficient attention)..."
pip install xformers==0.0.26.post1
if [ $? -ne 0 ]; then
    log_warning "xformers installation failed, continuing anyway..."
else
    log_success "xformers installed successfully"
fi

# Install training framework dependencies
log_info "Installing training framework dependencies..."
pip install accelerate==0.30.1 transformers==4.41.0 lightning wandb tqdm==4.66.4
if [ $? -ne 0 ]; then
    log_error "Failed to install training dependencies"
    exit 1
fi
log_success "Training dependencies installed"

# Install flash-attention (with skip CUDA build flag for faster installation)
log_info "Installing flash-attention (this may take a few minutes)..."
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn==2.5.8 --no-build-isolation
if [ $? -ne 0 ]; then
    log_warning "flash-attention installation failed, continuing without it..."
    log_info "Training will still work but may be slower without flash-attention"
else
    log_success "flash-attention installed successfully"
fi

# Install additional dependencies (if not already installed via requirements.txt)
log_info "Installing additional dependencies..."
pip install loguru mediapy einops einx huggingface-hub matplotlib \
    numpy scipy Pillow lpips tensorboard wheel packaging ninja
if [ $? -ne 0 ]; then
    log_warning "Some additional dependencies may have failed to install (may already be installed)"
else
    log_success "Additional dependencies installed"
fi

# Verify critical packages
log_info "Verifying critical package installations..."

# Verify PyTorch
log_info "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" || log_warning "PyTorch verification failed"

# Verify cosmos_tokenizer package
log_info "Verifying cosmos_tokenizer package installation..."
python -c "from cosmos_tokenizer import ContinuousTokenizer, DiscreteTokenizer; print('cosmos_tokenizer package: OK')" || {
    log_error "cosmos_tokenizer package verification failed!"
    log_error "The cosmos_tokenizer package should be importable after installation."
    exit 1
}

log_success "All critical packages verified successfully!"

log_success "All dependencies installed and verified!"

echo ""

# =============================================================================
# Phase 8: Update Configuration Paths
# =============================================================================

log_info "Phase 8: Updating configuration paths..."

# Update paths in config.py
CONFIG_FILE="build/masked_hwm/config.py"

if [ -f "$CONFIG_FILE" ]; then
    log_info "Updating paths in $CONFIG_FILE..."

    # Backup original config
    cp "$CONFIG_FILE" "$CONFIG_FILE.backup"

    # Update tokenizer checkpoint directory
    sed -i "s|tokenizer_checkpoint_dir: str = \".*\"|tokenizer_checkpoint_dir: str = \"$BASE_DIR/cosmos_tokenizer\"|g" "$CONFIG_FILE"

    # Update train data directory
    sed -i "s|train_data_dir: str = \".*\"|train_data_dir: str = \"$BASE_DIR/1xgpt/data/train_v2.0\"|g" "$CONFIG_FILE"

    # Update val data directory
    sed -i "s|val_data_dir: str = \".*\"|val_data_dir: str = \"$BASE_DIR/1xgpt/data/val_v2.0\"|g" "$CONFIG_FILE"

    # Update test data directory
    sed -i "s|test_data_dir: Optional\[str\] = \".*\"|test_data_dir: Optional[str] = \"$BASE_DIR/1xgpt/data/test_v2.0\"|g" "$CONFIG_FILE"

    log_success "Configuration paths updated!"
    log_info "Original config backed up to: $CONFIG_FILE.backup"
else
    log_warning "Config file not found: $CONFIG_FILE"
fi

# Update paths in training scripts
log_info "Updating paths in training scripts..."
for script in build/train_*.sh; do
    if [ -f "$script" ]; then
        # Backup
        cp "$script" "$script.backup"

        # Update paths
        sed -i "s|TRAIN_DATA_DIR=\".*\"|TRAIN_DATA_DIR=\"$BASE_DIR/1xgpt/data/train_v2.0\"|g" "$script"
        sed -i "s|VAL_DATA_DIR=\".*\"|VAL_DATA_DIR=\"$BASE_DIR/1xgpt/data/val_v2.0\"|g" "$script"
        sed -i "s|TEST_DATA_DIR=\".*\"|TEST_DATA_DIR=\"$BASE_DIR/1xgpt/data/test_v2.0\"|g" "$script"

        log_info "Updated: $script"
    fi
done

log_success "Training scripts updated!"

# Make training scripts executable
log_info "Making training scripts executable..."
for script in build/train_*.sh; do
    if [ -f "$script" ]; then
        chmod +x "$script"
        log_info "Made executable: $script"
    fi
done

# Also make other useful scripts executable
for script in build/*.py build/*.sh; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        chmod +x "$script" 2>/dev/null || true
    fi
done

log_success "Scripts made executable!"

echo ""

# =============================================================================
# Phase 9: Verify Installation
# =============================================================================

log_info "Phase 9: Verifying installation..."

# Create a simple verification script
cat > verify_quick.py << 'EOF'
import sys
import torch

print("=" * 60)
print("Quick Installation Verification")
print("=" * 60)

# Check PyTorch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: CUDA not available!")
    sys.exit(1)

# Check Cosmos Tokenizer
try:
    from cosmos_tokenizer import ContinuousTokenizer, DiscreteTokenizer
    print("Cosmos Tokenizer: OK")
except ImportError as e:
    print(f"Cosmos Tokenizer: FAILED - {e}")
    sys.exit(1)

# Check other imports
try:
    import einops
    import transformers
    import accelerate
    print("Core dependencies: OK")
except ImportError as e:
    print(f"Core dependencies: FAILED - {e}")
    sys.exit(1)

print("=" * 60)
print("Verification PASSED!")
print("=" * 60)
EOF

log_info "Running verification script..."
python verify_quick.py

if [ $? -eq 0 ]; then
    log_success "Installation verified successfully!"
else
    log_error "Verification failed. Please check the errors above."
    exit 1
fi

rm verify_quick.py

# Final checks before completion
log_info "Performing final system checks..."

# Check if training scripts exist and are executable
TRAINING_SCRIPTS_FOUND=0
for script in build/train_*.sh; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        TRAINING_SCRIPTS_FOUND=$((TRAINING_SCRIPTS_FOUND + 1))
    fi
done

if [ $TRAINING_SCRIPTS_FOUND -gt 0 ]; then
    log_success "Found $TRAINING_SCRIPTS_FOUND executable training script(s)"
else
    log_warning "No executable training scripts found in build/ directory"
fi

# Verify dataset directories are accessible
if [ -d "1xgpt/data/train_v2.0" ] && [ -d "1xgpt/data/val_v2.0" ]; then
    log_success "Dataset directories are accessible"
else
    log_error "Dataset directories not found - training will fail!"
fi

# Verify tokenizer files exist
if [ -f "cosmos_tokenizer/encoder.jit" ] && [ -f "cosmos_tokenizer/decoder.jit" ]; then
    log_success "Cosmos tokenizer files are present"
else
    log_error "Cosmos tokenizer files missing - training will fail!"
fi

# Check GPU availability one more time
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        log_success "GPU is accessible"
    else
        log_warning "GPU may not be accessible - check nvidia-smi"
    fi
fi

echo ""
echo ""
log_success "=========================================="
log_success "Setup Complete!"
log_success "=========================================="
echo ""
log_info "Installation summary:"
log_info "  Base directory: $BASE_DIR"
log_info "  Conda environment: $CONDA_ENV_NAME"
log_info "  Dataset: $BASE_DIR/1xgpt/data/"
log_info "  Build folder: $BASE_DIR/build/"
log_info "  Cosmos Tokenizer: $BASE_DIR/cosmos_tokenizer/"
echo ""
log_info "Next steps:"
log_info "  1. Activate environment: conda activate $CONDA_ENV_NAME"
log_info "  2. Navigate to project: cd $BASE_DIR"
log_info "  3. Run verification: python verify_setup.py"
echo ""
log_info "Available training scripts (in build/ directory):"
log_info "  - train_full_v2_a6000.sh    : A6000/A100 optimized (48GB/40GB+) - Full paper config"
log_info "  - train_full_v2.sh          : Full training on v2.0 dataset (24 layers, 512 dim)"
log_info "  - train_full_v2_rtx4090.sh  : RTX 4090 optimized training (24GB)"
log_info "  - train_10pct_v2_12gb.sh    : 10% subset training for 12GB GPUs"
log_info "  - train_quick.sh            : Quick test training"
echo ""
log_info "To start training:"
log_info "  1. Activate conda environment:"
log_info "     conda activate $CONDA_ENV_NAME"
log_info "     # Or if conda is not in PATH:"
log_info "     source \$(conda info --base)/etc/profile.d/conda.sh"
log_info "     conda activate $CONDA_ENV_NAME"
log_info ""
log_info "  2. Navigate to build directory:"
log_info "     cd $BASE_DIR/build"
log_info ""
log_info "  3. Run training script:"
log_info "     ./train_full_v2_a6000.sh    # For A6000/A100 (48GB/40GB+) - RECOMMENDED"
log_info "     ./train_full_v2_rtx4090.sh  # For RTX 4090 (24GB)"
log_info "     ./train_full_v2.sh          # For full training (with grad accum)"
log_info "     ./train_10pct_v2_12gb.sh    # For 12GB GPUs"
log_info "     ./train_quick.sh            # For quick test"
echo ""
log_info "Important notes:"
log_info "  - Ensure GPU is available: nvidia-smi"
log_info "  - Check disk space for checkpoints: df -h $BASE_DIR/build"
log_info "  - Monitor training: tail -f <log_file> or use tensorboard"
log_info "  - Training scripts are already configured with correct paths"
echo ""
log_success "Setup complete! Ready for training!"
echo ""
