#!/bin/bash
# =============================================================================
# Train Masked-HWM on Full v2.0 Dataset - A6000/A100 Optimized (48GB/40GB+)
# =============================================================================
# GPU Target: NVIDIA A6000 (48GB) or A100 (40GB/80GB)
#
# Full paper configuration - no memory optimizations needed:
#   - Full model architecture (24 layers, 512 dim, 8 heads) - UNCHANGED
#   - Batch size: 16 (exact paper specification)
#   - No gradient accumulation needed
#   - Optional: bf16 mixed precision for faster training
#   - Optional: Gradient checkpointing disabled (not needed with 48GB+)
#
# This matches the original paper training setup exactly.
# =============================================================================

cd "$(dirname "$0")"

# =============================================================================
# Configuration
# =============================================================================

TRAIN_DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0"
VAL_DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0"
TEST_DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/test_v2.0"
OUTPUT_DIR="./checkpoints_full_v2_a6000"

# =============================================================================
# Print Configuration
# =============================================================================

echo "=========================================="
echo "Training Masked-HWM on Full v2.0 Dataset"
echo "A6000/A100 Configuration (48GB/40GB+)"
echo "=========================================="
echo "Training data: $TRAIN_DATA_DIR"
echo "Validation data: $VAL_DATA_DIR"
echo "Test data: $TEST_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Model config (Full - Paper Spec):"
echo "  - 24 layers (UNCHANGED)"
echo "  - 8 heads (UNCHANGED)"
echo "  - 512 dim (UNCHANGED)"
echo "  - 2048 MLP (UNCHANGED)"
echo "  - vocab_size: 65536 (per factor) - UNCHANGED"
echo "  - action_dim: 25 (R^25) - UNCHANGED"
echo ""
echo "Training hyperparameters (per paper):"
echo "  - Batch size: 16 (exact paper specification)"
echo "  - Gradient accumulation: 1 (not needed with 48GB+)"
echo "  - Learning rate: 3e-5 (linear decay after warmup)"
echo "  - Max steps: 60,000"
echo "  - Warmup steps: 100"
echo "  - Optimizer: AdamW"
echo "  - Mixed precision: bf16 (optional, for faster training)"
echo "  - Gradient checkpointing: disabled (not needed)"
echo "  - Save steps: 1,000"
echo "  - Eval steps: 1,000"
echo "=========================================="
echo ""

# =============================================================================
# Verify Data Directories
# =============================================================================

if [ ! -d "$TRAIN_DATA_DIR" ]; then
    echo "ERROR: Training data directory not found: $TRAIN_DATA_DIR"
    exit 1
fi

if [ ! -d "$VAL_DATA_DIR" ]; then
    echo "ERROR: Validation data directory not found: $VAL_DATA_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# GPU Memory Check
# =============================================================================

echo "Checking GPU status..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
    
    # Get GPU name and memory
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
    
    echo "GPU: $GPU_NAME"
    echo "Total Memory: ${TOTAL_MEM}MB"
    echo "Free Memory: ${FREE_MEM}MB"
    echo ""
    
    # Check if it's A6000 or A100
    if [[ "$GPU_NAME" == *"A6000"* ]] || [[ "$GPU_NAME" == *"A100"* ]]; then
        echo "✓ Detected high-end GPU ($GPU_NAME)"
        echo "  Using full paper configuration with batch_size=16"
    else
        echo "⚠ Warning: GPU may not be A6000/A100"
        if [ "$TOTAL_MEM" -lt 40000 ]; then
            echo "  GPU has less than 40GB. Consider using train_full_v2_rtx4090.sh instead."
            read -p "Continue anyway? (y/n): " CONTINUE
            if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
    
    if [ "$FREE_MEM" -lt 35000 ]; then
        echo "⚠ Warning: Low GPU memory available (${FREE_MEM}MB)."
        echo "  Recommended: At least 35GB free for optimal training."
        read -p "Continue anyway? (y/n): " CONTINUE
        if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "WARNING: nvidia-smi not found. Cannot check GPU status."
fi

echo ""

# =============================================================================
# Training Command
# =============================================================================

echo "Starting training with full paper configuration..."
echo ""

# Train with full config - exact paper specification
# Uses MaskedHWMConfig by default (24 layers, 512 dim, etc.)
# Batch size 16 as per paper (no gradient accumulation needed)
python3 training/train.py \
    --train_data_dir "$TRAIN_DATA_DIR" \
    --val_data_dir "$VAL_DATA_DIR" \
    --batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --warmup_steps 100 \
    --max_steps 60000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --logging_steps 100 \
    --output_dir "$OUTPUT_DIR" \
    --seed 42 \
    --mixed_precision bf16

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Training completed successfully!"
    echo "=========================================="
    echo "Checkpoints saved to: $OUTPUT_DIR/"
    echo "  - checkpoint-1000/"
    echo "  - checkpoint-2000/"
    echo "  - ..."
    echo "  - checkpoint-final/"
    echo ""
    echo "To generate comparison videos, run:"
    echo "  python3 create_side_by_side_comparison.py \\"
    echo "    --checkpoint $OUTPUT_DIR/checkpoint-final/pytorch_model.bin \\"
    echo "    --data_dir $VAL_DATA_DIR \\"
    echo "    --output_dir $OUTPUT_DIR/comparison_videos \\"
    echo "    --num_samples 5"
else
    echo ""
    echo "=========================================="
    echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
    echo "=========================================="
    exit $TRAIN_EXIT_CODE
fi
