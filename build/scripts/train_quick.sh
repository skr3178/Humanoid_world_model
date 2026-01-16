#!/bin/bash
# Medium-to-small training script for 16-24GB GPUs
# Uses medium config (12 layers, 256 dim, 8 heads) with balanced settings

set -e

# Activate conda environment
source /home/skr/miniconda3/etc/profile.d/conda.sh
conda activate cosmos-tokenizer

cd /media/skr/storage/robot_world/humanoid_wm/build

# Data directories - use full dataset (not test subsets)
TRAIN_DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0"
VAL_DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0"
OUTPUT_DIR="./checkpoints_medium"

echo "=========================================="
echo "Medium-to-Small Training (~11GB GPU)"
echo "=========================================="
echo "Training data: $TRAIN_DATA_DIR"
echo "Validation data: $VAL_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Model config (optimized for ~11GB GPU):"
echo "  - 6 layers, 6 heads, 192 dim, 768 MLP"
echo "  - vocab_size: 65536 (per factor)"
echo "  - action_dim: 25 (R^25)"
echo "  - Sequence: 2 past + 1 future clip (per paper)"
echo "  - Mixed precision: bf16 (enabled)"
echo ""
echo "Training hyperparameters (optimized for ~11GB GPU):"
echo "  - Batch size: 1 (with grad_accum=16, effective batch=16)"
echo "  - Learning rate: 3e-5 (standard rate)"
echo "  - Max steps: 60,000"
echo "  - Warmup steps: 100"
echo "=========================================="
echo ""

# Check if data directories exist
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

# Clear GPU cache before training
python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true

# Set PyTorch memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training with medium config (optimized for ~11GB GPU)
# Uses: 6 layers, 192 dim, 6 heads, batch_size=1, grad_accum=16
python3 training/train.py \
    --train_data_dir "$TRAIN_DATA_DIR" \
    --val_data_dir "$VAL_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --use_12gb_config \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --warmup_steps 100 \
    --max_steps 60000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --logging_steps 100 \
    --seed 42

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
    echo "Checkpoints saved to: $OUTPUT_DIR/"
else
    echo ""
    echo "=========================================="
    echo "Training failed with exit code $TRAIN_EXIT_CODE"
    echo "=========================================="
    exit $TRAIN_EXIT_CODE
fi
