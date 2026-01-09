#!/bin/bash
# Quick training script optimized for <8GB GPU
# Uses ultra-minimal config (3 layers, 96 dim, 4 heads) with memory optimizations

set -e

# Activate conda environment
source /home/skr/miniconda3/etc/profile.d/conda.sh
conda activate cosmos-tokenizer

cd /media/skr/storage/robot_world/humanoid_wm/build

# Data directories - use full dataset (not test subsets)
TRAIN_DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0"
VAL_DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0"
OUTPUT_DIR="./checkpoints_minimal"

echo "=========================================="
echo "Quick Training - Ultra-Minimal Config (<8GB GPU)"
echo "=========================================="
echo "Training data: $TRAIN_DATA_DIR"
echo "Validation data: $VAL_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Model config (optimized for <8GB GPU):"
echo "  - 3 layers, 4 heads, 96 dim, 384 MLP"
echo "  - vocab_size: 65536 (per factor)"
echo "  - action_dim: 25 (R^25)"
echo "  - Sequence: 2 past + 1 future clip (per paper)"
echo "  - Mixed precision: bf16 (enabled)"
echo ""
echo "Training hyperparameters (AGGRESSIVE for faster convergence):"
echo "  - Batch size: 2 (with grad_accum=8, effective batch=16)"
echo "  - Learning rate: 5e-4 (5x higher for small model)"
echo "  - Max steps: 60,000"
echo "  - Warmup steps: 50 (short warmup)"
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

# Run training with ultra-minimal config
# Uses: 3 layers, 96 dim, 4 heads, batch_size=2, grad_accum=8
python3 training/train.py \
    --train_data_dir "$TRAIN_DATA_DIR" \
    --val_data_dir "$VAL_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --use_minimal_config \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-4 \
    --warmup_steps 50 \
    --max_steps 60000 \
    --save_steps 500 \
    --eval_steps 500 \
    --logging_steps 50 \
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
