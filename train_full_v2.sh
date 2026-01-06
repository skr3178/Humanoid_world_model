#!/bin/bash
# Train on the full v2.0 dataset with full model configuration
# Matches paper specifications: 24 layers, 8 heads, 512 dim, 60k steps

cd "$(dirname "$0")"

TRAIN_DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0"
VAL_DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0"
OUTPUT_DIR="./checkpoints_full_v2"

echo "=========================================="
echo "Training Masked-HWM on Full v2.0 Dataset"
echo "=========================================="
echo "Training data: $TRAIN_DATA_DIR"
echo "Validation data: $VAL_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Model config:"
echo "  - 24 layers, 8 heads, 512 dim, 2048 MLP"
echo "  - vocab_size: 65536 (per factor)"
echo "  - action_dim: 25 (R^25)"
echo ""
echo "Training hyperparameters (per paper):"
echo "  - Batch size: 4 (with grad_accum=4, effective batch=16)"
echo "  - Learning rate: 3e-5 (linear decay after warmup)"
echo "  - Max steps: 60,000"
echo "  - Warmup steps: 100"
echo "  - Optimizer: AdamW"
echo "  - Save steps: 1,000"
echo "  - Eval steps: 1,000"
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

# Train with full config (not test config)
# Uses MaskedHWMConfig by default (24 layers, 512 dim, etc.)
# Per paper: batch size 16, linear decay scheduler
# Note: Reduced batch size to 4 with gradient_accumulation_steps=4 to fit in 11GB GPU
#       This gives effective batch size of 16 (4 * 4 = 16)
python3 training/train.py \
    --train_data_dir "$TRAIN_DATA_DIR" \
    --val_data_dir "$VAL_DATA_DIR" \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --warmup_steps 100 \
    --max_steps 60000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --logging_steps 100 \
    --output_dir "$OUTPUT_DIR" \
    --seed 42

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
    echo "  conda run -n cosmos-tokenizer python3 create_side_by_side_comparison.py \\"
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
