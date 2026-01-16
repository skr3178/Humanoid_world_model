#!/bin/bash
# Train on the 10% v2.0 dataset with 12GB GPU configuration
# Uses increased model size (10 layers, 256 dim) for better learning capacity

cd "$(dirname "$0")"

TRAIN_DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0"
VAL_DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0"
TEST_DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/test_v2.0"
OUTPUT_DIR="./checkpoints_full_v2_12gb"

echo "=========================================="
echo "Training Masked-HWM on 10% v2.0 Dataset"
echo "Using 12GB GPU Configuration"
echo "=========================================="
echo "Training data: $TRAIN_DATA_DIR"
echo "Validation data: $VAL_DATA_DIR"
echo "Test data: $TEST_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Model config (12GB GPU):"
echo "  - 6 layers (25% of full 24-layer model)"
echo "  - 4 heads (192/4 = 48 dim per head)"
echo "  - 192 dim (37.5% of full 512-dim model)"
echo "  - 768 MLP (scaled with d_model, mlp_ratio=4.0)"
echo "  - vocab_size: 65536 (per factor) - UNCHANGED"
echo "  - action_dim: 25 (R^25) - UNCHANGED"
echo ""
echo "Training hyperparameters:"
echo "  - Batch size: 4 (with grad_accum=4, effective batch=16)"
echo "  - Learning rate: 3e-5 (paper value, linear decay after warmup)"
echo "  - Max steps: 60,000"
echo "  - Warmup steps: 500"
echo "  - Corruption rate: 0.2 (per paper, reduced from 0.5)"
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

# Train with 12GB config
# Uses MaskedHWM12GBConfig (6 layers, 192 dim, 4 heads)
python3 training/train.py \
    --train_data_dir "$TRAIN_DATA_DIR" \
    --val_data_dir "$VAL_DATA_DIR" \
    --test_data_dir "$TEST_DATA_DIR" \
    --use_12gb_config \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --warmup_steps 500 \
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
