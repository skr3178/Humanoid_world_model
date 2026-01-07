#!/bin/bash
# Minimal training script for testing model architecture
# Designed to run quickly on small GPUs (4-6GB) to verify:
# - Model architecture works correctly
# - Loss decreases over time
# - Input/output shapes are correct
# - No memory issues

cd "$(dirname "$0")"

TRAIN_DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0"
VAL_DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0"
OUTPUT_DIR="./checkpoints_minimal_test"

echo "=========================================="
echo "Minimal Training Test - Architecture Verification"
echo "=========================================="
echo "Training data: $TRAIN_DATA_DIR"
echo "Validation data: $VAL_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Model config (MINIMAL for testing):"
echo "  - 4 layers (minimal for testing)"
echo "  - 4 heads (128/4 = 32 dim per head)"
echo "  - 128 dim (minimal for testing)"
echo "  - 512 MLP (scaled with d_model)"
echo "  - 1 past clip + 1 future clip (reduced from 2+1)"
echo "  - vocab_size: 65536 (per factor) - UNCHANGED"
echo "  - action_dim: 25 (R^25) - UNCHANGED"
echo ""
echo "Training hyperparameters:"
echo "  - Batch size: 1 (with grad_accum=16, effective batch=16)"
echo "  - Learning rate: 1e-4"
echo "  - Max steps: 100 (quick test)"
echo "  - Warmup steps: 10"
echo "  - Corruption rate: 0.2 (per paper)"
echo "  - Save steps: 50"
echo "  - Eval steps: 50"
echo "  - Logging steps: 10"
echo ""
echo "Purpose: Verify architecture works before scaling up"
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

# First, inspect the model
echo "Step 1: Inspecting model architecture..."
python3 inspect_model.py

if [ $? -ne 0 ]; then
    echo "ERROR: Model inspection failed"
    exit 1
fi

echo ""
echo "Step 2: Starting minimal training test..."
echo ""

# Train with minimal config
python3 training/train.py \
    --train_data_dir "$TRAIN_DATA_DIR" \
    --val_data_dir "$VAL_DATA_DIR" \
    --use_minimal_config \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --warmup_steps 10 \
    --max_steps 100 \
    --save_steps 50 \
    --eval_steps 50 \
    --logging_steps 10 \
    --output_dir "$OUTPUT_DIR" \
    --seed 42

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Minimal training test completed successfully!"
    echo "=========================================="
    echo ""
    echo "What to check:"
    echo "  1. Loss should start around ~11.09 (log(65536)) and decrease"
    echo "  2. No NaN/Inf values in loss"
    echo "  3. Gradient norms should be reasonable (< 10)"
    echo "  4. Memory usage should be < 6GB"
    echo ""
    echo "If all checks pass, you can scale up to:"
    echo "  - config_12gb.py for 12GB GPU"
    echo "  - Full config for larger GPUs"
    echo ""
    echo "Checkpoints saved to: $OUTPUT_DIR/"
else
    echo ""
    echo "=========================================="
    echo "✗ Training test failed with exit code $TRAIN_EXIT_CODE"
    echo "=========================================="
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check GPU memory: nvidia-smi"
    echo "  2. Check logs for specific error"
    echo "  3. Try reducing batch_size further or using CPU"
    exit $TRAIN_EXIT_CODE
fi
