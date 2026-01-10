#!/bin/bash
# =============================================================================
# Train Masked-HWM on Full v2.0 Dataset - RTX 4090 Optimized (24GB)
# =============================================================================
# GPU Target: RTX 4090 (24GB VRAM) or similar
#
# Optimizations for 24GB GPU:
#   - Full model architecture (24 layers, 512 dim, 8 heads) - UNCHANGED
#   - Batch size: 8 (reduced from 16 to fit in 24GB)
#   - Gradient accumulation: 2 steps (effective batch size = 16)
#   - Mixed precision training (bf16)
#   - Gradient checkpointing enabled
#
# This configuration maintains the paper's effective batch size and model
# architecture while fitting within 24GB VRAM constraints.
# =============================================================================

cd "$(dirname "$0")"

# =============================================================================
# Configuration
# =============================================================================

TRAIN_DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0"
VAL_DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0"
TEST_DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/test_v2.0"
OUTPUT_DIR="./checkpoints_full_v2_rtx4090"

# =============================================================================
# Print Configuration
# =============================================================================

echo "=========================================="
echo "Training Masked-HWM on Full v2.0 Dataset"
echo "RTX 4090 Optimized Configuration (24GB)"
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
echo "Memory Optimizations for RTX 4090:"
echo "  - Batch size: 8 (reduced from 16)"
echo "  - Gradient accumulation: 2 (effective batch = 16)"
echo "  - Mixed precision: bf16"
echo "  - Gradient checkpointing: enabled"
echo ""
echo "Training hyperparameters (per paper):"
echo "  - Learning rate: 3e-5 (linear decay after warmup)"
echo "  - Max steps: 60,000"
echo "  - Warmup steps: 100"
echo "  - Optimizer: AdamW"
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

    # Get available memory
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
    if [ "$FREE_MEM" -lt 20000 ]; then
        echo "WARNING: Low GPU memory available (${FREE_MEM}MB)."
        echo "Recommended: At least 20GB free for training."
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

echo "Starting training..."
echo ""

# Train with RTX 4090 optimizations
# Uses full MaskedHWMConfig (24 layers, 512 dim, 8 heads)
# with reduced batch size and gradient accumulation
python3 training/train.py \
    --train_data_dir "$TRAIN_DATA_DIR" \
    --val_data_dir "$VAL_DATA_DIR" \
    --test_data_dir "$TEST_DATA_DIR" \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --use_gradient_checkpointing \
    --mixed_precision bf16 \
    --learning_rate 3e-5 \
    --warmup_steps 100 \
    --max_steps 60000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --logging_steps 100 \
    --output_dir "$OUTPUT_DIR" \
    --seed 42

TRAIN_EXIT_CODE=$?

# =============================================================================
# Post-Training Summary
# =============================================================================

echo ""
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
    echo "Checkpoints saved to: $OUTPUT_DIR/"
    echo "  - checkpoint-1000/"
    echo "  - checkpoint-2000/"
    echo "  - ..."
    echo "  - checkpoint-final/"
    echo ""
    echo "To generate comparison videos, run:"
    echo "  conda activate cosmos-tokenizer"
    echo "  python3 create_side_by_side_comparison.py \\"
    echo "    --checkpoint $OUTPUT_DIR/checkpoint-final/pytorch_model.bin \\"
    echo "    --data_dir $VAL_DATA_DIR \\"
    echo "    --output_dir $OUTPUT_DIR/comparison_videos \\"
    echo "    --num_samples 5"
    echo ""
    echo "Training statistics:"
    echo "  - Total steps: 60,000"
    echo "  - Effective batch size: 16 (8 x 2 accumulation)"
    echo "  - Model parameters: ~338M (24 layers, 512 dim)"
    echo ""
else
    echo "=========================================="
    echo "Training failed with exit code $TRAIN_EXIT_CODE"
    echo "=========================================="
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check GPU memory: nvidia-smi"
    echo "  2. Check logs in: $OUTPUT_DIR/logs/"
    echo "  3. Verify data paths are correct"
    echo "  4. Try reducing batch size further (--batch_size 4 --gradient_accumulation_steps 4)"
    echo ""
    exit $TRAIN_EXIT_CODE
fi
