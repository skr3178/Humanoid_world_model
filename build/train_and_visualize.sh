#!/bin/bash
# Complete pipeline: Train on test data, then visualize predictions

set -e

# Activate conda environment
source /home/skr/miniconda3/etc/profile.d/conda.sh
conda activate cosmos-tokenizer

cd /media/skr/storage/robot_world/humanoid_wm/build

# Paths
TRAIN_DATA="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0_test"
VAL_DATA="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0_test"
OUTPUT_DIR="./checkpoints_test"
TOKENIZER_DIR="/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer"

echo "=========================================="
echo "Step 1: Create test subsets (if needed)"
echo "=========================================="
if [ ! -d "$TRAIN_DATA" ]; then
    echo "Creating test subsets..."
    python create_test_subsets.py
else
    echo "Test subsets already exist"
fi

echo ""
echo "=========================================="
echo "Step 2: Train model on test data"
echo "=========================================="
python training/train.py \
    --train_data_dir "$TRAIN_DATA" \
    --val_data_dir "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --warmup_steps 10 \
    --max_steps 100 \
    --save_steps 50 \
    --eval_steps 50 \
    --logging_steps 10 \
    --use_test_config

echo ""
echo "=========================================="
echo "Step 3: Visualize predictions"
echo "=========================================="
FINAL_CHECKPOINT="$OUTPUT_DIR/checkpoint-final/pytorch_model.bin"

if [ ! -f "$FINAL_CHECKPOINT" ]; then
    echo "ERROR: Final checkpoint not found at $FINAL_CHECKPOINT"
    echo "Looking for latest checkpoint..."
    LATEST_CHECKPOINT=$(find "$OUTPUT_DIR" -name "pytorch_model.bin" -type f | sort -V | tail -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo "Using latest checkpoint: $LATEST_CHECKPOINT"
        FINAL_CHECKPOINT="$LATEST_CHECKPOINT"
    else
        echo "No checkpoint found. Exiting."
        exit 1
    fi
fi

python inference/visualize_predictions.py \
    --checkpoint "$FINAL_CHECKPOINT" \
    --data_dir "$VAL_DATA" \
    --output_dir "./visualizations" \
    --num_samples 5 \
    --tokenizer_dir "$TOKENIZER_DIR" \
    --use_test_config \
    --device cuda

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo "Checkpoint: $FINAL_CHECKPOINT"
echo "Visualizations: ./visualizations/"
