#!/bin/bash
# Train on the test subset dataset for quick testing
# After training, automatically runs video generation test

cd "$(dirname "$0")"

DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0_test"
OUTPUT_DIR="./checkpoints_test_subset"

echo "Starting training on test subset..."
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Use test config for fast iteration
python3 training/train.py \
    --train_data_dir "$DATA_DIR" \
    --val_data_dir "$DATA_DIR" \
    --use_test_config \
    --batch_size 4 \
    --max_steps 50 \
    --save_steps 25 \
    --eval_steps 25 \
    --logging_steps 5 \
    --output_dir "$OUTPUT_DIR"

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Training completed! Running video generation test..."
    echo ""
    
    # Run video generation test on final checkpoint
    python3 create_side_by_side_comparison.py \
        --checkpoint "$OUTPUT_DIR/checkpoint-final/pytorch_model.bin" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR/comparison_videos" \
        --tokenizer_dir /media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer \
        --num_samples 2
    
    TEST_EXIT_CODE=$?
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Pipeline complete!"
        echo "  Training checkpoints: $OUTPUT_DIR/checkpoint-*/"
        echo "  Comparison videos: $OUTPUT_DIR/comparison_videos/"
        echo "    - sample_*_comparison.mp4 (side-by-side: GT | Prediction)"
    else
        echo ""
        echo "⚠ Training succeeded but video generation test failed"
        exit $TEST_EXIT_CODE
    fi
else
    echo ""
    echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi
