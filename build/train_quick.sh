#!/bin/bash
# Quick training script for testing (~5 minutes)
# Uses reduced model (4 layers, 128 dim, 8K vocab) for fast iteration

set -e

# Activate conda environment
source /home/skr/miniconda3/etc/profile.d/conda.sh
conda activate cosmos-tokenizer

cd /media/skr/storage/robot_world/humanoid_wm/build

# Step 1: Create test subsets (if not exist)
if [ ! -d "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0_test" ]; then
    echo "Creating test subsets..."
    python create_test_subsets.py
fi

# Step 2: Run quick training (~5 min with 100 steps)
echo "Starting quick training (100 steps, ~5 minutes)..."
python training/train.py \
    --train_data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0_test \
    --val_data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0_test \
    --output_dir ./checkpoints_test \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --warmup_steps 10 \
    --max_steps 100 \
    --save_steps 50 \
    --eval_steps 50 \
    --logging_steps 10 \
    --use_test_config

echo "Quick training complete! Checkpoint saved to ./checkpoints_test/"
