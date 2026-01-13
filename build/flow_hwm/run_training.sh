#!/bin/bash
# Training script for Flow-HWM

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cosmos-tokenizer

# Basic training with default config
python build/flow_hwm/train.py \
    --train_data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0 \
    --val_data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0 \
    --checkpoint_dir ./checkpoints_flow_hwm \
    --batch_size 4 \
    --max_steps 60000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --logging_steps 100

# Medium config for limited GPU memory (between full and small)
# python build/flow_hwm/train.py \
#     --use_medium_config \
#     --train_data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0 \
#     --val_data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0 \
#     --checkpoint_dir ./checkpoints_flow_hwm_medium \
#     --max_steps 60000

# Small config for testing (faster, less memory)
# python build/flow_hwm/train.py \
#     --use_small_config \
#     --train_data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0 \
#     --val_data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0 \
#     --checkpoint_dir ./checkpoints_flow_hwm_small \
#     --max_steps 1000

# Test config for quick validation
# python build/flow_hwm/train.py \
#     --use_test_config \
#     --train_data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0 \
#     --val_data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0 \
#     --checkpoint_dir ./checkpoints_flow_hwm_test \
#     --max_steps 100

# Resume from checkpoint
# python build/flow_hwm/train.py \
#     --train_data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0 \
#     --val_data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0 \
#     --checkpoint_dir ./checkpoints_flow_hwm \
#     --resume_from ./checkpoints_flow_hwm/checkpoint-5000
