#!/bin/bash
# Inference script for Flow-HWM

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cosmos-tokenizer

# Basic inference
python build/flow_hwm/inference.py \
    --checkpoint ./checkpoints_flow_hwm/model-10000.pt \
    --data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0 \
    --sample_idx 0 \
    --num_steps 50 \
    --cfg_scale 1.5 \
    --output_dir ./generated_latents

# Generate multiple samples
# python build/flow_hwm/inference.py \
#     --checkpoint ./checkpoints_flow_hwm/model-10000.pt \
#     --data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0 \
#     --sample_idx 0 \
#     --num_samples 4 \
#     --num_steps 100 \
#     --cfg_scale 2.0 \
#     --output_dir ./generated_latents

# With small config
# python build/flow_hwm/inference.py \
#     --checkpoint ./checkpoints_flow_hwm_small/model-1000.pt \
#     --use_small_config \
#     --data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0 \
#     --output_dir ./generated_latents
