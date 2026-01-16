#!/bin/bash
# Train FlowHWM with fixed noise_std and sigma_min
# Run with: bash build/scripts/train_flow_hwm_fixed.sh

cd /media/skr/storage/robot_world/humanoid_wm/build

# Create logs directory if it doesn't exist
mkdir -p ../logs

nohup python -u -m flow_hwm.train \
    --use_medium_config \
    --checkpoint_dir ../checkpoints_flow_hwm_fixed \
    > ../logs/train_flow_hwm_fixed.log 2>&1 &

echo "Training started with PID: $!"
echo "Log: /media/skr/storage/robot_world/humanoid_wm/logs/train_flow_hwm_fixed.log"
echo "Monitor with: tail -f logs/train_flow_hwm_fixed.log"
