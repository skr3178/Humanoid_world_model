#!/bin/bash
# Quick test script for Flow-HWM

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cosmos-tokenizer

# Run test
cd /media/skr/storage/robot_world/humanoid_wm/build/flow_hwm
python3 test_model.py
