#!/bin/bash
# Run iterative decoding (step 6) for multiple checkpoints from checkpoints_rtx3060
#
# This script generates comparison videos showing iterative decoding with K=2,4,8,16
# for selected checkpoints to visualize training progression.
#
# Usage:
#   ./run_iterative_decoding_checkpoints.sh [checkpoint_step1] [checkpoint_step2] ...
#
# Examples:
#   ./run_iterative_decoding_checkpoints.sh                    # Use default selection
#   ./run_iterative_decoding_checkpoints.sh 10000 30000 60000  # Specific checkpoints

set -e

# Configuration
CHECKPOINTS_DIR="/media/skr/storage/robot_world/humanoid_wm/build/checkpoints_rtx3060"
OUTPUT_BASE="/media/skr/storage/robot_world/humanoid_wm/videos/iterative_decoding_rtx3060"
DATA_DIR="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0"
TOKENIZER_DIR="/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEBUG_SCRIPT="${SCRIPT_DIR}/../helper/debug_masked_hwm_videos.py"

# Default checkpoint selection (every 10k steps + first and last)
DEFAULT_CHECKPOINTS=(1000 10000 20000 30000 40000 50000 60000 70000 73000)

# Use command line args or defaults
if [ $# -gt 0 ]; then
    CHECKPOINTS=("$@")
else
    CHECKPOINTS=("${DEFAULT_CHECKPOINTS[@]}")
fi

echo "============================================================"
echo "Iterative Decoding Comparison for RTX3060 Checkpoints"
echo "============================================================"
echo "Checkpoints dir: ${CHECKPOINTS_DIR}"
echo "Output base: ${OUTPUT_BASE}"
echo "Selected steps: ${CHECKPOINTS[*]}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_BASE}"

# Process each checkpoint
for STEP in "${CHECKPOINTS[@]}"; do
    CHECKPOINT_PATH="${CHECKPOINTS_DIR}/checkpoint-${STEP}/pytorch_model.bin"
    OUTPUT_DIR="${OUTPUT_BASE}/step${STEP}"
    
    if [ ! -f "${CHECKPOINT_PATH}" ]; then
        echo "WARNING: Checkpoint not found: ${CHECKPOINT_PATH}, skipping..."
        continue
    fi
    
    echo ""
    echo "============================================================"
    echo "Processing checkpoint-${STEP}"
    echo "============================================================"
    
    python "${DEBUG_SCRIPT}" \
        --step 6 \
        --checkpoint "${CHECKPOINT_PATH}" \
        --data_dir "${DATA_DIR}" \
        --tokenizer_dir "${TOKENIZER_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --sample_indices "0"
    
    echo "Completed checkpoint-${STEP}, output in ${OUTPUT_DIR}"
done

echo ""
echo "============================================================"
echo "All checkpoints processed!"
echo "Output directory: ${OUTPUT_BASE}"
echo "============================================================"

# List generated videos
echo ""
echo "Generated videos:"
find "${OUTPUT_BASE}" -name "*.mp4" -type f | sort
