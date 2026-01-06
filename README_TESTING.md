# Testing Pipeline

## Overview

The training pipeline now includes automatic video generation testing at the end of training. This generates comparison videos showing:
- **Ground truth** videos (actual future frames from dataset)
- **Predicted** videos (model predictions)

## Running the Full Pipeline

### Quick Test on Subset

```bash
cd build
./train_test_subset.sh
```

This will:
1. Train the model on the test subset (50 steps)
2. Save checkpoints at steps 25 and 50
3. Automatically run video generation test on the final checkpoint
4. Save comparison videos to `checkpoints_test_subset/test_videos/`

### Manual Testing

After training, you can manually run the test:

```bash
cd build
python3 test_generate_video.py \
    --checkpoint checkpoints_test_subset/checkpoint-final/pytorch_model.bin \
    --data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0_test \
    --output_dir ./test_output \
    --tokenizer_dir /media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer \
    --num_samples 3 \
    --use_test_config
```

## Output Files

After running the test, you'll find:

```
test_output/
├── sample_0_comparison.mp4      # Side-by-side: Ground Truth | Predicted
├── sample_1_comparison.mp4      # Side-by-side: Ground Truth | Predicted
└── ...
```

Each comparison video shows:
- **Left side**: Ground truth (actual future frames from dataset)
- **Right side**: Predicted (model predictions)
- **Resolution**: 512×256 (256×256 for each side)
- **Frames**: 17 frames per video (one clip)
- **Labels**: Text overlays showing "Ground Truth" and "Predicted"

## Understanding the Videos

- **Side-by-side comparison**: Left shows ground truth, right shows predictions
- **Ground truth (left)**: Actual future frames from the dataset
- **Predicted (right)**: What the model predicted given past frames and actions
- Each video contains 17 frames (one clip worth of frames)
- Videos are at 30 FPS, 512×256 resolution (256×256 per side)

## What to Look For

1. **Visual quality**: Do predicted frames look reasonable?
2. **Temporal consistency**: Do frames flow smoothly?
3. **Action following**: Does the predicted video follow the given actions?
4. **Comparison**: How close are predictions to ground truth?

## Troubleshooting

If video generation fails:
- Check that `cosmos_tokenizer` is installed and accessible
- Verify tokenizer path is correct
- Ensure you're using the correct config (--use_test_config if training with test config)
- Check that checkpoints were saved successfully
