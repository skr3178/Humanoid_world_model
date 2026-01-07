# Training and Visualization Guide

Complete guide for training on test data and visualizing actual vs predicted frames.

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
cd /media/skr/storage/robot_world/humanoid_wm/build

# Activate conda environment
source /home/skr/miniconda3/etc/profile.d/conda.sh
conda activate cosmos-tokenizer

# Run complete pipeline
bash train_and_visualize.sh
```

This will:
1. Create test subsets (if needed)
2. Train model on test data
3. Save final checkpoint
4. Generate frame-by-frame visualizations

---

### Option 2: Run Steps Manually

#### Step 1: Create Test Subsets

```bash
cd /media/skr/storage/robot_world/humanoid_wm/build
source /home/skr/miniconda3/etc/profile.d/conda.sh
conda activate cosmos-tokenizer

python create_test_subsets.py
```

This creates:
- `/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0_test`
- `/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0_test`

#### Step 2: Train Model

```bash
cd /media/skr/storage/robot_world/humanoid_wm/build
source /home/skr/miniconda3/etc/profile.d/conda.sh
conda activate cosmos-tokenizer

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
```

**Output:**
- Checkpoints saved to `./checkpoints_test/checkpoint-{step}/`
- **Final checkpoint:** `./checkpoints_test/checkpoint-final/pytorch_model.bin`

#### Step 3: Visualize Predictions

```bash
cd /media/skr/storage/robot_world/humanoid_wm/build
source /home/skr/miniconda3/etc/profile.d/conda.sh
conda activate cosmos-tokenizer

python inference/visualize_predictions.py \
    --checkpoint ./checkpoints_test/checkpoint-final/pytorch_model.bin \
    --data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0_test \
    --output_dir ./visualizations \
    --num_samples 5 \
    --tokenizer_dir /media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer \
    --use_test_config \
    --device cuda
```

**Output:**
- Comparison grids: `./visualizations/sample_{N}_comparison.png`
- Individual frames: `./visualizations/sample_{N}_frames/frame_{T:02d}_comparison.png`
- Separate actual/predicted: `./visualizations/sample_{N}_frames/frame_{T:02d}_actual.png` and `frame_{T:02d}_predicted.png`

---

## Training Configuration

### Test Config (Quick Training)
- **Layers:** 4 (instead of 24)
- **Dimensions:** 128 (instead of 512)
- **Vocab:** 1024 per factor (instead of 65536)
- **Steps:** 100 (instead of 60000)
- **Time:** ~5 minutes

### Full Config (Production Training)
Remove `--use_test_config` flag and use:
- **Layers:** 24
- **Dimensions:** 512
- **Vocab:** 65536 per factor
- **Steps:** 60000
- **Time:** Several hours/days

---

## Visualization Output

For each sample, the script generates:

1. **Comparison Grid** (`sample_{N}_comparison.png`):
   - Grid showing all frames
   - Top row: Actual frames
   - Bottom row: Predicted frames

2. **Individual Frame Comparisons** (`sample_{N}_frames/`):
   - `frame_{T:02d}_comparison.png`: Side-by-side actual vs predicted
   - `frame_{T:02d}_actual.png`: Actual frame only
   - `frame_{T:02d}_predicted.png`: Predicted frame only

---

## Troubleshooting

### Checkpoint Not Found
If final checkpoint doesn't exist, the script will use the latest checkpoint:
```bash
find ./checkpoints_test -name "pytorch_model.bin" -type f | sort -V | tail -1
```

### Tokenizer Decoding Issues
If video decoding fails, the script falls back to token-level visualization (heatmaps showing token values).

### Memory Issues
- Reduce `--batch_size` (default: 4)
- Reduce `--num_samples` (default: 5)
- Use CPU: `--device cpu` (slower but uses less memory)

---

## File Structure

```
build/
├── checkpoints_test/
│   ├── checkpoint-50/
│   │   └── pytorch_model.bin
│   ├── checkpoint-100/
│   │   └── pytorch_model.bin
│   └── checkpoint-final/          ← Final checkpoint
│       └── pytorch_model.bin
├── visualizations/
│   ├── sample_1_comparison.png
│   ├── sample_1_frames/
│   │   ├── frame_00_comparison.png
│   │   ├── frame_00_actual.png
│   │   ├── frame_00_predicted.png
│   │   └── ...
│   └── ...
└── train_and_visualize.sh
```

---

## Next Steps

After visualization, you can:
1. Analyze prediction quality frame-by-frame
2. Compare different checkpoints
3. Adjust training hyperparameters
4. Scale up to full dataset training
