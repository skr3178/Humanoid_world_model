# Masked-HWM: Masked Humanoid World Model with Shared Parameters

Implementation of Masked-HWM (Masked Humanoid World Model) with shared parameters, following the specifications from the Humanoid World Models paper.

## Flow Matching Model Output

### Prediction
![Prediction](videos/debug_flowhwm_ckpt24000/sample_0/prediction.gif)

### Noise Schedule
![Noise Schedule](videos/debug_flowhwm_ckpt24000/sample_0/noise_schedule.gif)

### Comparison
![Comparison](videos/debug_flowhwm_ckpt24000/sample_0/comparison.gif)

## Masked Flow Model Output

Output from the Masked model:

![Masked Flow Matching](ScreenRecording2026-01-17at10.43.53AM-ezgif.com-video-to-gif-converter.gif)

## Loss Curve Comparison

> **Note:** The model size is much smaller than the paper implementation due to limited GPU access.

### Author Implementation

#### Flow HWM Loss
![Flow HWM Loss](build/Loss_curves_author/Author/Loss_Flow_HWM.png)

#### Experimental Flow Loss
![Experimental Flow Loss](build/Loss_curves_author/run/loss_curves_Flow_HWM.png)

#### Masked HWM Loss
![Masked HWM Loss](build/Loss_curves_author/Author/Masked_HWM_Loss1.png)

#### Experimental Mask Loss
![Experimental Mask Loss](build/Loss_curves_author/run/Masked_HWM_loss.png)

## Pre-trained Checkpoints

Pre-trained model checkpoints are available on Hugging Face: [sangramrout/Humanoid_WM](https://huggingface.co/sangramrout/Humanoid_WM)

### Masked HWM Checkpoint

- **Checkpoint**: `checkpoint-60000`
- **Location**: `Masked HWM/checkpoint-60000/`
- **Download**: Available on Hugging Face repository

### Flow HWM Checkpoint

- **Checkpoint**: `checkpoint-24000`
- **Location**: `checkpoints_flow_hwm_medium/checkpoint-24000/`
- **Download**: Available on Hugging Face repository

### Downloading Checkpoints

You can download the checkpoints using the Hugging Face Hub:

```python
from huggingface_hub import snapshot_download

# Download Masked HWM checkpoint
masked_checkpoint = snapshot_download(
    repo_id="sangramrout/Humanoid_WM",
    allow_patterns="Masked HWM/checkpoint-60000/*"
)

# Download Flow HWM checkpoint
flow_checkpoint = snapshot_download(
    repo_id="sangramrout/Humanoid_WM",
    allow_patterns="checkpoints_flow_hwm_medium/checkpoint-24000/*"
)
```

Or using the command line:

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Download using CLI
huggingface-cli download sangramrout/Humanoid_WM --local-dir ./checkpoints
```
