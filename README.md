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
