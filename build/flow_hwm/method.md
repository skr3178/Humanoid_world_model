Flow HWM - Methods:

- Flow matching in the continuous latent space
- Video generation frameworks and architectures
- Transformer block design

![Flow_HM](flow_hwm_transformer.png)

## Tokenizer method:

The Flow-HWM (Flow-Matching variant) in the "Humanoid World Models" paper uses continuous latent space modeling because it is fundamentally a deterministic continuous-time flow model- which operates naturally in continuous (not discrete) representation spaces — unlike the discrete token-based Masked-HWM.

This works best in a continuous vector space (real-valued latents), because:

Discrete tokens (finite codebook) would force jumps/discontinuities → poor velocity field learning.
Continuous latents allow smooth, differentiable trajectories → stable ODE integration during sampling.

They use Cosmos Continuous 8x16x16 tokenizer (continuous VAE latents, not discrete VQ-VAE), which provides:

16× spatial compression (256→16)
8× temporal compression
Real-valued latent vectors (not quantized)

This tokenizer compresses 256x256 frames to 16x16 latents and reduces time by 8x, enabling joint attention with larger flow-matching models than Masked-HWM.

This enables Flow Matching's ODE solver (Euler integration from noise to data) to generate smooth, high-fidelity future frames conditioned on actions and past context.

### Architecture

**Tokenization**  
Compressed latent video frames \(L_p\), \(L_f\) are divided into \(p_{lw} \times p_{lw}\) spatial and \(p_t\) temporal segments per token.  
Each token is projected to \(h\) channels via a convolutional layer.  
Action sequences \(a_p\) and \(a_f\) are embedded into the same \(h\)-dimensional space using an MLP.  
Timestep \(t\) is encoded using sinusoidal embeddings (DDPM-style).

**Transformer (Base Block – Paper-Aligned)**

Each token stream (past/future video and actions) uses **separate weights** for:
- Timestep modulation
- QKV projection
- Feedforward MLPs

**Joint Attention** integrates all streams (no spatial pooling).  
RoPE is applied by modality: **3D RoPE** for video token patches (captures spatial + temporal), **1D RoPE** for action tokens.

**Step 1** (on all streams individually):  
- Modulate with timestep using learned scale (\(\alpha_0\)) and shift (\(\beta_0\)) parameters  
- Apply stream-specific QKV projection  
- Concatenate past and future tokens separately for video and actions  
- Apply RoPE to queries and keys (3D for video patches, 1D for actions across time)

**Step 2**  
Joint self-attention across all concatenated patch tokens + action tokens (pure Base Block with **joint attention** and **no parameter sharing**).  
This connects all streams/channels together, preserving per-patch spatial information (matches Figure 3 caption).

**Step 3** (on all streams individually):  
- Feedforward layers modulated with timestep (using new parameters \(\alpha_1\), \(\beta_1\))  
- Apply stream-specific MLP  
- Final rescaling with \(\gamma_1\)

**Residual connections:**
1. From before Step 1 (pre-modulation) → after joint attention  
2. From after joint attention (Step 2) → after feedforward + timestep scaling (Step 3)

**Notes**  
- Keep video tokens at **patch resolution** (no mean-pooling before joint attention).  
- If VRAM is tight → add parameter sharing (Full or Modality) in deeper layers (paper shows minimal quality drop with 33–53% size reduction).  
- For best performance on Flow-HWM → consider testing the 2-stage Split Attention variant (paper's top ablation).

### Key Differences from Masked-HWM

- No Copilot-4D-style corruption or random token replacement (Masked-HWM uses U(0, 0.2) rate on latents).
- No masking — Flow-HWM operates in continuous latent space (Cosmos Continuous VAE), so no discrete token masking or cosine schedule.
- Noise is purely the Gaussian prior — no timestep-dependent noise scheduling beyond the linear interpolation path and small \(\sigma_{\min}\).

**Footnotes:**  
Implementation of 2-stage Split Attention (paper ablation variant):  
- Each stream first undergoes **independent self-attention** within its own sequence  
- After self-attention, apply **cross-attention** where future video \(v_f\) serves as queries, and keys/values come from the other streams (\(v_p\), \(a_p\), \(a_f\))



## Summary

Cosmos Continuous 8x16x16 tokenizer: 16x spatial compression (256x256 to 16x16) + 8x temporal compression. More spatial compression than Masked-HWM enables joint attention with larger flow-matching models. Training uses d=17 transformer layers, h=1172 tokens, AdamW, lr 1e-4, cosine LR schedule, batch size 128, 150k steps on 2x NVIDIA A6000. Patch sizes: p_lw=2, p_t=1. Final linear layers use Xavier init (zero-init was unstable). No LR warmup. Inference uses 50 denoising steps with classifier-free guidance 3.0.

Step 1: AdaLN (α₀, β₀) → QKV → Concat → RoPE
         ↓
Step 2: Joint Attention → Output Proj → γ₀ rescaling → Residual 1 ✓
         ↓
Step 3: AdaLN2 (α₁, β₁) → MLP → γ₁ rescaling → Residual 2 ✓


Joint attention over all tokens (no pooling)
γ₀ rescaling after attention
Residual during attention (after attention + γ₀)
α₁, β₁ modulation before feedforward
Stream-specific MLP
γ₁ rescaling after MLP
Residual during feedforward (after MLP + γ₁)s



Use command

```
nohup stdbuf -oL -eL conda run --no-capture-output -n cosmos-tokenizer python -u /media/skr/storage/robot_world/humanoid_wm/build/flow_hwm/train.py --use_medium_config --checkpoint_dir /media/skr/storage/robot_world/humanoid_wm/checkpoints_flow_hwm_medium --train_data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0 --val_data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0 > /media/skr/storage/robot_world/humanoid_wm/flow_hwm_medium_train.log 2>&1 &


```