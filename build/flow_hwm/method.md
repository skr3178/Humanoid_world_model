Flow HWM - Methods:

- Flow matching in the continuous latent space
- Video generation frameworks and architectures
- Transformer block design

![Flow_HM](flow_hwm_transformer.png)

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

**Footnotes:**  
Implementation of 2-stage Split Attention (paper ablation variant):  
- Each stream first undergoes **independent self-attention** within its own sequence  
- After self-attention, apply **cross-attention** where future video \(v_f\) serves as queries, and keys/values come from the other streams (\(v_p\), \(a_p\), \(a_f\))



## Summary

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