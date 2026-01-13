### 2.3. Flow Matching for Video Generation

We train the Flow-HWM variant using the Flow Matching (FM) framework (Lipman et al., 2023; Albergo et al., 2023; Liu et al., 2022). The framework formulates video generation as a continuous transformation of samples from a simple prior distribution (Gaussian noise) into data samples drawn from the target distribution. As opposed to learning a reversed stochastic process like in traditional diffusion models (Song et al., 2021), FM directly learns a time-dependent velocity field that drives this transformation.

Let $X_t$ denote a video sample in the latent space, and let $X_0 \sim \mathcal{N}(0, I)$ represent a random sample from the Gaussian prior. We train the model by sampling an intermediate time $t \in [0, 1]$ and construct a point along the trajectory $X_t$ using linear interpolation:

$$
X_t = tX_1 + (1 - (1 - \sigma_{\min})t)X_0, \quad (1)
$$

where $\sigma_{\min}$ is a small positive constant ensuring non-zero support at $t = 1$. The ground-truth velocity of the transformation path is then given by the time derivative:

$$
V_t = \frac{dX_t}{dt} = X_1 - (1 - \sigma_{\min})X_0. \quad (2)
$$

Our model, parameterized by $\theta$, predicts the instantaneous velocity field $u_{\theta}(X_t, \mathbf{P}, t)$ conditioned on the past video frames $v_p$, past actions $a_p$, future actions $a_f$, and time $t$, where $\mathbf{P} = \{v_p, a_p, a_f\}$ represents the conditioning context. The training objective of the model is to minimize the expected mean squared error between the predicted and ground-truth velocity:

$$
\mathbb{E}_{t,X_0,X_1,a_p,a_f,v_p} \left[ \|u_{\theta}(X_t, a_p, a_f, v_p, t) - V_t\|^2 \right]. \quad (3)
$$

We adopt classifier-free guidance (Ho & Salimans, 2022) to improve conditional generation by enabling the model to better balance conditioning signals from actions and past context during training and inference. During inference, generation proceeds by integrating the learned velocity field from $t = 0$ to $t = 1$, starting from pure Gaussian noise and employing the first-order Euler ODE solver.

#### 2.3.1. Architecture

**Tokenization.** We tokenize the compressed latent video frames $I_p$ and $I_f$ by dividing them into $p_{lw} \times p_{lw}$ spatial and $p_t$ temporal segments per token. Each token is projected to $h$ channels via a convolutional layer. Action sequences $a_p$ and $a_f$ are embedded using an MLP into the same $h$-dimensional space. The timestep $t$ is encoded using sinusoidal embeddings following DDPM (Ho et al., 2020).

**Transformer.** After tokenization, each of the four streams of tokens ($v_f, v_p, a_f, a_p$) are kept separate and processed by $d$ transformer blocks sequentially. In the final layer, we apply time modulation as described in DDPM, followed by a linear projection of the future tokens $v_f$ from $h$ dimensions back to $l$ latent dimensions. The resulting tokens are then reshaped into the original video's spatiotemporal format to be decoded back to pixel space by the VAE. We detail the transformer block design in Section 2.4.
