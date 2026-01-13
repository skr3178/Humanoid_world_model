# Flow Matching: Training, Inference, and Visualization Flow

## Overview
This document shows the complete flow of data during training, inference, and visualization in the flow matching implementation.

---

## 1. TRAINING FLOW

### Step-by-Step Process

```
┌─────────────────────────────────────────────────────────────┐
│ TRAINING LOOP (main.py lines 19-27)                         │
│ - n_steps iterations (e.g., 1500 steps)                     │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌───────────────────────────────────────────────────────────── ┐
│ Step 1: Generate Dynamic Conditioning                        │
│ (utils.py: generate_dynamic_conditioning)                    │
│                                                              │
│ Input: step, total_steps                                     │
│ Process:                                                     │
│   - progress = step / total_steps                            │
│   - x = -1.0 + 2.0 * progress  (linearly -1→1)               │
│   - y = 0.5 * sin(2π * progress)  (sinusoidal)               │
│                                                              │
│ Output: conditioning_info = [x, y] ∈ ℝ²                      │
└───────────────────────────────────────────────────────────── ┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Train Step (flow_matching.py: train_step)           │
│                                                             │
│ 2a. Sample Random Timestep (line 28)                        │
│     timestep = random value ∈ [0, 1)                        │
│                                                             │
│ 2b. Generate Noisy Data (utils.py: generate_noisy_data)     │
│     Input: target_data, timestep                            │
│     Process:                                                │
│       - noise_scale = 1.0 - timestep                        │
│       - noise = randn() * noise_scale                       │
│       - noisy_data = target_data * timestep + noise         │
│     Output: noisy_data ∈ ℝ^(N×2)                            │
│                                                             │
│ 2c. Get Target Vector Field (utils.py: get_target_vf)       │
│     Input: noisy_data, target_data, timestep                │
│     Process:                                                │
│       target_vf = target_data - noisy_data                  │
│     Output: target_vf ∈ ℝ^(N×2)                             │
│                                                             │
│ 2d. Prepare Model Input (line 32)                           │
│     input_data = concatenate([                              │
│         noisy_data,         # shape: (N, 2)                 │
│         conditioning_info   # shape: (N, 2)                 │
│     ], dim=1)                                               │
│     Output: input_data ∈ ℝ^(N×4)                            │
│                                                             │
│ 2e. Forward Pass (line 33)                                  │
│     predicted_vf = model(input_data)                        │
│     Through VectorFieldNet:                                 │
│       - Input: (N, 4) → Linear(4,64)                        │
│       - ReLU → Linear(64,64)                                │
│       - ReLU → Linear(64,2)                                 │
│     Output: predicted_vf ∈ ℝ^(N×2)                          │
│                                                             │
│ 2f. Compute Loss and Optimize (lines 35-39)                │
│     loss = MSE(predicted_vf, target_vf)                    │
│     optimizer.zero_grad()                                  │
│     loss.backward()                                        │
│     optimizer.step()                                       │
│                                                            │
│ Output: loss value                                         │
└────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Periodic Visualization (lines 25-27)                │
│ If step % 100 == 0:                                         │
│   - Call visualize_flow_matching()                          │
│   - Save snapshot image                                     │
└─────────────────────────────────────────────────────────────┘
```

### Training Data Flow Diagram

```
Target Data (Ellipse/Circles)          Conditioning Info
   ℝ^(100×2)                                     ℝ^2
        │                                       │
        │                                       │
        ▼                                       ▼
    [Timestep Sampling]                    [Concatenate]
        t ∈ [0,1)                              │
        │                                      │
        ▼                                      │
    [Add Noise]                                │
   noisy_data =                                │
   target * t + noise                          │
        │                                      │
        │──────────────────────────────────────┴───┐
        │                                           │
        ▼                                           ▼
  [Input]    noisy_data (N,2) + conditioning (N,2)  │
                ↓                                    │
           [Model]                                   │
     VectorFieldNet(4) → ... → (2)                   │
                ↓                                    │
       predicted_vf ∈ ℝ^(N×2)                       │
                │                                    │
                ▼                                    ▼
        target_vf = target - noisy_data
        [Compute MSE Loss]
        [Backward & Update]
```

---

## 2. INFERENCE/FLOW FLOW

### Step-by-Step Process

```
┌───────────────────────────────────────────────────────────── ┐
│ Generate Flow Animation (main.py lines 30-33)                │
│ - Creates animation showing transformation                   │
└───────────────────────────────────────────────────────────── ┘
                         ↓
┌───────────────────────────────────────────────────────────── ┐
│ visualize_flow (visualization.py: lines 26-38)               │
│                                                              │
│ Creates animation with 100 frames                            │
└───────────────────────────────────────────────────────────── ┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ For Each Frame (update function)                            │
│                                                             │
│ 1. Calculate timestep (line 30)                             │
│    timestep = frame / 100  [0 → 1)                          │
│                                                             │
│ 2. Generate noisy data for current timestep                 │
│    Uses same generate_noisy_data() function                 │
│    - For t=0: Mostly noise, small signal                    │
│    - For t→1: Less noise, more signal                       │
│                                                             │
│ 3. Visualize current state                                  │
│    - Scatter plot of noisy_data                             │
│    - Shows how data evolves from noise to target            │
│                                                             │
│ Key Difference from Training:                               │
│   Training: timestep is RANDOM                              │
│   Inference: timestep is SEQUENTIAL (0→1)                   │
└─────────────────────────────────────────────────────────────┘
```

### Inference Animation Flow

```
Frame 0   Frame 10  Frame 50  Frame 100
  t=0       t=0.1     t=0.5     t=1.0
   │          │         │         │
   ▼          ▼         ▼         ▼
[Noise]   [Noisey]  [Mixed]   [Target]
mostly      lots      some      clean
noise       noise     signal    signal

Data progressively transforms from random noise
to the target shape (ellipse/circle)
```

---

## 3. VISUALIZATION FLOW

### Two Types of Visualizations

#### A. Snapshot Visualization (visualize_flow_matching)

```
┌─────────────────────────────────────────────────────────────┐
│ visualize_flow_matching()                                   │
│ (visualization.py: lines 7-24)                              │
│                                                             │
│ Purpose: Show learned vector field at specific step         │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. Create 2D Grid (lines 9-10)                              │
│    x, y ∈ [-3, 3] × [-3, 3]                                 │
│    grid_points: 400 points (20×20)                          │
│                                                             │
│ 2. Evaluate Model on Grid (lines 12-15)                     │
│    For each grid point:                                     │
│      - Create input = [grid_point, conditioning]            │
│      - predicted_vf = model(input)                          │
│    Output: vector_field ∈ ℝ^(400×2)                         │
│                                                             │
│ 3. Create Visualization (lines 17-23)                       │
│    - Quiver plot: Shows vector field directions             │
│    - Scatter plot: Shows target data points                 │
│    - Saves: images/vector_field_step_X.png                  │
└─────────────────────────────────────────────────────────────┘
```

#### B. Animation Visualization (visualize_flow)

```
┌────────────────────────────────────────────────────────── ───┐
│ visualize_flow()                                             │
│ (visualization.py: lines 26-38)                              │
│                                                              │
│ Purpose: Show how data evolves during flow                   │
└────────────────────────────────────────────────────────── ───┘
                         ↓
┌─────────────────────────────────────────────────────────── ──┐
│ For each frame i ∈ [0, 100):                                 │
│                                                              │
│ 1. Calculate timestep (line 30)                              │
│    t = i / 100                                               │
│                                                              │
│ 2. Generate noisy data (line 31)                             │
│    noisy_data = generate_noisy_data(target_data, t)          │
│    - Shows data at timestep t during forward ODE             │
│                                                              │
│ 3. Plot current state (lines 32-36)                          │
│    - Blue scatter plot of noisy_data                         │
│    - Axes: [-3, 3] × [-3, 3]                                 │
│                                                              │
│ 4. Combine frames (line 38)                                  │
│    - FuncAnimation creates GIF                               │
│    - Shows transformation from noise to target               │
└──────────────────────────────────────────────────────────── ─┘
```

---

## Key Differences Summary

| Aspect | Training | Inference/Visualization |
|--------|----------|------------------------|
| **Timestep** | Random sampling ∈ [0,1) | Sequential 0→1 |
| **Purpose** | Learn vector field | Show learned flow |
| **Data** | Random batch | Specific target shape |
| **Output** | Loss value | GIF/Image files |
| **Frequency** | Every 100 steps | End of training |

---

## Data Dimensions

```
Input Dimensions:
- Target data:        ℝ^(100×2)    # 100 points in 2D
- Grid points:        ℝ^(400×2)    # 20×20 grid
- Conditioning:      ℝ^2          # 2D conditioning vector

Model Input:
- Training:          ℝ^(N×4)      # Concatenate data + conditioning
- Inference:         ℝ^(N×4)      # Same for visualization

Model Output:
- Vector Field:      ℝ^(N×2)      # 2D velocity vector

Loss:
- Target VF:         ℝ^(N×2)      # Ground truth velocity
- Predicted VF:      ℝ^(N×2)      # Model prediction
```

---

## Network Architecture

```
VectorFieldNet
├── Input: (N, 4)   → [noisy_data (2) + conditioning (2)]
├── Linear(4→64)    → ReLU
├── Linear(64→64)   → ReLU
└── Linear(64→2)    → vector field output
```

