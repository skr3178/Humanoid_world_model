# Training Fixes Applied - High Priority

## Summary
Applied critical fixes to address loss not decreasing during training. These fixes target the most common causes of failed training: poor gradient estimates, unstable optimization, and improper initialization.

## Changes Made

### 1. Increased Batch Size (config_12gb.py)
**Issue**: Batch size of 1 provides extremely noisy gradient estimates and prevents proper normalization.

**Fix**:
- Changed `batch_size` from 1 to 4
- Adjusted `gradient_accumulation_steps` from 16 to 4
- Maintains effective batch size of 16 (4 × 4 = 16)

**Impact**:
- 4x better gradient estimates per update
- More stable LayerNorm statistics
- Better learning dynamics

### 2. Reduced Learning Rate (config_12gb.py)
**Issue**: Learning rate of 1e-4 was too high, causing instability with small batch sizes.

**Fix**: Reduced learning rate from 1e-4 to 5e-5
- Between paper's recommended 3e-5 and previous 1e-4
- More conservative for stability

**Impact**: More stable training, less risk of divergence

### 3. Switched to Cosine Schedule (train.py)
**Issue**: Linear decay doesn't provide smooth convergence at the end of training.

**Fix**: Changed from `get_linear_schedule_with_warmup` to `get_cosine_schedule_with_warmup`

**Impact**:
- Smoother convergence
- Better final performance
- Less aggressive LR decay

### 4. Scaled Factorized Embedding Initialization (model.py)
**Issue**: Summing 3 factorized embeddings increases output variance by 3x, leading to unstable early training.

**Fix**:
- Added `scaled_init_std = init_std / sqrt(num_factored_vocabs)`
- Applied to all factorized embeddings (video tokens)
- Maintains variance at intended level after summing

**Mathematical reasoning**:
```
If X1, X2, X3 are independent with Var(Xi) = σ²
Then Var(X1 + X2 + X3) = 3σ²

To maintain Var = σ², initialize each with Var = σ²/3
=> std = σ / sqrt(3)
```

**Impact**:
- Stable initial gradients
- Faster convergence in early training
- Proper variance propagation

### 5. Enhanced Gradient Logging (train.py)
**Issue**: Insufficient gradient diagnostics made it hard to identify training problems.

**Fix**: Added comprehensive gradient tracking:
- Pre-clipping gradient norm (for diagnostics)
- Post-clipping gradient norm (actual updates)
- Max/min gradient values
- Automatic warnings for gradient issues:
  - Large gradients (>100) → instability warning
  - Small gradients (<0.001) → vanishing gradient warning
  - NaN gradients → error

**Impact**:
- Easy identification of gradient problems
- Better debugging capabilities
- Early detection of training issues

## Expected Results

With these fixes, you should observe:

1. **Initial Loss**: ~11.1 (log(65536) for random prediction)
2. **After 100 steps**: Loss should drop to ~9-10 range
3. **After 1000 steps**: Loss should be in 7-8 range
4. **Gradient norms**: Should stay in 0.1-10 range (healthy)
5. **No warnings**: Gradient warnings should be rare

## How to Monitor Training

1. **Loss trend**: Should decrease consistently, even if slowly
2. **Gradient norms**: Watch the logged gradient statistics
   - Pre-clip norm should be 0.1-10 (healthy)
   - If pre-clip >> post-clip, clipping is active (expected early on)
   - If pre-clip >100, reduce LR or check data
   - If pre-clip <0.001, increase LR or check initialization

3. **Learning rate**: Follows cosine schedule
   - Starts at 0 (warmup)
   - Reaches 5e-5 after 500 steps
   - Gradually decays to ~0 by step 60,000

## Training Command

To train with the fixed configuration:

```bash
cd build && ./train_10pct_v2_12gb.sh
```

Or manually:

```bash
python3 build/training/train.py \
    --train_data_dir 1xgpt/data/train_v2.0 \
    --val_data_dir 1xgpt/data/val_v2.0 \
    --use_12gb_config \
    --max_steps 60000
```

## Files Modified

1. `build/masked_hwm/config_12gb.py` - Updated batch size and learning rate
2. `build/training/train.py` - Switched to cosine schedule, enhanced logging
3. `build/masked_hwm/model.py` - Scaled factorized embedding initialization
