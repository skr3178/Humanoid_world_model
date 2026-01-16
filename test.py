import torch
from PIL import Image
import sys
sys.path.insert(0, 'build')
from flow_hwm.dataset_latent import FlowHWMDataset
from flow_hwm.flow_matching import construct_flow_path, sample_noise
from cosmos_tokenizer.video_lib import CausalVideoTokenizer

ds = FlowHWMDataset('1xgpt/data/train_v2.0', num_past_clips=2, num_future_clips=1, latent_dim=16, max_shards=1)
x1 = ds[0]['latent_future'].unsqueeze(0).cuda()

torch.manual_seed(42)
x0 = sample_noise(x1.shape, 'cuda', std=0.5)

# Test with sigma_min=0 (exact x1 at t=1)
x_t_sigma0 = construct_flow_path(x0, x1, torch.tensor(1.0, device='cuda'), sigma_min=0.0)

diff = (x1 - x_t_sigma0).abs()
print(f'With sigma_min=0: max_diff={diff.max():.8f}')
print(f'Are they exactly equal? {torch.equal(x1, x_t_sigma0)}')

decoder = CausalVideoTokenizer(checkpoint_dec='cosmos_tokenizer/decoder.jit', device='cuda', dtype='bfloat16')

# Decode
B, C, T, H, W = x_t_sigma0.shape
lat3 = torch.stack([x_t_sigma0[:, i*5, :, :, :] for i in range(3)], dim=1)
tokens = ((lat3 + 1) / 2 * 65535).round().long().clamp(0, 65535)
with torch.no_grad():
    out = decoder.decode(tokens[:, :, 0, :, :])
frame = out[0, :, 0].float().cpu()
frame = ((frame + 1) / 2 * 255).clamp(0, 255).byte()
Image.fromarray(frame.permute(1, 2, 0).numpy()).save('decode_sigma0.png')
print('Saved decode_sigma0.png')
