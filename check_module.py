import torch
import os

# 检查 best_model
if os.path.exists('best_model.pt'):
    best = torch.load('best_model.pt', map_location='cpu')
    print(f"Best Model: Ep {best.get('episode', '未知')}")
else:
    print("Best Model 不存在")

# 检查 latest checkpoint
if os.path.exists('checkpoints/ckpt_latest.pt'):
    latest = torch.load('checkpoints/ckpt_latest.pt', map_location='cpu')
    print(f"Latest Checkpoint: Ep {latest.get('episode', '未知')}")
else:
    print("Latest Checkpoint 不存在")