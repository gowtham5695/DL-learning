import torch
import torch.nn as nn

p = torch.tensor([0.9, 0.2, 0.3])
y = torch.tensor([1.0, 0.0, 1.0])
loss_each = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))
print("loss per sample:", loss_each)          # [0.105, 0.223, 1.204]
print("average (log-loss):", loss_each.mean().item())   # 0.511


print("loss:",nn.BCELoss()(p,y).item())