import torch
import torch.nn as nn

l = nn.Sequential(nn.Linear(64, 32))

while True:
    a = torch.randn(1024, 64, dtype=torch.float32)
    y = l(a)
    yy = l(a[:10])
    print((y[:10] - yy).sum())
