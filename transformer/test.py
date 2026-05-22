import torch
import math

x = torch.randn(1, 6, 512)
x = x.view(1, 6, 8, 64)
print(x.shape)
print(x)