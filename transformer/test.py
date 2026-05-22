import torch
import math

seq_len = 6
mymask = torch.tensor([[True, True, True, True],
                     [True, True, False, False],
                     [True, True, True, False]])


mask = mymask.unsqueeze(1).unsqueeze(-2)
mask = mask.expand(-1, -1, 4, -1)
t_mask = mask.transpose(2, 3)

result = mask & t_mask
print(mymask.shape[0])
print(result)