import torch
from torch import nn

loss = nn.BCEWithLogitsLoss()
a = torch.tensor([True, False, True, False]).float()
print(a)
b = torch.rand(4)
results = loss(a, b)
print(results)