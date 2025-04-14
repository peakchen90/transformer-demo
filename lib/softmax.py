import torch
import torch.nn.functional as F

def simple_softmax(x):
    x_exp = torch.exp(x - torch.max(x))
    x_sum = torch.sum(x_exp, dim=-1, keepdim=True)
    return x_exp / x_sum

x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
y = simple_softmax(x)
print(y)
