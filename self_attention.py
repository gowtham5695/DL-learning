import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.scale = dim ** 0.5

    def forward(self, x):

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        scores = Q @ K.transpose(1, 2)

        scores = scores / self.scale

        weights = F.softmax(scores, dim=-1)

        output = weights @ V

        return output, weights


x = torch.randn(1, 4, 8)

attention = SelfAttention(8)

output, weights = attention(x)
print("Output:", output)
print("Output:", output.shape)
print("Weights:", weights)
print("Weights:", weights.shape)

print("Sum of weights:")
print(weights.sum(dim=-1))