import torch
import torch.nn.functional as F
import math

Q = torch.randn(2, 4)
K = torch.randn(3, 4)
V = torch.randn(3, 5)

d = Q.shape[-1]

scores = Q @ K.T

scaled_scores = scores / math.sqrt(d)

weights = F.softmax(scaled_scores, dim=-1)

output = weights @ V

print("Scores:", scores.shape)
print("Weights:", weights.shape)
print("Output:", output.shape)