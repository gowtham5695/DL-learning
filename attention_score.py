import torch
import torch.nn.functional as F

Q = torch.tensor([[1.0, 0.0]])

K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

V = torch.tensor([
    [10.0, 0.0],
    [0.0, 10.0],
    [5.0, 5.0]
])

# Calculate scores
scores = Q @ K.T

# Calculate attention weights
weights = F.softmax(scores, dim=-1)

# Calculate final output
output = weights @ V

print("Scores:", scores)
print("Weights:", weights)
print("Attention Output:", output)