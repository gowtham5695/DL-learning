import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

print("="*50)
print("PART 1: Matrix Multiplication")
print("="*50)

A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

print("A @ B (matrix multiplication):")
print(A @ B)

print("\nA * B (element-wise, NOT the same!):")
print(A * B)


print("\n" + "="*50)
print("PART 2: Tensor Basics")
print("="*50)

x = torch.tensor([1, 2, 3])
print("x:", x, "| shape:", x.shape, "| dtype:", x.dtype)

x_float = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print("x_float dtype:", x_float.dtype, "| requires_grad:", x_float.requires_grad)

zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
rand = torch.rand(2, 3)
print("\nzeros:\n", zeros)
print("ones:\n", ones)
print("rand:\n", rand)


print("\n" + "="*50)
print("PART 3: GPU / Device Setup")
print("="*50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

gpu_tensor = torch.rand(3, 3).to(device)
print("gpu_tensor is on:", gpu_tensor.device)


print("\n" + "="*50)
print("PART 4: CPU + GPU mismatch error (Practice Task 4)")
print("="*50)

cpu_tensor = torch.rand(3, 3)   # stays on CPU, never moved
try:
    result = gpu_tensor + cpu_tensor
    print("No error — you're likely on CPU-only mode")
except Exception as e:
    print("Error (expected):")
    print(e)


print("\n" + "="*50)
print("PART 5: Dataset and DataLoader (Practice Task 3)")
print("="*50)

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 10 samples, 3 features each (made up numbers)
X = [[i, i+1, i+2] for i in range(10)]
y = [i % 2 for i in range(10)]

dataset = MyDataset(X, y)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Dataset size: {len(dataset)}")
for batch_num, (X_batch, y_batch) in enumerate(loader):
    print(f"Batch {batch_num}: X_batch shape = {X_batch.shape}, y_batch shape = {y_batch.shape}")