import torch
import subprocess

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

x = torch.rand(4000, 4000, device=device)
for i in range(300):
    y = x @ x                      # big matrix multiplication
torch.cuda.synchronize()           # wait for the GPU to finish

print("memory held by PyTorch (MB):", torch.cuda.memory_allocated() / 1024**2)
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)