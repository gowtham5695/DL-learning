import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

print("With out activation function:")
x = torch.tensor(2.0)
w1, w2, v1, v2 = 0.5, -0.3, 0.8, 0.6
two_layers=v1*(w1*x)+v2*(w2*x)
one_layer=(v1*w1+v2*w2)*x
print("Two layers:", two_layers.item())
print("One layer:", one_layer.item())

print("\nPART 2: Activations on the same inputs")
z = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print("Input:", z)
print("sigmoid:", torch.sigmoid(z))
print("tanh:", torch.tanh(z))
print("ReLU:", F.relu(z))
print("Leaky ReLU:", F.leaky_relu(z,negative_slope=0.01))

print("softmax")
prob=F.softmax(z,dim=0)
print("Probabilities:", prob)
print("sum of probabilities:", prob.sum().item())

print("\nPART 4: Gradients")
z = torch.tensor([0.0, 5.0, -5.0], requires_grad=True)
torch.sigmoid(z).sum().backward()
print("sigmoid gradient at [0, 5, -5]:", z.grad)   # [0.25, 0.0066, 0.0066]

z = torch.tensor([-3.0, 0.0, 3.0], requires_grad=True)
F.relu(z).sum().backward()
print("relu gradient at [-3, 0, 3]   :", z.grad)   # [0, 0, 1]

z = torch.linspace(-5, 5, 200)
plt.figure(figsize=(10, 6))
plt.plot(z, torch.sigmoid(z), label="sigmoid")
plt.plot(z, torch.tanh(z), label="tanh")
plt.plot(z, F.relu(z), label="ReLU")
plt.plot(z, F.leaky_relu(z, 0.1), label="Leaky ReLU (slope 0.1, exaggerated)")

plt.axhline(0, color="gray", linewidth=0.5)
plt.axvline(0, color="gray", linewidth=0.5)
plt.ylim(-2, 5)
plt.legend()
plt.title("Activation functions")
plt.grid(True)
plt.show()