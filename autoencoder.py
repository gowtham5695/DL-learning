
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Linear(784, 128)
        self.decoder = nn.Linear(128, 784)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = Autoencoder()

x = torch.randn(1, 784)
output = model(x)

print(output.shape)
