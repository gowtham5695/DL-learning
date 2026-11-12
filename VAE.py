
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Linear(784, 256)
        self.mu = nn.Linear(256, 20)
        self.logvar = nn.Linear(256, 20)

        self.decoder = nn.Linear(20, 784)

    def forward(self, x):
        h = torch.relu(self.encoder(x))

        mu = self.mu(h)
        logvar = self.logvar(h)

        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        output = torch.sigmoid(self.decoder(z))

        return output, mu, logvar

model = VAE()
x = torch.randn(1, 784)

output, mu, logvar = model(x)

print(output.shape)
