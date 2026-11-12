import torch
import torch.nn as nn

class Block(nn.Module):

    def __init__(self, dim, heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            dim,
            heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x, mask=None):

        h = self.norm1(x)

        a, _ = self.attn(
            h, h, h,
            attn_mask=mask
        )

        x = x + a

        x = x + self.ffn(self.norm2(x))

        return x

x = torch.randn(2, 5, 32)

mask = torch.triu(
    torch.ones(5, 5),
    diagonal=1
).bool()

block = Block(32, 4)

output = block(x, mask)

print(output.shape)