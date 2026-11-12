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

        a, _ = self.attn(h, h, h, attn_mask=mask)

        x = x + a

        x = x + self.ffn(self.norm2(x))

        return x


class MiniTransformer(nn.Module):

    def __init__(self, vocab, dim, heads, layers, max_len):
        super().__init__()

        self.tok = nn.Embedding(vocab, dim)

        self.pos = nn.Embedding(max_len, dim)

        self.blocks = nn.ModuleList([
            Block(dim, heads) for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(dim)

        self.out = nn.Linear(dim, vocab)

    def forward(self, ids):

        n = ids.shape[1]

        x = self.tok(ids) + self.pos(
            torch.arange(n, device=ids.device)
        )

        mask = torch.triu(
            torch.ones(n, n, device=ids.device),
            diagonal=1
        ).bool()

        for block in self.blocks:
            x = block(x, mask)

        return self.out(self.norm(x))


model = MiniTransformer(
    vocab=100,
    dim=32,
    heads=4,
    layers=2,
    max_len=20
)

ids = torch.randint(0, 100, (2, 10))

logits = model(ids)

print(logits.shape)

print(
    sum(p.numel() for p in model.parameters()),
    "weights"
)