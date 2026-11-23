import torch
import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):
    """Vanilla AE (sparse=False) dan Sparse AE (sparse=True).

    Arsitektur: 784 -> 256 -> latent(64) -> 256 -> 784 (sigmoid).
    Sparse AE: kode laten h = ReLU(.) (non-negatif) + penalti lambda*||h||_1 pada training.
    """

    def __init__(self, latent: int = 64, sparse: bool = False):
        super().__init__()
        self.sparse = sparse
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return F.relu(h) if self.sparse else h

    def decode(self, h):
        return self.dec(h).view(-1, 1, 28, 28)

    def forward(self, x):
        h = self.encode(x)
        return self.decode(h), h

    @torch.no_grad()
    def reconstruct(self, x):
        return self.forward(x)[0]
