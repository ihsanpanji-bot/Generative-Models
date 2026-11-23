import torch
import torch.nn as nn


class VAE(nn.Module):
    """VAE: 784 -> 512 -> 256 -> (mu, logvar) [64] -> 256 -> 512 -> 784 (sigmoid)."""

    def __init__(self, latent: int = 64):
        super().__init__()
        self.latent = latent
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.fc_mu = nn.Linear(256, latent)
        self.fc_logvar = nn.Linear(256, latent)
        self.dec = nn.Sequential(nn.Linear(latent, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(),
                                 nn.Linear(512, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.dec(z).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)   # reparameterization trick
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def reconstruct(self, x):
        return self.decode(self.encode(x)[0])   # pakai mean posterior (deterministik)

    @torch.no_grad()
    def sample(self, n, z=None):
        dev = next(self.parameters()).device
        z = torch.randn(n, self.latent, device=dev) if z is None else z.to(dev)
        return self.decode(z)
