import torch
import torch.nn as nn

from src.common import NOISE_DIM


def _init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class MLPGenerator(nn.Module):
    def __init__(self, zdim=NOISE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zdim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 784), nn.Tanh())

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1024), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1))   # logit (tanpa sigmoid; pakai BCEWithLogits)

    def forward(self, x):
        return self.net(x).view(-1)


class DCGenerator(nn.Module):
    """z(100) -> 256x7x7 -> 128x14x14 -> 1x28x28 (ConvTranspose + BatchNorm + ReLU, output Tanh)."""

    def __init__(self, zdim=NOISE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(zdim, 256, 7, 1, 0, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class DCDiscriminator(nn.Module):
    """Conv strided (tanpa pooling) + BatchNorm + LeakyReLU."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 1, 7, 1, 0))

    def forward(self, x):
        return self.net(x).view(-1)


class GANPair(nn.Module):
    """Pasangan G & D. kind='mlp' (Vanilla GAN) atau 'dc' (DCGAN). Data di [-1,1]."""

    def __init__(self, kind="mlp", zdim=NOISE_DIM):
        super().__init__()
        self.zdim = zdim
        self.G = MLPGenerator(zdim) if kind == "mlp" else DCGenerator(zdim)
        self.D = MLPDiscriminator() if kind == "mlp" else DCDiscriminator()
        if kind == "dc":
            self.apply(_init)

    @torch.no_grad()
    def sample(self, n, z=None):
        """Mengembalikan citra di [0,1]."""
        dev = next(self.parameters()).device
        z = torch.randn(n, self.zdim, device=dev) if z is None else z.to(dev)
        return (self.G(z) + 1) / 2
