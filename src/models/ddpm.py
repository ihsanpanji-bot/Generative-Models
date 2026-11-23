import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * freqs[None]
        return torch.cat([a.sin(), a.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(F.silu(te))[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """U-Net kecil untuk 28x28: resolusi 28 -> 14 -> 7 -> 14 -> 28, dengan time embedding."""

    def __init__(self, b=64):
        super().__init__()
        td = b * 4
        self.temb = nn.Sequential(SinEmb(b), nn.Linear(b, td), nn.SiLU(), nn.Linear(td, td))
        self.inc = nn.Conv2d(1, b, 3, padding=1)
        self.rb1, self.down1 = ResBlock(b, b, td), nn.Conv2d(b, b, 3, 2, 1)
        self.rb2, self.down2 = ResBlock(b, 2 * b, td), nn.Conv2d(2 * b, 2 * b, 3, 2, 1)
        self.mid1, self.mid2 = ResBlock(2 * b, 2 * b, td), ResBlock(2 * b, 2 * b, td)
        self.up2, self.rb3 = nn.Conv2d(2 * b, 2 * b, 3, padding=1), ResBlock(4 * b, 2 * b, td)
        self.up1, self.rb4 = nn.Conv2d(2 * b, b, 3, padding=1), ResBlock(2 * b, b, td)
        self.out = nn.Sequential(nn.GroupNorm(8, b), nn.SiLU(), nn.Conv2d(b, 1, 3, padding=1))

    def forward(self, x, t):
        te = self.temb(t)
        d1 = self.rb1(self.inc(x), te)                       # 28x28
        d2 = self.rb2(self.down1(d1), te)                    # 14x14
        h = self.mid2(self.mid1(self.down2(d2), te), te)     # 7x7
        h = self.up2(F.interpolate(h, scale_factor=2))       # 14x14
        h = self.rb3(torch.cat([h, d2], 1), te)
        h = self.up1(F.interpolate(h, scale_factor=2))       # 28x28
        h = self.rb4(torch.cat([h, d1], 1), te)
        return self.out(h)


class DDPM(nn.Module):
    """DDPM (Ho et al., 2020): forward q(x_t|x_0) tertutup, objektif prediksi derau epsilon (MSE),
    reverse ancestral sampling dengan sigma_t^2 = beta_t. Data di [-1,1]."""

    def __init__(self, T=1000, base=64):
        super().__init__()
        self.T = T
        self.net = UNet(base)
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", torch.cumprod(alphas, 0))

    def loss(self, x0):
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        eps = torch.randn_like(x0)
        ab = self.abar[t].view(-1, 1, 1, 1)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        return F.mse_loss(self.net(xt, t), eps)

    @torch.no_grad()
    def sample(self, n, return_traj=False, traj_points=10):
        """Mengembalikan citra [0,1]; jika return_traj, juga daftar frame x_T -> x_0."""
        dev = self.betas.device
        x = torch.randn(n, 1, 28, 28, device=dev)
        keep = set(torch.linspace(self.T - 1, 0, traj_points).round().long().tolist())
        to01 = lambda v: (v.clamp(-1, 1) + 1) / 2
        traj = [to01(x).cpu()] if return_traj else []
        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=dev, dtype=torch.long)
            eps = self.net(x, t)
            a, ab, b = self.alphas[i], self.abar[i], self.betas[i]
            mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            x = mean + b.sqrt() * torch.randn_like(x) if i > 0 else mean
            if return_traj and i in keep:
                traj.append(to01(x).cpu())
        out = to01(x)
        return (out, traj) if return_traj else out
