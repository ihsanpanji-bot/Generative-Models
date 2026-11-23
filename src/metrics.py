"""Metrik: MSE / PSNR / SSIM (rekonstruksi), FID (sampel generatif), latency."""
import time

import numpy as np
import torch
from skimage.metrics import structural_similarity

from src.common import sync


def recon_metrics(x: torch.Tensor, xr: torch.Tensor) -> dict:
    """x, xr: (N,1,28,28) di [0,1], CPU. Mengembalikan rata-rata MSE, PSNR (dB), SSIM."""
    x, xr = x.float().cpu(), xr.float().cpu()
    mse_i = ((x - xr) ** 2).flatten(1).mean(1)
    psnr_i = 10 * torch.log10(1.0 / mse_i.clamp_min(1e-10))
    ssim_i = [structural_similarity(a[0].numpy(), b[0].numpy(), data_range=1.0) for a, b in zip(x, xr)]
    return {"mse": mse_i.mean().item(), "psnr": psnr_i.mean().item(), "ssim": float(np.mean(ssim_i))}


def compute_fid(real: torch.Tensor, fake: torch.Tensor, device, batch: int = 200) -> float:
    """FID (Inception-v3, 2048-d) antara citra real & generatif, keduanya (N,1,28,28) di [0,1]."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for src, is_real in ((real, True), (fake, False)):
        for i in range(0, len(src), batch):
            b = src[i:i + batch].to(device).clamp(0, 1).repeat(1, 3, 1, 1)   # grayscale -> 3 kanal
            fid.update(b, real=is_real)
    return float(fid.compute().item())


@torch.no_grad()
def time_fn(fn, device, warmup: int = 5, reps: int = 20) -> float:
    """Rata-rata waktu (ms) satu pemanggilan fn, dengan warm-up dan sinkronisasi GPU."""
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / reps * 1000.0
