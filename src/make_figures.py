"""Figur 1, 2, 4, 5 (Figur 3 dihasilkan oleh src.mlp_proof).  Jalankan: python -m src.make_figures"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib import gridspec
from torchvision.utils import make_grid

from src.common import NOISE_DIM, RES_DIR, SEED, get_device, get_loaders, set_seed
from src.evaluate import load

DPI = 200


def grid(x, nrow):
    return make_grid(x.cpu(), nrow=nrow, padding=2, pad_value=1.0)[0].numpy()


def fig1(models, x_test, dev):
    x = x_test[:8]
    rows = [("Ground Truth", x)] + [(n, models[k].reconstruct(x.to(dev)).cpu())
                                    for n, k in (("Vanilla AE", "ae"), ("Sparse AE", "sae"), ("VAE", "vae"))]
    fig, axes = plt.subplots(4, 1, figsize=(10, 5.6))
    for ax, (t, im) in zip(axes, rows):
        ax.imshow(grid(im, 8), cmap="gray", vmin=0, vmax=1); ax.set_ylabel(t, rotation=0, ha="right", va="center", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Figur 1. Rekonstruksi Test Set: Ground Truth vs Vanilla AE vs Sparse AE vs VAE")
    fig.savefig(RES_DIR / "fig1_reconstruction.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)


def fig2(models, dev):
    set_seed(SEED)
    panels = [("A. VAE (z~N(0,I))", "vae"), ("B. Vanilla GAN (z~N(0,I))", "gan"),
              ("C. DCGAN (z~N(0,I))", "dcgan"), ("D. DDPM (x_T~N(0,I), reverse denoising)", "ddpm")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, (t, k) in zip(axes, panels):
        ax.imshow(grid(models[k].sample(16), 4), cmap="gray", vmin=0, vmax=1)
        ax.set_title(t, fontsize=10); ax.axis("off")
    fig.suptitle("Figur 2. Random Generative Sampling (kisi 4x4 per model)")
    fig.savefig(RES_DIR / "fig2_sampling.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)


@torch.no_grad()
def pick_far_pair(model, dev, n=40):
    """GAN/DCGAN tidak punya encoder: ambil dua z dari prior yang menghasilkan citra paling berbeda
    (jarak piksel terbesar dari 40 kandidat), agar transisinya terlihat jelas."""
    z = torch.randn(n, NOISE_DIM, device=dev)
    imgs = model.sample(n, z=z).flatten(1)
    i, j = divmod(torch.cdist(imgs, imgs).argmax().item(), n)
    return z[i:i + 1], z[j:j + 1]


@torch.no_grad()
def fig4(models, x_test, y_test, dev):
    set_seed(SEED)
    # dua kelas kontras: Trouser(1) vs Bag(8)
    iA, iB = (y_test == 1).nonzero()[0].item(), (y_test == 8).nonzero()[0].item()
    xa, xb = x_test[iA:iA + 1].to(dev), x_test[iB:iB + 1].to(dev)
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for name, k in (("Vanilla AE", "ae"), ("Sparse AE", "sae")):
        za, zb = models[k].encode(xa), models[k].encode(xb)
        rows.append((name, torch.cat([models[k].decode((1 - a) * za + a * zb) for a in alphas]).cpu()))
    mu_a, mu_b = models["vae"].encode(xa)[0], models["vae"].encode(xb)[0]
    rows.append(("VAE", torch.cat([models["vae"].decode((1 - a) * mu_a + a * mu_b) for a in alphas]).cpu()))
    for name, k in (("Vanilla GAN", "gan"), ("DCGAN", "dcgan")):   # tanpa encoder -> dua z acak dari prior
        za, zb = pick_far_pair(models[k], dev)
        rows.append((name, torch.cat([models[k].sample(1, z=(1 - a) * za + a * zb) for a in alphas]).cpu()))
    fig, axes = plt.subplots(5, 1, figsize=(8, 7.8))
    for ax, (t, im) in zip(axes, rows):
        ax.imshow(grid(im, 6), cmap="gray", vmin=0, vmax=1); ax.set_ylabel(t, rotation=0, ha="right", va="center", fontsize=9)
        ax.set_yticks([]); ax.set_xticks([2 + 14 + 30 * i for i in range(6)])
        ax.set_xticklabels([f"α={a}" for a in alphas], fontsize=8)
    fig.suptitle("Figur 4. Linear Latent Interpolation  z(α) = (1-α) z_A + α z_B")
    fig.savefig(RES_DIR / "fig4_interpolation.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)


def fig5(models, dev):
    H = {k: json.load(open(RES_DIR / k / "history.json")) for k in ("ae", "sae", "vae", "gan", "dcgan", "ddpm")}
    fig = plt.figure(figsize=(16, 10.5))
    gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 0.9], hspace=0.5, wspace=0.3)
    def ax_(r, c, title, ylabel):
        a = fig.add_subplot(gs[r, c]); a.set_title(title, fontsize=10); a.set_xlabel("epoch"); a.set_ylabel(ylabel)
        a.grid(alpha=0.3); return a
    a = ax_(0, 0, "Vanilla AE", "MSE"); a.plot(H["ae"]["train_loss"], label="train"); a.plot(H["ae"]["val_loss"], label="val"); a.legend()
    a = ax_(0, 1, "Sparse AE (MSE + λ‖h‖₁)", "loss"); a.plot(H["sae"]["train_loss"], label="train (total)"); a.plot(H["sae"]["val_loss"], label="val (MSE)"); a.legend()
    a = ax_(0, 2, "VAE (-ELBO)", "BCE + KL"); a.plot(H["vae"]["train_loss"], label="train"); a.plot(H["vae"]["val_loss"], label="val"); a.legend()
    a = ax_(0, 3, "DDPM (prediksi derau)", "MSE(ε, ε_θ)"); a.plot(H["ddpm"]["train_loss"])
    for c, (k, t) in enumerate((("gan", "Vanilla GAN"), ("dcgan", "DCGAN"))):
        a = ax_(1, c, f"{t}: loss G & D", "BCE"); a.plot(H[k]["d_loss"], label="D loss"); a.plot(H[k]["g_loss"], label="G loss"); a.legend()
    for c, (k, t) in zip((2, 3), (("gan", "Vanilla GAN"), ("dcgan", "DCGAN"))):
        a = ax_(1, c, f"{t}: output Discriminator", "rata-rata D(·)")
        a.plot(H[k]["d_real"], label="D(x real)"); a.plot(H[k]["d_fake"], label="D(G(z))"); a.axhline(0.5, ls="--", c="gray"); a.legend()
    # trajectory reverse diffusion
    set_seed(SEED)
    _, traj = models["ddpm"].sample(1, return_traj=True, traj_points=10)
    ax = fig.add_subplot(gs[2, :])
    ax.imshow(grid(torch.cat(traj), len(traj)), cmap="gray", vmin=0, vmax=1)
    T = models["ddpm"].T
    steps = ["x_T"] + [f"t={int(round(s))}" for s in torch.linspace(T - 1, 0, 10).tolist()]
    ax.set_xticks([2 + 14 + 30 * i for i in range(len(traj))]); ax.set_xticklabels(steps, fontsize=8); ax.set_yticks([])
    ax.set_title("DDPM reverse denoising trajectory (x_T → x_0)", fontsize=10)
    fig.suptitle("Figur 5. Training Loss Curves (6 model) dan DDPM Denoising Trajectory", fontsize=13)
    fig.savefig(RES_DIR / "fig5_losses_ddpm.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)


def main():
    dev = get_device()
    loaders = get_loaders()
    x_test = torch.cat([x for x, _ in loaders["test"]]); y_test = torch.cat([y for _, y in loaders["test"]])
    models = {k: load(k, dev) for k in ("ae", "sae", "vae", "gan", "dcgan", "ddpm")}
    fig1(models, x_test, dev); print("fig1 ok")
    fig2(models, dev); print("fig2 ok")
    fig4(models, x_test, y_test, dev); print("fig4 ok")
    fig5(models, dev); print("fig5 ok")
    print("Figur tersimpan di", RES_DIR, "(Figur 3: python -m src.mlp_proof)")


if __name__ == "__main__":
    main()
