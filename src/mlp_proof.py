"""Bagian III: pembuktian batasan MLP Classifier untuk image generation (Tahap 1-5) -> Figur 3.
Prasyarat: python -m src.train --model mlp   dan   python -m src.train --model ae
Jalankan:  python -m src.mlp_proof
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import gridspec
from torchvision.utils import make_grid

from src.common import CKPT_DIR, CLASS_NAMES, LATENT_DIM, RES_DIR, SEED, get_device, get_loaders, set_seed
from src.metrics import recon_metrics
from src.models import build


def load(name, dev):
    m = build(name).to(dev)
    m.load_state_dict(torch.load(CKPT_DIR / f"{name}.pt", map_location=dev)["state"])
    for p in m.parameters():
        p.requires_grad = False          # bekukan seluruh bobot (requires_grad=False)
    return m.eval()


def row(x, nrow=None):
    x = x.cpu()
    return make_grid(x, nrow=nrow or len(x), padding=2, pad_value=1.0)[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dec-epochs", type=int, default=20)
    a = ap.parse_args()
    set_seed(SEED)
    dev = get_device()
    loaders = get_loaders()
    x_test = torch.cat([x for x, _ in loaders["test"]]); y_test = torch.cat([y for _, y in loaders["test"]])
    out = {}

    # ---------------- Tahap 1: MLP classifier (sudah dilatih via src.train --model mlp)
    mlp = load("mlp", dev)
    W1, W2, W3 = mlp.fc1.weight, mlp.fc2.weight, mlp.fc3.weight          # (256,784) (64,256) (10,64)
    with torch.no_grad():
        acc = (torch.cat([mlp(x.to(dev)).argmax(1).cpu() for x, _ in loaders["test"]]) == y_test).float().mean().item()
        z_all = {k: torch.cat([mlp.features(x.to(dev)).cpu() for x, _ in loaders[k]]) for k in ("train", "val", "test")}
    out["mlp_test_acc"] = acc
    print(f"Tahap 1: MLP test acc = {acc:.4f}")

    # ---------------- Tahap 2: proyeksi balik one-hot y_c lewat transpos bobot (tanpa training, bias diabaikan)
    with torch.no_grad():
        Y = torch.eye(10, device=dev)
        z_hat = F.relu(Y @ W3)                       # (W3^T y_c)^T  -> (10,64)
        h1_hat = F.relu(z_hat @ W2)                  # -> (10,256)
        x_cls = torch.sigmoid(h1_hat @ W1).view(-1, 1, 28, 28).cpu()   # -> (10,1,28,28)
        means = torch.stack([x_test[y_test == c].mean(0) for c in range(10)])
    out["stage2_mse_vs_class_mean"] = F.mse_loss(x_cls, means).item()
    out["stage2_std_across_classes"] = x_cls.flatten(1).std(0).mean().item()   # ~0 => semua kelas hampir sama

    # ---------------- Tahap 3: proyeksi balik z_MLP citra uji lewat transpos bobot
    with torch.no_grad():
        z_te = z_all["test"].to(dev)
        x_lat = torch.sigmoid(F.relu(z_te @ W2) @ W1).view(-1, 1, 28, 28).cpu()
    out["stage3_latent_transpose"] = recon_metrics(x_test, x_lat)
    print("Tahap 3 (W^T dari z_MLP):", out["stage3_latent_transpose"])

    # ---------------- Tahap 4: frozen latent decoding (decoder 64->256->784, MSE) vs Vanilla AE
    dec = nn.Sequential(nn.Linear(LATENT_DIM, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid()).to(dev)
    opt = torch.optim.Adam(dec.parameters(), 1e-3)
    ztr, zva = z_all["train"], z_all["val"]
    xtr = torch.cat([x for x, _ in loaders["train"]]); xva = torch.cat([x for x, _ in loaders["val"]])
    g = torch.Generator().manual_seed(SEED)
    best, best_state = float("inf"), None
    for ep in range(a.dec_epochs):
        perm = torch.randperm(len(ztr), generator=g)
        dec.train()
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            xr = dec(ztr[idx].to(dev)).view(-1, 1, 28, 28)
            loss = F.mse_loss(xr, xtr[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        dec.eval()
        with torch.no_grad():
            v = F.mse_loss(dec(zva.to(dev)).view(-1, 1, 28, 28), xva.to(dev)).item()
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in dec.state_dict().items()}
        print(f"Tahap 4 ep {ep + 1}/{a.dec_epochs} val_mse={v:.5f}")
    dec.load_state_dict(best_state); dec.eval()
    with torch.no_grad():
        x_fd = dec(z_te).view(-1, 1, 28, 28).cpu()
    ae = load("ae", dev)
    with torch.no_grad():
        x_ae = torch.cat([ae.reconstruct(x.to(dev)).cpu() for x, _ in loaders["test"]])
    out["stage4_frozen_decoder_mlp"] = recon_metrics(x_test, x_fd)
    out["stage4_vanilla_ae"] = recon_metrics(x_test, x_ae)
    print("Tahap 4  z_MLP :", out["stage4_frozen_decoder_mlp"])
    print("Tahap 4  z_AE  :", out["stage4_vanilla_ae"])

    # ---------------- Tahap 5: sampling prior N(mu_z, diag sigma_z^2) pada ruang laten MLP
    mu, sd = z_all["test"].mean(0), z_all["test"].std(0)
    set_seed(SEED)
    z_rand = mu + sd * torch.randn(64, LATENT_DIM)
    with torch.no_grad():
        x_rand = dec(z_rand.to(dev)).view(-1, 1, 28, 28).cpu()
    out["stage5_latent_stats"] = dict(frac_dead_units=(z_all["test"].max(0).values < 1e-6).float().mean().item(),
                                      frac_zero_activations=(z_all["test"] < 1e-6).float().mean().item())
    json.dump(out, open(RES_DIR / "mlp_proof.json", "w"), indent=1)
    torch.save(x_rand, RES_DIR / "mlp_prior_samples.pt")

    # ---------------- Figur 3
    fig = plt.figure(figsize=(13, 11))
    gs = gridspec.GridSpec(6, 1, height_ratios=[1, 1.15, 1, 1, 1, 1], hspace=0.55)
    panels = [
        ("(1) Citra asli (ground truth, test set)", row(x_test[:8])),
        ("(2a) Proyeksi balik transpos bobot dari one-hot kelas y_c: Sigmoid(W1ᵀ ReLU(W2ᵀ ReLU(W3ᵀ y_c)))", row(x_cls)),
        ("(2b) Proyeksi balik transpos bobot dari z_MLP citra uji: Sigmoid(W1ᵀ ReLU(W2ᵀ z_MLP))", row(x_lat[:8])),
        (f"(3a) Frozen Latent Decoder dari z_MLP  (MSE={out['stage4_frozen_decoder_mlp']['mse']:.4f}, "
         f"SSIM={out['stage4_frozen_decoder_mlp']['ssim']:.3f})", row(x_fd[:8])),
        (f"(3b) Rekonstruksi Vanilla AE dari z_AE  (MSE={out['stage4_vanilla_ae']['mse']:.4f}, "
         f"SSIM={out['stage4_vanilla_ae']['ssim']:.3f})", row(x_ae[:8])),
        ("(4) Random prior sampling z ~ N(μ_z, diag σ_z²) pada ruang laten MLP -> Decoder (16 sampel)", row(x_rand[:16])),
    ]
    for i, (t, img) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(t, fontsize=9, loc="left")
        ax.set_yticks([])
        if i == 1:
            ax.set_xticks([2 + 14 + 30 * c for c in range(10)]); ax.set_xticklabels(CLASS_NAMES, fontsize=7)
        else:
            ax.set_xticks([])
    fig.savefig(RES_DIR / "fig3_mlp_limitation.png", dpi=200, bbox_inches="tight")
    print("Figur 3 disimpan ->", RES_DIR / "fig3_mlp_limitation.png")


if __name__ == "__main__":
    main()
