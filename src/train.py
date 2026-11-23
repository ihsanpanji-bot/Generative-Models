"""Training semua model. Contoh:  python -m src.train --model ae
Model: ae | sae | vae | gan | dcgan | ddpm | mlp
"""
import argparse
import copy
import json
import time

import torch
import torch.nn.functional as F

from src.common import (CKPT_DIR, NOISE_DIM, RES_DIR, SEED, get_device, get_loaders, set_seed, sync)
from src.models import build

DEFAULT_EPOCHS = dict(ae=20, sae=20, vae=30, gan=40, dcgan=30, ddpm=25, mlp=25)


def save_history(name, hist):
    (RES_DIR / name).mkdir(parents=True, exist_ok=True)
    with open(RES_DIR / name / "history.json", "w") as f:
        json.dump(hist, f, indent=1)


def save_ckpt(name, model):
    CKPT_DIR.mkdir(exist_ok=True)
    torch.save({"state": model.state_dict()}, CKPT_DIR / f"{name}.pt")


def log(name, ep, epochs, **kv):
    print(f"[{name}] ep {ep + 1}/{epochs} | " + " | ".join(f"{k}={v:.4f}" for k, v in kv.items()), flush=True)


# ------------------------------------------------------------------ AE / Sparse AE
def train_ae(name, epochs, lr, lam, dev, loaders):
    model = build(name).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr)
    h = dict(train_loss=[], val_loss=[], active_frac=[], epoch_time=[])
    best = float("inf")
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.perf_counter(), 0.0, 0
        for x, _ in loaders["train"]:
            x = x.to(dev)
            xr, z = model(x)
            loss = F.mse_loss(xr, x)
            if name == "sae":
                loss = loss + lam * z.abs().mean()     # penalti sparsity lambda*||h||_1
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(x); n += len(x)
        sync(dev)
        h["epoch_time"].append(time.perf_counter() - t0)
        model.eval()
        with torch.no_grad():
            vs, vn, act = 0.0, 0, 0.0
            for x, _ in loaders["val"]:
                x = x.to(dev)
                xr, z = model(x)
                vs += F.mse_loss(xr, x, reduction="sum").item() / 784; vn += len(x)
                act += (z.abs() > 1e-3).float().mean(1).sum().item()
        h["train_loss"].append(tot / n); h["val_loss"].append(vs / vn); h["active_frac"].append(act / vn)
        if vs / vn < best:
            best = vs / vn; save_ckpt(name, model)
        log(name, ep, epochs, train=tot / n, val_mse=vs / vn, active=act / vn)
    return h


# ------------------------------------------------------------------ VAE
def vae_loss(model, x):
    xr, mu, logvar = model(x)
    rec = F.binary_cross_entropy(xr, x, reduction="sum") / len(x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x)
    return rec, kl


def train_vae(name, epochs, lr, lam, dev, loaders):
    model = build("vae").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr)
    h = dict(train_loss=[], train_rec=[], train_kl=[], val_loss=[], epoch_time=[])
    best = float("inf")
    for ep in range(epochs):
        model.train()
        t0, tl, tr, tk, n = time.perf_counter(), 0.0, 0.0, 0.0, 0
        for x, _ in loaders["train"]:
            x = x.to(dev)
            rec, kl = vae_loss(model, x)
            loss = rec + kl                       # -ELBO = BCE(rekonstruksi) + KL(q||p)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(x); tr += rec.item() * len(x); tk += kl.item() * len(x); n += len(x)
        sync(dev)
        h["epoch_time"].append(time.perf_counter() - t0)
        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for x, _ in loaders["val"]:
                x = x.to(dev); r, k = vae_loss(model, x); vl += (r + k).item() * len(x); vn += len(x)
        h["train_loss"].append(tl / n); h["train_rec"].append(tr / n); h["train_kl"].append(tk / n)
        h["val_loss"].append(vl / vn)
        if vl / vn < best:
            best = vl / vn; save_ckpt(name, model)
        log(name, ep, epochs, train=tl / n, rec=tr / n, kl=tk / n, val=vl / vn)
    return h


# ------------------------------------------------------------------ GAN / DCGAN
def train_gan(name, epochs, lr, lam, dev, loaders):
    model = build(name).to(dev)
    G, D = model.G, model.D
    oG = torch.optim.Adam(G.parameters(), lr, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), lr, betas=(0.5, 0.999))
    bce = F.binary_cross_entropy_with_logits
    h = dict(d_loss=[], g_loss=[], d_real=[], d_fake=[], epoch_time=[])
    for ep in range(epochs):
        model.train()
        t0, a = time.perf_counter(), dict(d=0.0, g=0.0, dr=0.0, df=0.0)
        n = 0
        for x, _ in loaders["train"]:
            x = x.to(dev) * 2 - 1                 # [0,1] -> [-1,1] (Tanh)
            B = len(x)
            z = torch.randn(B, NOISE_DIM, device=dev)
            fake = G(z)
            # --- Discriminator
            lr_, lf_ = D(x), D(fake.detach())
            d_loss = bce(lr_, torch.ones_like(lr_)) + bce(lf_, torch.zeros_like(lf_))
            oD.zero_grad(); d_loss.backward(); oD.step()
            # --- Generator (non-saturating loss: maksimumkan log D(G(z)))
            lg = D(fake)
            g_loss = bce(lg, torch.ones_like(lg))
            oG.zero_grad(); g_loss.backward(); oG.step()
            a["d"] += d_loss.item() * B; a["g"] += g_loss.item() * B
            a["dr"] += torch.sigmoid(lr_).mean().item() * B; a["df"] += torch.sigmoid(lf_).mean().item() * B
            n += B
        sync(dev)
        h["epoch_time"].append(time.perf_counter() - t0)
        h["d_loss"].append(a["d"] / n); h["g_loss"].append(a["g"] / n)
        h["d_real"].append(a["dr"] / n); h["d_fake"].append(a["df"] / n)
        log(name, ep, epochs, D=a["d"] / n, G=a["g"] / n, D_real=a["dr"] / n, D_fake=a["df"] / n)
    save_ckpt(name, model)
    return h


# ------------------------------------------------------------------ DDPM
def train_ddpm(name, epochs, lr, lam, dev, loaders):
    model = build("ddpm").to(dev)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.Adam(model.parameters(), lr)
    h = dict(train_loss=[], epoch_time=[])
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.perf_counter(), 0.0, 0
        for x, _ in loaders["train"]:
            x = x.to(dev) * 2 - 1
            loss = model.loss(x)                  # MSE(eps, eps_theta(x_t, t))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():                 # EMA bobot -> sampel lebih bersih
                for pe, p in zip(ema.parameters(), model.parameters()):
                    pe.mul_(0.999).add_(p.detach(), alpha=0.001)
            tot += loss.item() * len(x); n += len(x)
        sync(dev)
        h["epoch_time"].append(time.perf_counter() - t0)
        h["train_loss"].append(tot / n)
        save_ckpt(name, ema)                      # simpan EMA tiap epoch (aman bila terhenti)
        log(name, ep, epochs, loss=tot / n)
    return h


# ------------------------------------------------------------------ MLP classifier
def train_mlp(name, epochs, lr, lam, dev, loaders):
    model = build("mlp").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr)
    h = dict(train_loss=[], val_acc=[], epoch_time=[])
    best = -1.0

    @torch.no_grad()
    def acc(loader):
        model.eval(); c = t = 0
        for x, y in loader:
            c += (model(x.to(dev)).argmax(1).cpu() == y).sum().item(); t += len(y)
        return c / t

    for ep in range(epochs):
        model.train()
        t0, tot, n = time.perf_counter(), 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(dev), y.to(dev)
            loss = F.cross_entropy(model(x), y)   # Cross-Entropy -> memodelkan P(y|x)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(x); n += len(x)
        sync(dev)
        h["epoch_time"].append(time.perf_counter() - t0)
        va = acc(loaders["val"])
        h["train_loss"].append(tot / n); h["val_acc"].append(va)
        if va > best:
            best = va; save_ckpt(name, model)
        log(name, ep, epochs, train=tot / n, val_acc=va)
    model.load_state_dict(torch.load(CKPT_DIR / f"{name}.pt", map_location=dev)["state"])
    h["test_acc"] = acc(loaders["test"])
    print(f"[mlp] TEST ACCURACY = {h['test_acc']:.4f} (syarat soal: >= 0.88)")
    return h


TRAINERS = dict(ae=train_ae, sae=train_ae, vae=train_vae, gan=train_gan, dcgan=train_gan,
                ddpm=train_ddpm, mlp=train_mlp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(TRAINERS))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lam", type=float, default=1e-3, help="bobot penalti L1 untuk Sparse AE")
    a = p.parse_args()
    lr = a.lr or {"gan": 2e-4, "dcgan": 2e-4, "ddpm": 2e-4}.get(a.model, 1e-3)
    epochs = a.epochs or DEFAULT_EPOCHS[a.model]
    set_seed(SEED)
    dev = get_device()
    print(f"device={dev} model={a.model} epochs={epochs} lr={lr}")
    hist = TRAINERS[a.model](a.model, epochs, lr, a.lam, dev, get_loaders())
    hist["config"] = dict(model=a.model, epochs=epochs, lr=lr, lam=a.lam, seed=SEED, device=str(dev))
    save_history(a.model, hist)


if __name__ == "__main__":
    main()
