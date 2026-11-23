"""Konfigurasi bersama, seed, dan data pipeline (dipakai SEMUA model agar perbandingan adil)."""
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
RES_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"

BATCH_SIZE = 128      # soal: 128 atau 256
LATENT_DIM = 64       # dimensi laten AE / Sparse AE / VAE (sama dg z_MLP agar adil)
NOISE_DIM = 100       # dimensi prior noise GAN / DCGAN
CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_raw():
    """Fashion-MNIST: gabungkan train(60k)+test(10k)=70k, lalu di-split ulang 80/10/10."""
    if os.environ.get("SMOKE") == "1":   # hanya untuk uji cepat kode, bukan untuk hasil akhir
        g = torch.Generator().manual_seed(0)
        y = torch.randint(0, 10, (3000,), generator=g)
        x = torch.rand(3000, 28, 28, generator=g) * 0.3 + (y.float() / 10).view(-1, 1, 1) * 0.7
        return x.unsqueeze(1).clamp(0, 1), y
    from torchvision import datasets
    tr = datasets.FashionMNIST(DATA_DIR, train=True, download=True)
    te = datasets.FashionMNIST(DATA_DIR, train=False, download=True)
    x = torch.cat([tr.data, te.data]).float().div(255).unsqueeze(1)   # [0,1], (N,1,28,28)
    y = torch.cat([tr.targets, te.targets])
    return x, y


def get_splits():
    x, y = _load_raw()
    n = len(x)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    idx = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
    return {k: (x[i], y[i]) for k, i in idx.items()}


def get_loaders(batch_size: int = BATCH_SIZE):
    s = get_splits()
    g = torch.Generator().manual_seed(SEED)
    return {
        "train": DataLoader(TensorDataset(*s["train"]), batch_size, shuffle=True, generator=g),
        "val": DataLoader(TensorDataset(*s["val"]), batch_size, shuffle=False),
        "test": DataLoader(TensorDataset(*s["test"]), batch_size, shuffle=False),
    }
