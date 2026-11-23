"""Evaluasi semua model pada Test Set -> results/metrics.json + results/metrics_table.md
Jalankan:  python -m src.evaluate [--fid-n 5000]
"""
import argparse
import json

import numpy as np
import torch

from src.common import CKPT_DIR, LATENT_DIM, NOISE_DIM, RES_DIR, SEED, get_device, get_loaders, set_seed
from src.metrics import compute_fid, recon_metrics, time_fn
from src.models import MODEL_NAMES, build

DISPLAY = dict(ae="Vanilla Autoencoder", sae="Sparse Autoencoder", vae="VAE",
               gan="Vanilla GAN (MLP)", dcgan="DCGAN", ddpm="DDPM")
LATENT_STR = dict(ae=str(LATENT_DIM), sae=f"{LATENT_DIM} (L1)", vae=str(LATENT_DIM),
                  gan=f"{NOISE_DIM} (prior noise)", dcgan=f"{NOISE_DIM} (prior noise)", ddpm="28x28 (dimensi citra)")


def load(name, dev):
    m = build(name).to(dev)
    m.load_state_dict(torch.load(CKPT_DIR / f"{name}.pt", map_location=dev)["state"])
    return m.eval()


@torch.no_grad()
def generate(model, name, n, dev):
    chunk = 500 if name == "ddpm" else 1000
    out = []
    for i in range(0, n, chunk):
        out.append(model.sample(min(chunk, n - i)).cpu())
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fid-n", type=int, default=5000, help="jumlah sampel untuk FID (sama utk semua model)")
    p.add_argument("--skip-fid", action="store_true")
    p.add_argument("--models", nargs="+", default=MODEL_NAMES)
    a = p.parse_args()
    dev = get_device()
    loaders = get_loaders()
    x_test = torch.cat([x for x, _ in loaders["test"]])
    x64 = x_test[:64].to(dev)
    results = {}

    for name in a.models:
        print(f"== {name}")
        set_seed(SEED)
        model = load(name, dev)
        r = dict(display=DISPLAY[name], latent=LATENT_STR[name],
                 params_M=sum(p.numel() for p in model.parameters()) / 1e6)
        hist = json.load(open(RES_DIR / name / "history.json"))
        r["train_s_per_epoch"] = float(np.mean(hist["epoch_time"]))

        # --- rekonstruksi (hanya model dengan encoder)
        if name in ("ae", "sae", "vae"):
            with torch.no_grad():
                xr = torch.cat([model.reconstruct(x.to(dev)).cpu() for x, _ in loaders["test"]])
            r.update(recon_metrics(x_test, xr))

        # --- latency per batch 64 (inference "bangkitkan/rekonstruksi 64 citra")
        if name in ("ae", "sae"):
            fn = lambda: model.reconstruct(x64)
        else:
            fn = lambda: model.sample(64)
        r["latency_ms_b64"] = time_fn(fn, dev, warmup=1 if name == "ddpm" else 5, reps=2 if name == "ddpm" else 20)

        # --- FID (hanya model generatif dari prior)
        if name in ("vae", "gan", "dcgan", "ddpm") and not a.skip_fid:
            n = min(a.fid_n, len(x_test))
            set_seed(SEED)
            fake = generate(model, name, n, dev)
            r["fid"] = compute_fid(x_test[:n], fake, dev)
        results[name] = r
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        del model

    (RES_DIR).mkdir(exist_ok=True)
    json.dump(results, open(RES_DIR / "metrics.json", "w"), indent=1)

    f = lambda v, d=4: "N/A" if v is None else f"{v:.{d}f}"
    lines = ["| Model | Dim Laten | Train (s/epoch) | Latency (ms/batch 64) | Test MSE | Test PSNR (dB) | Test SSIM | FID | Params (M) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        lines.append(f"| {r['display']} | {r['latent']} | {r['train_s_per_epoch']:.1f} | {r['latency_ms_b64']:.2f} | "
                     f"{f(r.get('mse'), 5)} | {f(r.get('psnr'), 2)} | {f(r.get('ssim'))} | {f(r.get('fid'), 2)} | {r['params_M']:.2f} |")
    (RES_DIR / "metrics_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
