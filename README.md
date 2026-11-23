# Studi Komparatif Model Generatif: AE, Sparse AE, VAE, Vanilla GAN, DCGAN, DDPM

Tugas Take-Home Sesi 06 (Autoencoders) + Sesi 07 (GANs & Diffusion). Dataset: **Fashion-MNIST**
(70.000 citra, split 80/10/10 = 56k/7k/7k, seed 42, batch size 128).

Selain enam model generatif, repo ini berisi eksperimen pembuktian batasan **MLP Classifier**
untuk image generation (Bagian III laporan).

## Instalasi
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt  # untuk GPU NVIDIA, install torch dari https://pytorch.org sesuai versi CUDA
```

## Menjalankan (dari root repo)
```bash
python -m src.train --model ae        # juga: sae | vae | mlp | gan | dcgan | ddpm
python -m src.mlp_proof               # Tahap 1-5 pembuktian MLP -> Figur 3 (butuh model mlp & ae)
python -m src.evaluate                # tabel metrik -> results/metrics_table.md (butuh semua model)
python -m src.make_figures            # Figur 1, 2, 4, 5
python run_all.py                     # semuanya berurutan
```
Opsi berguna: `--epochs N` (train), `--lam` (bobot L1 Sparse AE), `--fid-n` / `--skip-fid` (evaluate).
Semua keluaran ada di `results/` (gambar 200 dpi, `metrics.json`, `mlp_proof.json`, `history.json` tiap model).

## Struktur
```
src/common.py      seed, split data, DataLoader (dipakai semua model)
src/metrics.py     MSE/PSNR/SSIM, FID (torchmetrics, Inception-v3), pengukur latency
src/models/        ae.py (Vanilla & Sparse) | vae.py | gan.py (MLP & DC) | ddpm.py | mlp.py
src/train.py       training semua model
src/mlp_proof.py   eksperimen pembuktian batasan MLP classifier (Tahap 1-5, Figur 3)
src/evaluate.py    evaluasi Test Set + tabel komparasi
src/make_figures.py  Figur 1, 2, 4, 5
```

## Desain & konfigurasi
| Model | Arsitektur | Loss | Optimizer |
|---|---|---|---|
| Vanilla AE | 784-256-64-256-784 (sigmoid) | MSE | Adam 1e-3 |
| Sparse AE | sama, laten ReLU | MSE + λ·mean(\|h\|), λ=1e-3 | Adam 1e-3 |
| VAE | 784-512-256-(μ,logσ² 64)-256-512-784 | BCE + KL | Adam 1e-3 |
| Vanilla GAN | G: 100-256-512-1024-784 (tanh); D: 784-1024-512-256-1 | BCE adversarial (non-saturating G) | Adam 2e-4, β1=0.5 |
| DCGAN | G: ConvT 7→14→28 + BN; D: Conv strided + BN + LeakyReLU | BCE adversarial | Adam 2e-4, β1=0.5 |
| DDPM | U-Net kecil + time embedding, T=1000, β linear 1e-4..0.02 | MSE(ε, ε_θ) | Adam 2e-4, EMA 0.999 |
| MLP Classifier | 784-256-64-10 | Cross-Entropy | Adam 1e-3 |

Catatan standardisasi: AE/VAE bekerja di [0,1] (sigmoid); GAN dan DDPM di [-1,1] (tanh) lalu dikonversi
kembali ke [0,1] sebelum menghitung metrik. Dimensi laten AE/Sparse AE/VAE = 64 (sama dengan z_MLP).
Latency = rata-rata waktu per batch 64 (warm-up + sinkronisasi GPU); DDPM = sampling penuh T=1000 langkah.
FID dihitung dengan jumlah sampel sama untuk semua model (default 5000 vs test set).

## Notebook
`run_experiments.ipynb` menjalankan seluruh pipeline di atas (lokal atau Google Colab dengan GPU).
Kode utama tetap berupa modul `.py` di `src/`.
