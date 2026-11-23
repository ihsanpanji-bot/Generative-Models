| Model | Dim Laten | Train (s/epoch) | Latency (ms/batch 64) | Test MSE | Test PSNR (dB) | Test SSIM | FID | Params (M) |
|---|---|---|---|---|---|---|---|---|
| Vanilla Autoencoder | 64 | 0.8 | 0.19 | 0.00705 | 22.61 | 0.8174 | N/A | 0.44 |
| Sparse Autoencoder | 64 (L1) | 1.4 | 0.28 | 0.00868 | 21.62 | 0.7816 | N/A | 0.44 |
| VAE | 64 | 2.3 | 0.14 | 0.01495 | 19.17 | 0.7017 | 62.42 | 1.12 |
| Vanilla GAN (MLP) | 100 (prior noise) | 4.9 | 0.37 | N/A | N/A | N/A | 114.05 | 2.95 |
| DCGAN | 100 (prior noise) | 16.5 | 1.36 | N/A | N/A | N/A | 22.04 | 1.92 |
| DDPM | 28x28 (dimensi citra) | 50.2 | 11803.16 | N/A | N/A | N/A | 11.84 | 2.15 |
