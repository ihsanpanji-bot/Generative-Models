import os

from src.common import LATENT_DIM
from src.models.ae import AE
from src.models.ddpm import DDPM
from src.models.gan import GANPair
from src.models.mlp import MLPClassifier
from src.models.vae import VAE

MODEL_NAMES = ["ae", "sae", "vae", "gan", "dcgan", "ddpm"]


def build(name: str):
    return {
        "ae": lambda: AE(LATENT_DIM, sparse=False),
        "sae": lambda: AE(LATENT_DIM, sparse=True),
        "vae": lambda: VAE(LATENT_DIM),
        "gan": lambda: GANPair("mlp"),
        "dcgan": lambda: GANPair("dc"),
        "ddpm": lambda: DDPM(T=int(os.environ.get("DDPM_T", 1000))),   # DDPM_T hanya utk smoke test
        "mlp": lambda: MLPClassifier(),
    }[name]()
