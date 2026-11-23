import argparse
import subprocess
import sys

ORDER = ["ae", "sae", "vae", "mlp", "gan", "dcgan", "ddpm"]

ap = argparse.ArgumentParser()
ap.add_argument("--skip", nargs="*", default=[])
a = ap.parse_args()
run = lambda *c: subprocess.run([sys.executable, "-m", *c], check=True)
for m in ORDER:
    if m not in a.skip:
        run("src.train", "--model", m)
run("src.mlp_proof")
run("src.evaluate")
run("src.make_figures")
