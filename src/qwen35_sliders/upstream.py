"""Load the actual, hash-checked Glimmer formulation; no local GAN objective."""
from pathlib import Path
import subprocess

REVISION = "1f5b380d317b73779a8aeab63b6bec0093c2c65c"


def load(path=None):
    from glimmer_slider.upstream import load_upstream
    root = Path(path or ".cache/particle-sliders-pg82").resolve()
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/HyperGAN/particle-sliders.git", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "--detach", REVISION], check=True)
    return load_upstream(root)
