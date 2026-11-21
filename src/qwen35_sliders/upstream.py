"""Load the shared slider engine; GAN mathematics remains in ParticleGAN."""
from importlib.metadata import version, distribution
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

REVISION = "ec1161f5d74481be3540cb0dfa63d0e8f8b54e90"


def load(path=None):
    import particle_sliders
    from particle_sliders import winning_formulation, fit_routed_down
    from particle_sliders.engine import ConditionalParticleGAN, PG_VERSION, _released_sources
    from particle_sliders.readout_critic import ConditionalReadoutCritic
    if version("particle-sliders-core") != "0.15.2" or PG_VERSION != "0.11.1":
        raise RuntimeError("Install the pinned particle-sliders-core 0.15.2 / ParticleGAN 0.11.1 dependencies")
    if path is not None:
        expected = Path(path).resolve() / "packages/particle-sliders-core/src/particle_sliders"
        if Path(particle_sliders.__file__).resolve().parent != expected:
            raise ValueError("Configured upstream path differs from the installed shared core")
    provenance = json.loads(distribution("particle-sliders-core").read_text("direct_url.json") or "{}")
    actual = provenance.get("vcs_info", {}).get("commit_id")
    if actual is None:
        try:
            actual = subprocess.check_output(["git", "-C", str(Path(particle_sliders.__file__).parent),
                                               "rev-parse", "HEAD"], text=True,
                                              stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:
            raise RuntimeError("Install the shared core from the pinned Git dependency or its checkout") from None
    if actual != REVISION:
        raise RuntimeError("Installed shared-core revision differs from the project's pinned dependency")
    sources = _released_sources()
    return SimpleNamespace(stamp=winning_formulation(), fit_routed_down=fit_routed_down,
        ConditionalParticleGAN=ConditionalParticleGAN, ConditionalReadoutCritic=ConditionalReadoutCritic,
        identity=dict(core_version="0.15.2", core_revision=REVISION,
                      particlegan_version=PG_VERSION, particlegan_sources_sha256=sources))
