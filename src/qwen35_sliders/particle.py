"""Qwen response binding for the shared ParticleGAN engine; no local GAN loss."""
from dataclasses import dataclass
import torch

from .host import hidden


@dataclass(frozen=True)
class HiddenRow:
    row: dict
    target: torch.Tensor
    neutral: torch.Tensor


class HiddenBackend:
    def __init__(self, model, slider):
        self.model, self.network = model, slider

    def metadata(self):
        return dict(model="Qwen3.5 text hidden response", targets=self.network.names,
                    protocol="worked-solution-context-paired-hidden-v2", strength=1.)

    def generated(self, prepared):
        with self.network.scaled(1.):
            return hidden(self.model, prepared.row["ids"])


def make_game(model, slider, training, config, api):
    rows = [HiddenRow(dict(ids=r["ids"]), r["target"], r["neutral"]) for r in training]
    target = torch.cat([r.target for r in rows])
    neutral = torch.cat([r.neutral for r in rows])
    scale = float((target-neutral).square().mean().sqrt().clamp_min(1e-4))
    conditions = torch.nn.functional.normalize(neutral, dim=-1).detach()
    critic = api.ConditionalReadoutCritic(target.shape[1], conditions.shape[1], hidden=48).to(target)
    overrides = dict(lr=config.get("lr", .0006),
                     d_lr_mult=config.get("d_lr_mult", 1.5),
                     prior_lr_mult=config.get("particle_lr_mult", 10.))
    game = api.ConditionalPG09(HiddenBackend(model, slider), rows, critic,
        conditions=conditions, particles=slider.particles,
        response_domain="continuous_vector", penalty_input_domain="continuous",
        response_scale=scale, preset="bispf_optimized", batch_size=config.get("batch", 4),
        seed=config.get("seed", 7), total_steps=config["steps"], recipe_overrides=overrides)
    return game


def initialize_ema(slider):
    return {k:v.detach().clone() for k,v in slider.state_dict().items()}


@torch.no_grad()
def update_ema(ema, slider, decay=.995):
    for key,value in slider.state_dict().items():
        ema[key].lerp_(value, 1.-decay)
