"""Attach to real Qwen hybrid attention/MLP dimensions, preserving the base."""
from contextlib import contextmanager
import math
import re

import torch
from torch import nn


LANGUAGE = re.compile(r"model\.language_model\.layers\.(\d+)\.(.+)")


def targets(model, scope="all"):
    if scope not in {"attention", "all"}:
        raise ValueError("scope must be attention or all")
    selected = {}
    for name, module in model.named_modules():
        match = LANGUAGE.fullmatch(name)
        if match and type(module) is nn.Linear:
            part = match.group(2)
            if scope == "all" or part.startswith(("self_attn.", "linear_attn.")):
                selected[name] = module
    if not selected:
        raise ValueError("No genuine Qwen language projections found")
    return selected


class Branch(nn.Module):
    def __init__(self, module, particles, api=None, rank=8, ordinary=False):
        super().__init__()
        self.down = nn.Linear(module.in_features, rank, bias=False, device=module.weight.device)
        self.up = nn.Linear(rank, module.out_features, bias=False, device=module.weight.device)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.ordinary = ordinary
        if not ordinary:
            self.bridge = api.stamp.bridge().to(module.weight.device)
            object.__setattr__(self, "particles", particles)
        object.__setattr__(self, "original", module.forward)
        self.strength = 0.

    def bottleneck(self, x):
        low = self.down(x.to(self.down.weight))
        return low if self.ordinary else self.bridge(low, self.particles)

    def forward(self, x):
        base = self.original(x)
        if self.strength == 0:
            return base
        return base + self.up(self.bottleneck(x)).to(base) * self.strength


class Slider(nn.Module):
    def __init__(self, model, api=None, *, scope="all", rank=8, ordinary=False):
        super().__init__()
        if not ordinary and rank != 8:
            raise ValueError("The inherited particle architecture fixes rank=alpha=8")
        self.ordinary, self.rank, self.scope = ordinary, rank, scope
        self.selected = targets(model, scope)
        self.names = list(self.selected)
        self.flags = [(p, p.requires_grad) for p in model.parameters()]
        model.eval().requires_grad_(False)
        if not ordinary:
            self.particles = nn.Parameter(torch.randn(128, 4, device=next(model.parameters()).device))
        self.branches = nn.ModuleList([
            Branch(module, None if ordinary else self.particles, api, rank, ordinary)
            for module in self.selected.values()
        ])
        for module, branch in zip(self.selected.values(), self.branches):
            module.forward = branch.forward

    @contextmanager
    def scaled(self, strength):
        if not math.isfinite(strength):
            raise ValueError("finite strength required")
        previous = [b.strength for b in self.branches]
        for branch in self.branches:
            branch.strength = float(strength)
        try:
            yield
        finally:
            for branch, value in zip(self.branches, previous):
                branch.strength = value

    def close(self):
        for module, branch in zip(self.selected.values(), self.branches):
            module.forward = branch.original
        for parameter, flag in self.flags:
            parameter.requires_grad_(flag)


def hidden(model, ids):
    """Real final normalized text hidden state, without materializing vocabulary logits."""
    value = torch.as_tensor(ids, device=next(model.parameters()).device, dtype=torch.long).reshape(1, -1)
    text = model.model.language_model
    result = text(input_ids=value, attention_mask=torch.ones_like(value), use_cache=False)
    return result.last_hidden_state[:, -1].float()


@torch.no_grad()
def linear_deltas(slider, strength=1.):
    if not slider.ordinary:
        raise ValueError("Nonlinear particles require calibration/distillation before weight export")
    return {name + ".weight": (branch.up.weight.float() @ branch.down.weight.float()).cpu() * strength
            for name, branch in zip(slider.names, slider.branches)}


@torch.no_grad()
def distill(slider, model, ids, api, *, rows=256):
    if slider.ordinary or not 2 <= rows <= 512:
        raise ValueError("particle teacher and 2..512 calibration rows required")
    captured = [[] for _ in slider.branches]
    handles = []
    for index, module in enumerate(slider.selected.values()):
        def capture(_module, args, index=index):
            count = sum(len(x) for x in captured[index])
            if count < rows:
                x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
                # Spread samples across prompts instead of using only their common prefix.
                count_here = min(16, rows - count, len(x))
                chosen = torch.linspace(0, len(x)-1, count_here, device=x.device).long()
                captured[index].append(x[chosen].cpu())
        handles.append(module.register_forward_pre_hook(capture))
    try:
        with slider.scaled(1.):
            for value in ids:
                hidden(model, value)
    finally:
        for handle in handles:
            handle.remove()
    state, metrics = {}, {}
    for index, branch in enumerate(slider.branches):
        x = torch.cat(captured[index]).to(branch.down.weight)
        desired = branch.bottleneck(x)
        down, ridge = api.fit_routed_down(x, desired, ridge_fraction=.01)
        prefix = f"branches.{index}."
        state[prefix + "down.weight"] = down.cpu().contiguous()
        state[prefix + "up.weight"] = branch.up.weight.detach().cpu().contiguous()
        residual = (x @ down.T - desired).square().mean().sqrt()
        metrics[slider.names[index]] = dict(rows=len(x), ridge=float(ridge), bottleneck_rms=float(residual),
            relative_rms=float(residual / desired.square().mean().sqrt().clamp_min(1e-8)))
    return state, metrics
