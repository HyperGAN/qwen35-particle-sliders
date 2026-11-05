from types import SimpleNamespace
import os

import pytest
import torch
from torch import nn

from qwen35_sliders.host import Slider, targets, linear_deltas, distill, hidden
from qwen35_sliders.upstream import load


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = nn.ModuleDict(dict(in_proj_qkv=nn.Linear(12, 20, bias=False),
                                              out_proj=nn.Linear(20, 12, bias=False)))
        self.mlp = nn.ModuleDict(dict(up_proj=nn.Linear(12, 24, bias=False),
                                      down_proj=nn.Linear(24, 12, bias=False)))

    def forward(self, x):
        x = x + self.linear_attn["out_proj"](self.linear_attn["in_proj_qkv"](x).tanh())
        return x + self.mlp["down_proj"](self.mlp["up_proj"](x).tanh())


class TinyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 12)
        self.layers = nn.ModuleList([TinyLayer(), TinyLayer()])

    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids).cumsum(1)
        for layer in self.layers:
            x = layer(x)
        return SimpleNamespace(last_hidden_state=x)


class TinyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = TinyText()
        self.model.visual = nn.Linear(12, 12)


@pytest.fixture
def api():
    torch.set_num_threads(1)
    return load(os.environ.get("QWEN35_PG82_ROOT"))


def test_hybrid_targets_and_exact_zero_after_training(api):
    torch.manual_seed(11)
    model = TinyQwen()
    original = {k:v.clone() for k,v in model.state_dict().items()}
    off = hidden(model, [1, 2, 3]).detach()
    assert len(targets(model, "attention")) == 4
    assert len(targets(model, "all")) == 8
    slider = Slider(model, api)
    try:
        optimizer = torch.optim.Adam(slider.parameters(), lr=.01)
        for _ in range(2):
            optimizer.zero_grad()
            with slider.scaled(1.):
                hidden(model, [1,2,3]).square().mean().backward()
            optimizer.step()
        assert all(not p.requires_grad for p in model.parameters())
        assert torch.equal(hidden(model, [1,2,3]), off)
        with slider.scaled(1.):
            assert not torch.equal(hidden(model, [1,2,3]), off)
        assert all(torch.equal(v, original[k]) for k,v in model.state_dict().items())
        with pytest.raises(ValueError, match="distillation"):
            linear_deltas(slider)
    finally:
        slider.close()
    assert torch.equal(hidden(model, [1,2,3]), off)


def test_actual_upstream_step_and_distillation(api):
    torch.manual_seed(7)
    model = TinyQwen()
    slider = Slider(model, api)
    ids = [[1,2,3], [2,3,4], [4,5,6], [7,8,9]]
    try:
        with torch.no_grad():
            neutral = torch.cat([hidden(model, row) for row in ids])
            target = torch.cat([hidden(model, row+[10]) for row in ids])
        critic, g, d = api.game.build_game(slider, target, neutrals=neutral, critic="gmix",
            critic_config=dict(tokens=8, width=48, layers=1, heads=4, score_bound=8.))
        sampler = api.game.BridgeSampler(4, 7, batch_size=2)
        def predict(index, phase):
            with slider.scaled(1.):
                return critic.normalize(hidden(model, ids[index]))
        record = api.game.update(slider, critic, g, d, target, predict, sampler=sampler, step=0)
        assert record["particle_grad_norm"] > 0
        assert any(b.up.weight.abs().sum() > 0 for b in slider.branches)
        student, fit = distill(slider, model, ids, api, rows=32)
        assert all(torch.isfinite(v).all() for v in student.values())
        assert all(v["rows"] >= 2 for v in fit.values())
    finally:
        slider.close()
    student_slider = Slider(model, ordinary=True)
    try:
        student_slider.load_state_dict(student, strict=True)
        with student_slider.scaled(.7):
            expected = hidden(model, [3,4,5]).detach()
        deltas = linear_deltas(student_slider, .7)
    finally:
        student_slider.close()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in deltas:
                parameter.add_(deltas[name])
    torch.testing.assert_close(hidden(model, [3,4,5]), expected, atol=2e-6, rtol=2e-6)
