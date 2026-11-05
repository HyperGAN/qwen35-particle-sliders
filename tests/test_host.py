from types import SimpleNamespace

import pytest
import torch
from torch import nn

from qwen35_sliders.host import Slider, targets, linear_deltas, distill, hidden
from qwen35_sliders.upstream import load
from qwen35_sliders.particle import make_game


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
    return load()


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
        training = [dict(ids=row, neutral=neutral[i:i+1], target=target[i:i+1]) for i,row in enumerate(ids)]
        game = make_game(model, slider, training, dict(steps=3, batch=2, lr=1e-5), api)
        before = slider.particles.detach().clone()
        record = game.step()
        assert record["step"] == 1 and record["prior_regularization"] >= 0
        assert not torch.equal(before, slider.particles)
        packet = game.state_dict()
        assert packet["config"]["pg_version"] == "0.9.5"
        assert packet["config"]["penalty_input_domain"] == "continuous"
        expected = game.step()
        game.load_state_dict(packet)
        assert game.step() == expected
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


def test_training_export_and_interrupted_resume_match(api, tmp_path, monkeypatch):
    """Exercise the actual training entrypoint, checkpoint, EMA and student export."""
    import json
    from safetensors.torch import load_file
    import qwen35_sliders.training as training_module

    data = tmp_path / 'data.json'
    data.write_text(json.dumps({key:[0,1,2,3] for key in ('train','dev','calibration')}))
    ids = [[1,2,3], [2,3,4], [4,5,6], [7,8,9]]
    monkeypatch.setattr(training_module, 'token_rows', lambda *args, **kwargs:
        [dict(ids=row, positive_ids=row+[10]) for row in ids])

    def runner():
        with torch.random.fork_rng():
            torch.manual_seed(42)
            model = TinyQwen()
        return SimpleNamespace(model=model, tokenizer=None), {'parent':'fixture-base'}

    config = dict(data=str(data), device='cpu', seed=7, steps=3, batch=2,
                  checkpoint_every=1, calibration_rows=32, lr=1e-5)
    full = tmp_path / 'full'
    training_module.train_particle(dict(config, output=str(full)), runner=runner())

    interrupted = tmp_path / 'interrupted'
    factory = training_module.make_game
    def interrupting_factory(*args):
        game = factory(*args)
        original = game.step
        def step():
            if game.completed_steps == 1:
                raise RuntimeError('simulated interruption')
            return original()
        game.step = step
        return game
    monkeypatch.setattr(training_module, 'make_game', interrupting_factory)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        training_module.train_particle(dict(config, output=str(interrupted)), runner=runner())
    monkeypatch.setattr(training_module, 'make_game', factory)
    resumed = tmp_path / 'resumed'
    training_module.train_particle(dict(config, output=str(resumed),
        resume=str(interrupted/'continuation.pt')), runner=runner())
    for filename in ('particle.safetensors','distilled.safetensors','deltas.safetensors'):
        left, right = load_file(full/filename), load_file(resumed/filename)
        assert left.keys() == right.keys()
        assert all(torch.equal(left[k], right[k]) for k in left)
    contract = json.loads((resumed/'contract.json').read_text())
    assert contract['formulation']['pg_version'] == '0.9.5'
    assert contract['formulation']['recipe']['name'] == 'bispf_optimized'
    assert contract['formulation']['penalty_input_domain'] == 'continuous'
