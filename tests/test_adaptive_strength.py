"""CPU-only checks on the original conditional Qwen3.5 and Delta League driver."""
from contextlib import nullcontext
from hashlib import sha256
import json
import os
from zipfile import ZipFile

from delta_cli.local import Runner
import pytest
from safetensors.torch import save_file
import torch
from transformers import BatchEncoding, Qwen3_5Config, Qwen3_5ForConditionalGeneration

from qwen35_sliders.adaptive_strength import CachedSteering, SteeringPlan
from qwen35_sliders.candidate import applied_teacher
from qwen35_sliders.host import Slider
from qwen35_sliders.steering_core import load_steering_core
from qwen35_sliders.token_strength import cached_strengths
from qwen35_sliders.upstream import load


@pytest.fixture
def wheel():
    path = os.getenv("QWEN35_STEERING_CORE_WHEEL")
    if path is None:
        pytest.skip("Set QWEN35_STEERING_CORE_WHEEL to the exact shared #269 wheel")
    return path


@pytest.fixture
def runner():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    config = Qwen3_5Config(
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=1, linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"], pad_token_id=0,
            bos_token_id=1, eos_token_id=None),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16),
    )
    native = object.__new__(Runner)
    native.model = Qwen3_5ForConditionalGeneration(config).eval()
    native.device = "cpu"
    native.torch = torch

    class Tokenizer:
        eos_token = pad_token = None

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, prompt, **kwargs):
            return BatchEncoding(dict(input_ids=torch.tensor([[1, 3, 4]]),
                                     attention_mask=torch.ones((1, 3), dtype=torch.long)))

        def decode(self, tokens, **kwargs):
            return str(tokens.tolist())

    native.tokenizer = Tokenizer()
    return native


def settings(tmp_path, wheel, strength=0.5, **options):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(layer=0, evidence_ref="native CPU fixture", **options)))
    return SteeringPlan(plan, wheel, strength)


def generate(runner, context=None):
    logits = []
    with (context if context is not None else nullcontext()) as trace:
        handle = runner.model.register_forward_hook(lambda _m, _a, output: logits.append(output.logits.detach().clone()))
        try:
            result = runner.generate("Fixture", dict(temperature=0, presence_penalty=0, max_tokens=4), seed=10)
        finally:
            handle.remove()
    return result, logits, trace


def test_real_native_cfp_reads_pre_edit_residual_and_changes_only_last_prefill_row(tmp_path, runner, wheel):
    layer = runner.model.model.language_model.layers[0]
    before = []
    handle = layer.register_forward_pre_hook(lambda _m, args: before.append(args[0].clone()))
    plan = settings(tmp_path, wheel, method="cfp", intervention="residual", parameter=100.,
                    direction=[1.] + [0.] * 15)
    policy = CachedSteering(runner.model, None, plan, 4)
    seen = []
    # The actual layer receives an edited residual only after the probe's read.
    observed_norms = layer.input_layernorm.register_forward_pre_hook(lambda _m, args: seen.append(args[0].clone()))
    try:
        result, _, receipt = generate(runner, policy.request())
    finally:
        observed_norms.remove()
        handle.remove()
    assert result["tokens"] == receipt["usage"]["eligible"] == receipt["usage"]["fired"] == 4
    assert [step["cache_position"] for step in receipt["trace"]] == [2, 3, 4, 5]
    assert [step["alignment"] for step in receipt["trace"]] == [x[0, -1, 0].item() for x in before]
    assert torch.equal(before[0][:, :-1], seen[0][:, :-1])
    for original, edited in zip(before, seen, strict=True):
        expected = original[:, -1].clone()
        expected[:, 0] += 0.5
        assert torch.equal(expected, edited[:, -1])
    assert not runner.model._forward_hooks and not runner.model._forward_pre_hooks


def test_matching_raw_endpoints_replay_through_existing_teacher_loader(tmp_path, runner, wheel):
    model = runner.model
    path = tmp_path / "teacher.safetensors"
    slider = Slider(model, load())
    try:
        with torch.no_grad():
            for branch in slider.branches:
                branch.up.weight.fill_(0.03)
        save_file({name: tensor.detach().contiguous() for name, tensor in slider.state_dict().items()},
                  path, metadata={"weights_kind": "raw"})
    finally:
        slider.close()
    _, off, _ = generate(runner)
    with applied_teacher(model, path, 0., expose_slider=True) as (_, slider):
        for rate in (0., 0.5, 1.):
            plan = settings(tmp_path, wheel, method="random", intervention="raw_adapter", parameter=rate, seed=7)
            policy = CachedSteering(model, slider, plan, 4)
            _, actual, receipt = generate(runner, policy.request())
            _, replay, _ = generate(runner, cached_strengths(model, slider, receipt["effective_strengths"]))
            assert all(torch.equal(left, right) for left, right in zip(actual, replay, strict=True))
            assert receipt["effective_strengths"][0] == 0.
            assert receipt["usage"]["eligible"] == 3 and len(receipt["trace"]) == 3
            if rate == 0.:
                assert all(torch.equal(left, right) for left, right in zip(actual, off, strict=True))
            assert all(branch.strength == 0. for branch in slider.branches)
    assert not model._forward_hooks and not model._forward_pre_hooks


def test_native_clabs_resets_each_request_and_stops_at_eos(tmp_path, runner, wheel):
    model = runner.model
    with torch.no_grad():
        model.lm_head.weight.zero_()
    model.generation_config.eos_token_id = 0
    config = dict(mean=0., std=1., epsilon=1e-6, setpoint=1., proportional_gain=0.1, integral_gain=0.2, beta=0.3)
    plan = settings(tmp_path, wheel, method="clabs", intervention="residual", parameter=0.5, seed=3,
                    clabs=config, direction=[1.] + [0.] * 15)
    policy = CachedSteering(model, None, plan, 4)
    first, _, one = generate(runner, policy.request())
    second, _, two = generate(runner, policy.request())
    assert first["tokens"] == second["tokens"] == one["usage"]["eligible"] == 1
    assert one == two and len(one["trace"]) == 1
    step = one["trace"][0]
    assert step["probability"] == plan.new_gate().step(step["alignment"]).probability
    with pytest.raises(RuntimeError, match="native failure"):
        with policy.request():
            assert model._forward_pre_hooks
            raise RuntimeError("native failure")
    assert not model._forward_pre_hooks and not model._forward_hooks
    assert not model.model.language_model.layers[0]._forward_pre_hooks


def test_fitted_shared_s_psr_keeps_zero_exact_and_validates_digest(tmp_path, runner, wheel):
    original_rng = torch.random.get_rng_state()
    artifact = tmp_path / "fitted.safetensors"
    save_file({"probe.weight": torch.zeros(1, 16), "probe.bias": torch.ones(1),
               "direction.weight": torch.tensor([[1.] + [0.] * 15])}, artifact)
    options = dict(method="s_psr", intervention="residual", weights=artifact.name,
                   weights_sha256=sha256(artifact.read_bytes()).hexdigest(), weights_kind="raw", ema_decay=None)
    model = runner.model
    zero = CachedSteering(model, None, settings(tmp_path, wheel, strength=0., **options), 4)
    fitted = CachedSteering(model, None, settings(tmp_path, wheel, **options), 4)
    assert torch.equal(original_rng, torch.random.get_rng_state())
    _, off, _ = generate(runner)
    _, zero_logits, zero_trace = generate(runner, zero.request())
    _, applied, trace = generate(runner, fitted.request())
    assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(off, zero_logits, strict=True))
    assert not torch.equal(off[0], applied[0])
    assert trace["effective_strengths"] == [0.5] * 4
    assert zero_trace["effective_strengths"] == [0.] * 4 and not zero_trace["trace"]
    with pytest.raises(ValueError, match="digest"):
        settings(tmp_path, wheel, **{**options, "weights_sha256": "0" * 64})


def test_changed_core_member_is_rejected_even_after_a_valid_load(tmp_path, wheel):
    core = load_steering_core(wheel)
    bad = tmp_path / "bad.whl"
    with ZipFile(wheel) as source, ZipFile(bad, "w") as target:
        for member in ("adaptive_steering", "prompt_steering"):
            name = f"particle_sliders/{member}.py"
            content = source.read(name)
            target.writestr(name, b"!" + content[1:] if member == "adaptive_steering" else content)
    with pytest.raises(ValueError, match="checksum"):
        load_steering_core(bad)
    assert load_steering_core(wheel) is core


def test_evaluate_math_records_matched_protocol_and_native_gate(tmp_path, runner, wheel, monkeypatch):
    from delta_cli import local, scoring
    import datasets
    from qwen35_sliders import training
    from qwen35_sliders.data import digest
    from qwen35_sliders.evaluation import document_hash
    from qwen35_sliders.math_evaluation import evaluate_math

    rows = [dict(question="Fixture question", answer="Fixture answer #### 1")]
    validation = dict(text={"gsm8k": {"selection": {"gsm8k": dict(
        population=1, indices=[0], doc_sha256=[document_hash(rows[0])])}}},
        dataset_revisions={"openai/gsm8k": "fixture"})
    manifest = tmp_path / "validation.json"
    manifest.write_text(json.dumps(validation))
    config = dict(id="fixture", validation_sha256=digest(manifest), generation=dict(temperature=0))
    monkeypatch.setattr(local, "read_workspace", lambda _root: (
        tmp_path, {"model_id": "fixture"}, config, {"id": "fixture"}))
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: rows)
    monkeypatch.setattr(training, "load_host", lambda *args: (runner, {"runtime": {"device": "cpu"}}))
    monkeypatch.setattr(scoring, "score_answer", lambda *args: 0.)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(method="random", intervention="residual", layer=0, evidence_ref="fixture",
                                    parameter=1., seed=7, direction=[1.] + [0.] * 15)))
    with pytest.raises(ValueError, match="explicit calibrated strength"):
        evaluate_math(tmp_path, tmp_path / "bad.json", steering_plan=plan, steering_core=wheel)
    off = evaluate_math(tmp_path, tmp_path / "off.json", limit=1, max_tokens=4, device="cpu", strength=0.)
    got = evaluate_math(tmp_path, tmp_path / "adaptive.json", limit=1, max_tokens=4, device="cpu", strength=0.5,
                        steering_plan=plan, steering_core=wheel)
    answer = json.loads((tmp_path / "adaptive.generations.jsonl").read_text())
    assert off["validation_sha256"] == got["validation_sha256"]
    assert got["artifact"]["adaptive_steering"]["plan_sha256"] == digest(plan)
    assert answer["adaptive_steering"]["usage"] == {"eligible": 4, "fired": 4}
    assert answer["effective_strengths"] == [0.5] * answer["tokens"]
