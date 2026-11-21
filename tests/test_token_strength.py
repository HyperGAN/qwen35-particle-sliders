from contextlib import contextmanager

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from qwen35_sliders.token_strength import cached_strengths, resolve


def test_qwen35_native_cache_uses_scheduled_strength_in_order():
    config = Qwen3_5TextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        layer_types=["full_attention", "full_attention"], pad_token_id=0, bos_token_id=1,
        eos_token_id=None,
    )
    model = Qwen3_5ForCausalLM(config).eval()
    active = 0.0
    observed = []

    class Slider:
        @staticmethod
        @contextmanager
        def scaled(value):
            nonlocal active
            previous = active
            active = value
            try:
                yield
            finally:
                active = previous

    ids = torch.tensor([[1, 3, 4]])
    request = dict(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=3, do_sample=False)
    base = model.generate(**request)
    with cached_strengths(model, Slider(), [0.5] * 3):
        constant = model.generate(**request)
    assert torch.equal(base, constant)

    probe = model.model.register_forward_pre_hook(lambda *_: observed.append(active))
    try:
        values = resolve(0.5, 3, [1.5, 1, 0.5])
        with cached_strengths(model, Slider(), values) as applied:
            model.generate(**request)
    finally:
        probe.remove()
    assert observed == applied == [0.75, 0.5, 0.25]
    assert active == 0.0 and not model._forward_pre_hooks and not model._forward_hooks
