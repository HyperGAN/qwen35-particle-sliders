"""Explicit strength on Qwen3.5 prompt prefill and incremental token forwards."""

from contextlib import ExitStack, contextmanager
from math import isfinite
from sys import float_info


def resolve(strength, max_tokens, step_gains):
    if type(strength) not in (int, float) or not -float_info.max <= strength <= float_info.max:
        raise ValueError("Qwen3.5 nominal strength must be finite")
    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != max_tokens:
        raise ValueError("Qwen3.5 requires one gain per possible generated token")
    if any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("Qwen3.5 token gains must be finite and nonnegative")
    values = [strength * gain for gain in step_gains]
    if any(not isfinite(value) for value in values):
        raise ValueError("Qwen3.5 effective strengths must be finite")
    return values


@contextmanager
def cached_strengths(model, slider, strengths):
    scope = ExitStack()
    applied = []
    prefix = None

    def before(_model, _args, request):
        nonlocal prefix
        tokens = request.get("input_ids")
        cache = request.get("past_key_values")
        if tokens is None or tokens.ndim != 2 or not callable(getattr(cache, "get_seq_length", None)):
            raise RuntimeError("Qwen3.5 scheduled generation requires native cached token IDs")
        index = len(applied)
        if index == 0:
            prefix = tokens.shape[1]
        previous = prefix + index - 1 if index else 0
        if (index >= len(strengths) or tokens.shape != (1, 1 if index else prefix)
                or cache.get_seq_length() != previous):
            raise RuntimeError("Qwen3.5 cached token position differs from the requested schedule")
        strength = strengths[index]
        scope.enter_context(slider.scaled(strength))
        applied.append(strength)

    def after(_model, _args, _output):
        scope.close()

    with ExitStack() as handles:
        handles.callback(model.register_forward_pre_hook(before, with_kwargs=True).remove)
        handles.callback(model.register_forward_hook(after, always_call=True).remove)
        handles.callback(scope.close)
        yield applied
