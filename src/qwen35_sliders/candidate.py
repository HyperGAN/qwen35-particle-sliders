"""Scope a nonlinear particle teacher to an evaluation and restore model hooks."""
from contextlib import contextmanager

from safetensors.torch import load_file

from .host import Slider
from .upstream import load


@contextmanager
def applied_teacher(model, path, strength, upstream=None):
    if path is None:
        yield None
        return
    api = load(upstream)
    slider = Slider(model, api)
    try:
        slider.load_state_dict(load_file(str(path)), strict=True)
        with slider.scaled(strength):
            yield api.identity
    finally:
        slider.close()
