"""Persist exact frozen-model positive/neutral hidden-state pairs together."""
import base64
import hashlib
import inspect

import torch

from .host import hidden
from .teacher_data import Cache, fingerprint


def encode(tensor):
    value = tensor.detach().cpu().contiguous()
    if value.dtype != torch.float32 or value.ndim != 2 or value.shape[0] != 1 or not torch.isfinite(value).all():
        raise ValueError("Cache finite float32 final hidden states of shape [1,width]")
    return dict(shape=list(value.shape), dtype="float32",
                bytes_base64=base64.b64encode(value.numpy().tobytes()).decode())


def decode(packet):
    if packet["dtype"] != "float32" or len(packet["shape"]) != 2 or packet["shape"][0] != 1 or packet["shape"][1] < 1:
        raise ValueError("Invalid cached hidden-state format")
    raw = base64.b64decode(packet["bytes_base64"], validate=True)
    if len(raw) != 4 * packet["shape"][1]:
        raise ValueError("Cached hidden-state byte length differs")
    value = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(packet["shape"]).clone()
    if not torch.isfinite(value).all():
        raise ValueError("Nonfinite cached hidden state")
    return value


class FrozenPairCache:
    def __init__(self, root, model_identity):
        self.cache = Cache(root)
        self.identity = model_identity
        self.extractor = dict(protocol="final-normalized-text-hidden-last-token-fp32-v1",
            implementation_sha256=hashlib.sha256(inspect.getsource(hidden).encode()).hexdigest())
        self.references = []
        self.hits = self.computed = 0

    def get_pair(self, row, compute, device):
        recipe = dict(schema="qwen35-frozen-pair-v1", model=self.identity, extractor=self.extractor,
            neutral_ids=row["ids"], positive_ids=row["positive_ids"],
            neutral_text=row.get("neutral_text"), positive_text=row.get("positive_text"),
            history_ids=row.get("history_ids"), provenance=row.get("pair_provenance"))
        if not recipe["neutral_ids"] or not recipe["positive_ids"]:
            raise ValueError("Both paired model inputs require token IDs")
        def calculate():
            neutral, target = compute()
            if neutral.shape != target.shape:
                raise ValueError("Positive and neutral hidden-state dimensions differ")
            return dict(neutral=encode(neutral), target=encode(target))
        value, hit, key = self.cache.once("latent-pair", recipe, calculate)
        neutral, target = decode(value["neutral"]), decode(value["target"])
        if neutral.shape != target.shape:
            raise ValueError("Cached pair dimensions differ")
        self.hits += int(hit); self.computed += int(not hit)
        self.references.append(dict(cache_key=key, provenance=recipe["provenance"],
            neutral_ids_sha256=fingerprint(recipe["neutral_ids"]), positive_ids_sha256=fingerprint(recipe["positive_ids"]),
            neutral_tensor_sha256=fingerprint(value["neutral"]), positive_tensor_sha256=fingerprint(value["target"])))
        return neutral.to(device), target.to(device)
