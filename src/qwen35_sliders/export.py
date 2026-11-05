"""Exact dense replacement patches over immutable registered parent weights."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .data import digest, write_json


def build(workspace, deltas, output, *, strength=1., norm_gain=1.):
    from delta_cli.artifacts import validate_manifest, reconstruct
    from delta_cli.local import read_workspace, model_directory
    if not torch.isfinite(torch.tensor([strength, norm_gain])).all() or norm_gain <= 0:
        raise ValueError("finite strength and positive finite norm gain required")
    root, identity, config, metadata = read_workspace(workspace)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError(output)
    with model_directory(root, metadata) as (parent, provenance):
        base = load_file(str(parent / "model.safetensors"))
    delta = load_file(str(deltas))
    if set(delta) - set(base):
        raise ValueError("Delta names do not exist in parent")
    payload, mapping = {}, {}
    expected = dict(base)
    for name, value in delta.items():
        if not name.startswith("model.language_model.layers.") or not name.endswith(".weight"):
            raise ValueError("Only language projection weights can be changed by this adapter")
        if value.shape != base[name].shape or not torch.isfinite(value).all():
            raise ValueError("Invalid projection delta: " + name)
        expected[name] = (base[name].float() + strength * value.float()).to(base[name].dtype)
    if norm_gain != 1.:
        name = "model.language_model.norm.weight"
        if name not in base:
            raise ValueError("Expected genuine Qwen final norm")
        # Qwen RMSNorm uses (1 + weight), so scaling raw weight is incorrect.
        expected[name] = ((base[name].float() + 1.) * norm_gain - 1.).to(base[name].dtype)
    changed = []
    for name in sorted(base):
        if torch.equal(expected[name], base[name]):
            continue
        value_name = f"t{len(changed)}.values"
        payload[value_name] = expected[name].contiguous()
        mapping[name] = dict(values=value_name)
        changed.append(name)
    if not changed:
        raise ValueError("No changed tensors after registered dtype rounding")
    weights = output / "weights.safetensors"
    save_file(payload, str(weights))
    manifest = dict(schema_version="delta.submission.v1", reconstruction_version="delta-reconstruct-v1",
        model_config_id=identity["model_id"], parent_version_id=metadata["id"],
        parent_content_hash=metadata["content_hash"], format="dense-diff", operation="replace",
        artifacts=[dict(name=weights.name, sha256=digest(weights), size_bytes=weights.stat().st_size)], tensors=mapping)
    validate_manifest(manifest)
    write_json(output / "manifest.json", manifest)
    actual = reconstruct(root / "model/model.safetensors", weights, manifest)
    if set(actual) != set(base) or any(not torch.equal(actual[k], expected[k]) for k in base):
        raise AssertionError("Official reconstruction differs from merged checkpoint")
    if any(not torch.isfinite(v).all() for v in actual.values()):
        raise ValueError("Nonfinite reconstructed checkpoint")
    receipt = dict(schema="qwen35-exact-export-v1", parent_version_id=metadata["id"],
        parent_content_hash=metadata["content_hash"], evaluation_id=config["id"],
        parent_weights_sha256=provenance["weights_sha256"], deltas_sha256=digest(deltas),
        strength=strength, norm_gain=norm_gain, changed_tensors=changed,
        artifact_sha256=digest(weights), exact_reconstruction=True,
        all_other_tensors_unchanged=True, finite=True)
    write_json(output / "verification.json", receipt)
    return receipt
