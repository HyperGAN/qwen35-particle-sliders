"""Verify lineage, canonical identity, and loaded patch-versus-delta inference."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .data import digest, write_json

# Immutable identity advertised by the registered qwen35-08b-v1 model config.
QWEN35_IDENTITY = "96336424b3b32c1b39d0ea157a559e148930a51bb019333b7bf6614391cfb025"


def verify_candidate(workspace, deltas, patch, *, strength=1., norm_gain=1., device="cuda"):
    from delta_cli.artifacts import content_hash, reconstruct
    from delta_cli.local import Runner, read_workspace, model_directory
    root, identity, config, metadata = read_workspace(workspace)
    if identity["model_id"] != "qwen35-08b-v1":
        raise ValueError("Canonical identity pin applies only to qwen35-08b-v1")
    patch = Path(patch)
    manifest = json.loads((patch / "manifest.json").read_text())
    receipt = json.loads((patch / "verification.json").read_text())
    if (receipt["deltas_sha256"] != digest(deltas) or receipt["strength"] != strength
            or receipt["norm_gain"] != norm_gain):
        raise ValueError("Declared candidate differs from the export receipt")
    torch.set_num_threads(4)
    with model_directory(root, metadata) as (parent, _):
        base = load_file(str(parent / "model.safetensors"))
        canonical_parent = content_hash(base, QWEN35_IDENTITY)
        if canonical_parent != metadata["content_hash"]:
            raise ValueError("Canonical parent content identity differs from registered lineage")
        reconstructed = reconstruct(parent / "model.safetensors", patch / "weights.safetensors", manifest,
                                    allowed_prefix="model.language_model.")
        canonical_candidate = content_hash(reconstructed, QWEN35_IDENTITY)
        del reconstructed, base
        direct = Runner(parent, device)
    parameters = dict(direct.model.named_parameters())
    with torch.no_grad():
        for name, delta in load_file(str(deltas)).items():
            p = parameters[name]
            p.copy_((p.float()+strength*delta.to(p.device)).to(p.dtype))
        if norm_gain != 1.:
            p = parameters["model.language_model.norm.weight"]
            p.copy_(((p.float()+1)*norm_gain-1).to(p.dtype))
    prompts = ["What is 17 times 23?", "Explain why water freezes.", "Write a Python function that adds two numbers."]
    def responses(runner):
        rows = []
        with torch.inference_mode():
            for prompt in prompts:
                rendered = runner.tokenizer.apply_chat_template([dict(role="user",content=prompt)],
                    tokenize=False,add_generation_prompt=True,enable_thinking=True)
                inputs = runner.tokenizer(rendered,return_tensors="pt",add_special_tokens=False).to(runner.device)
                rows.append(runner.model(**inputs,use_cache=False,logits_to_keep=1).logits.float().cpu())
        return rows
    expected = responses(direct)
    del direct, parameters
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with model_directory(root, metadata, patch=patch) as (directory, artifact):
        loaded = Runner(directory, device)
    actual = responses(loaded)
    if any(not torch.equal(a,b) for a,b in zip(actual,expected)):
        raise AssertionError("Loaded submission inference differs from the evaluated direct delta")
    result = dict(schema="qwen35-loaded-submission-verification-v1", parent_version_id=metadata["id"],
        identity_hash=QWEN35_IDENTITY, canonical_parent_content_hash=canonical_parent,
        canonical_candidate_content_hash=canonical_candidate, artifact_sha256=digest(patch / "weights.safetensors"),
        reconstructed_weights_sha256=artifact["weights_sha256"], direct_deltas_sha256=digest(deltas),
        strength=strength,norm_gain=norm_gain,exact_loaded_logits=True,probe_count=len(prompts),
        changed_prefix="model.language_model.",runtime=loaded.details,
        note="Probes verify serialization and load equivalence; they are not capability evaluations.")
    write_json(patch / "loaded-verification.json", result)
    return result
