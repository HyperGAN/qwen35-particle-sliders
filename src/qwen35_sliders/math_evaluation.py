"""Bounded generated-answer comparison on manifest-selected public GSM8K rows."""
import hashlib
import json
from pathlib import Path
import statistics

import torch

from .data import digest, write_json
from .evaluation import document_hash


def evaluate_math(workspace, output, *, deltas=None, strength=1., norm_gain=1.,
                  limit=16, max_tokens=1024, teacher=None, upstream=None, device="cuda"):
    from datasets import load_dataset
    from safetensors.torch import load_file
    from delta_cli.local import read_workspace
    from delta_cli.scoring import score_answer
    from .training import load_host
    from .host import Slider
    from .upstream import load
    if deltas and teacher:
        raise ValueError("Choose merged deltas or nonlinear teacher")
    if not 1 <= limit <= 64 or not 1 <= max_tokens <= 8192:
        raise ValueError("Declared local evaluation cap exceeded")
    output = Path(output)
    generations = output.with_suffix(".generations.jsonl")
    if output.exists() or generations.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root, identity, config, metadata = read_workspace(workspace)
    if digest(root / "validation.json") != config["validation_sha256"]:
        raise ValueError("Official manifest changed")
    suite = json.loads((root / "validation.json").read_text())
    selection = suite["text"]["gsm8k"]["selection"]["gsm8k"]
    dataset = load_dataset("openai/gsm8k", "main", split="test", revision=suite["dataset_revisions"]["openai/gsm8k"])
    if len(dataset) != selection["population"]:
        raise ValueError("GSM8K population differs")
    indices = selection["indices"][:limit]
    docs = [dataset[i] for i in indices]
    if [document_hash(doc) for doc in docs] != selection["doc_sha256"][:limit]:
        raise ValueError("Public GSM8K document hashes differ")
    torch.set_num_threads(4)
    runner, provenance = load_host(root, device)
    with torch.no_grad():
        parameters = dict(runner.model.named_parameters())
        if deltas:
            for name, delta in load_file(str(deltas)).items():
                p = parameters[name]
                p.copy_((p.float()+strength*delta.to(p.device)).to(p.dtype))
        if norm_gain != 1.:
            p = parameters["model.language_model.norm.weight"]
            p.copy_(((p.float()+1)*norm_gain-1).to(p.dtype))
    slider = None
    if teacher:
        slider = Slider(runner.model, load(upstream))
        slider.load_state_dict(load_file(str(teacher)), strict=True)
    generation = dict(config["generation"], max_tokens=max_tokens)
    records = []
    try:
        with generations.open("x") as stream:
            for index, doc in zip(indices, docs):
                if slider:
                    with slider.scaled(strength):
                        response = runner.generate(doc["question"], generation, 1000)
                else:
                    response = runner.generate(doc["question"], generation, 1000)
                gold = doc["answer"].rsplit("####", 1)[-1].strip()
                score = score_answer(dict(scorer="math", answer=gold), response["text"])
                record = dict(benchmark="gsm8k", example_index=index, sample=0, score=score, **response)
                stream.write(json.dumps(record)+"\n"); stream.flush()
                records.append(record)
                print(json.dumps({k:record[k] for k in ("example_index", "score", "tokens", "truncated")}), flush=True)
    finally:
        if slider:
            slider.close()
    protocol = dict(indices=indices, hashes=selection["doc_sha256"][:limit], generation=generation, seed=1000)
    result = dict(schema="qwen35-local-gsm8k-advisory-v1", model_id=identity["model_id"], config_id=config["id"],
        validation_sha256=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest(),
        official_manifest_sha256=config["validation_sha256"], source_version_id=metadata["id"],
        artifact=dict(deltas_sha256=digest(deltas) if deltas else None, teacher_sha256=digest(teacher) if teacher else None,
                      strength=strength, norm_gain=norm_gain), generation=generation,
        validation_score=statistics.mean(r["score"] for r in records)*100,
        validation_benchmarks=dict(gsm8k=statistics.mean(r["score"] for r in records)*100),
        truncation_rate=statistics.mean(r["truncated"] for r in records), coverage=dict(examples=len(records), full_official_suite=False),
        runtime=provenance["runtime"], protocol=protocol, note="Local GSM8K subset with explicit generation cap; advisory only.")
    write_json(output, result)
    return result
