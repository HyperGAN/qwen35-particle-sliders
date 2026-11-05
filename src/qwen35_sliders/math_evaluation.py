"""Bounded generated-answer comparison on manifest-selected public GSM8K rows."""
import hashlib
import json
from pathlib import Path
import statistics

import torch

from .data import digest, write_json
from .evaluation import document_hash


def evaluate_math(workspace, output, *, deltas=None, strength=1., norm_gain=1.,
                  limit=16, max_tokens=1024, teacher=None, upstream=None, device="cuda",
                  offset=0, seed=1000):
    from datasets import load_dataset
    from safetensors.torch import load_file
    from delta_cli.local import read_workspace
    from delta_cli.scoring import score_answer
    from .training import load_host
    from .candidate import applied_teacher
    if deltas and teacher:
        raise ValueError("Choose merged deltas or nonlinear teacher")
    if not 1 <= limit <= 64 or not 1 <= max_tokens <= 8192:
        raise ValueError("Declared local evaluation cap exceeded")
    if type(offset) is not int or offset < 0 or type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Use a nonnegative selection offset and seed in [0, 2**63)")
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
    if offset + limit > len(selection["indices"]):
        raise ValueError("Requested slice exceeds the manifest's public selection")
    indices = selection["indices"][offset:offset+limit]
    expected_hashes = selection["doc_sha256"][offset:offset+limit]
    docs = [dataset[i] for i in indices]
    if [document_hash(doc) for doc in docs] != expected_hashes:
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
    generation = dict(config["generation"], max_tokens=max_tokens)
    records = []
    with applied_teacher(runner.model, teacher, strength, upstream) as teacher_identity:
        with generations.open("x") as stream:
            for index, doc in zip(indices, docs):
                response = runner.generate(doc["question"], generation, seed)
                gold = doc["answer"].rsplit("####", 1)[-1].strip()
                score = score_answer(dict(scorer="math", answer=gold), response["text"])
                record = dict(benchmark="gsm8k", example_index=index, sample=0, score=score, **response)
                stream.write(json.dumps(record)+"\n"); stream.flush()
                records.append(record)
                print(json.dumps({k:record[k] for k in ("example_index", "score", "tokens", "truncated")}), flush=True)
    protocol = dict(indices=indices, hashes=expected_hashes, generation=generation, seed=seed,
                    selection_offset=offset, prompt="zero-shot-user-question-chat")
    result = dict(schema="qwen35-local-gsm8k-advisory-v1", model_id=identity["model_id"], config_id=config["id"],
        validation_sha256=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest(),
        official_manifest_sha256=config["validation_sha256"], source_version_id=metadata["id"],
        artifact=dict(deltas_sha256=digest(deltas) if deltas else None, teacher_sha256=digest(teacher) if teacher else None,
                      teacher_engine=teacher_identity, strength=strength, norm_gain=norm_gain), generation=generation,
        validation_score=statistics.mean(r["score"] for r in records)*100,
        validation_benchmarks=dict(gsm8k=statistics.mean(r["score"] for r in records)*100),
        truncation_rate=statistics.mean(r["truncated"] for r in records), coverage=dict(examples=len(records), full_official_suite=False),
        runtime=provenance["runtime"], protocol=protocol, note="Local GSM8K subset with explicit generation cap; advisory only.")
    write_json(output, result)
    return result
