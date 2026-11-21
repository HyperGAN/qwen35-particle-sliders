"""Bounded generated-answer comparison on manifest-selected public GSM8K rows."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics

import torch

from .data import digest, write_json
from .evaluation import document_hash


def evaluate_math(workspace, output, *, deltas=None, strength=None, norm_gain=1.,
                  limit=16, max_tokens=1024, teacher=None, upstream=None, device="cuda",
                  offset=0, seed=1000, step_gains=None, steering_plan=None, steering_core=None):
    from datasets import load_dataset
    from safetensors.torch import load_file
    from delta_cli.local import read_workspace
    from delta_cli.scoring import score_answer
    from .training import load_host
    from .candidate import applied_teacher
    from .token_strength import cached_strengths, resolve
    if deltas and teacher:
        raise ValueError("Choose merged deltas or nonlinear teacher")
    if not 1 <= limit <= 64 or not 1 <= max_tokens <= 8192:
        raise ValueError("Declared local evaluation cap exceeded")
    if strength is None:
        if steering_plan is not None or steering_core is not None:
            raise ValueError("Adaptive steering requires explicit calibrated strength")
        strength = 1.
    strengths = None
    if step_gains is not None:
        if teacher is None:
            raise ValueError("Token schedules require an explicit nonlinear RAW teacher")
        strengths = resolve(strength, max_tokens, step_gains)
    plan = None
    if steering_plan is not None or steering_core is not None:
        if steering_plan is None or steering_core is None or strengths is not None:
            raise ValueError("Adaptive steering requires a plan and core wheel, independently of fixed token gains")
        if deltas is not None or norm_gain != 1.:
            raise ValueError("Use the calibrated base with adaptive steering")
        from .adaptive_strength import CachedSteering, SteeringPlan
        plan = SteeringPlan(steering_plan, steering_core, strength)
        if (plan.intervention == "raw_adapter") != (teacher is not None):
            raise ValueError("Provide a nonlinear RAW teacher only for the raw_adapter intervention")
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
    teacher_strength = strength if plan is None else 0.
    with applied_teacher(runner.model, teacher, teacher_strength, upstream, expose_slider=True) as (teacher_identity, slider):
        adaptive = CachedSteering(runner.model, slider, plan, max_tokens) if plan is not None else None
        with generations.open("x") as stream:
            for index, doc in zip(indices, docs):
                schedule = nullcontext() if strengths is None else cached_strengths(runner.model, slider, strengths)
                with (adaptive.request() if adaptive is not None else schedule) as applied:
                    response = runner.generate(doc["question"], generation, seed)
                if adaptive is not None:
                    response["adaptive_steering"] = applied
                    applied = applied["effective_strengths"]
                if applied is not None:
                    if len(applied) != response["tokens"]:
                        raise RuntimeError("Qwen3.5 scheduled calls differ from generated token count")
                    response["effective_strengths"] = applied
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
        runtime=provenance["runtime"], protocol=protocol, generations_sha256=digest(generations), note="Local GSM8K subset with explicit generation cap; advisory only.")
    if step_gains is not None:
        result["artifact"]["step_gains"] = list(step_gains)
    if plan is not None:
        result["artifact"]["adaptive_steering"] = plan.identity
    write_json(output, result)
    return result
