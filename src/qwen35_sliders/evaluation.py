"""Manifest-pinned local likelihood comparisons; never an official suite score."""
import copy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import statistics

import torch

from .data import digest, write_json


DEFAULT_TASKS = ["arc_challenge", "arc_easy", "boolq", "commonsense_qa",
                 "hellaswag", "openbookqa", "piqa", "winogrande"]


def document_hash(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def prepare_tasks(workspace, names, limit=None):
    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._yaml_loader import load_yaml
    from lm_eval.api.task import ConfigurableTask
    manifest = json.loads((Path(workspace) / "validation.json").read_text())
    if version("lm-eval") != manifest["versions"]["lm-eval"]:
        raise ValueError("lm-eval version differs from the public suite manifest")
    manager = TaskManager()
    tasks, samples, identities = [], {}, {}
    for name in names:
        matches = [group["selection"][name] for group in manifest["text"].values()
                   if name in group["selection"]]
        if len(matches) != 1:
            raise ValueError("Individual task must occur exactly once in public selection: " + name)
        selected = matches[0]
        # The index keeps !function values as strings; execute the same pinned
        # YAML loader as the real task factory before instantiating tasks.
        config = copy.deepcopy(load_yaml(manager.task_index[name].yaml_path, resolve_func=True))
        dataset = config["dataset_path"]
        if dataset not in manifest["dataset_revisions"]:
            raise ValueError("Task dataset is not pinned in public manifest: " + dataset)
        config["dataset_kwargs"] = dict(config.get("dataset_kwargs") or {},
                                        revision=manifest["dataset_revisions"][dataset])
        config["num_fewshot"] = selected["num_fewshot"]
        task = ConfigurableTask(config=config)
        if task.OUTPUT_TYPE != "multiple_choice":
            raise ValueError("Likelihood-only task required: " + name)
        docs = task.test_docs() if task.has_test_docs() else task.validation_docs()
        if len(docs) != selected["population"]:
            raise ValueError("Public dataset population differs: " + name)
        indices = selected["indices"][:limit] if limit else selected["indices"]
        expected_hashes = selected["doc_sha256"][:len(indices)]
        hashes = [document_hash(docs[index]) for index in indices]
        if hashes != expected_hashes:
            raise ValueError("Public document identity differs: " + name)
        tasks.append(task)
        samples[name] = indices
        identities[name] = dict(indices=indices, document_hashes=hashes, dataset_revision=config["dataset_kwargs"]["revision"])
    return tasks, samples, identities


def evaluate(workspace, output, *, deltas=None, strength=1., norm_gain=1.,
             names=None, limit=None, device="cuda", batch=4, teacher=None, upstream=None):
    if deltas and teacher:
        raise ValueError("Choose merged deltas or nonlinear teacher")
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from safetensors.torch import load_file
    from .training import load_host
    from .candidate import applied_teacher
    from delta_cli.local import read_workspace
    torch.set_num_threads(4)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".generations.jsonl").exists():
        raise FileExistsError(output)
    root, identity, config, metadata = read_workspace(workspace)
    if digest(root / "validation.json") != config["validation_sha256"]:
        raise ValueError("Official manifest was modified")
    names = names or DEFAULT_TASKS
    suite = json.loads((root / "validation.json").read_text())
    owners = {task:group for group in names for task in suite["text"][group]["selection"]}
    individual = list(owners)
    tasks, samples, selected = prepare_tasks(root, individual, limit)
    runner, provenance = load_host(root, device)
    with torch.no_grad():
        parameters = dict(runner.model.named_parameters())
        if deltas:
            for name, delta in load_file(str(deltas)).items():
                parameter = parameters[name]
                parameter.copy_((parameter.float()+strength*delta.to(parameter.device)).to(parameter.dtype))
        if norm_gain != 1.:
            parameter = parameters["model.language_model.norm.weight"]
            parameter.copy_(((parameter.float()+1)*norm_gain-1).to(parameter.dtype))
    model = HFLM(pretrained=runner.model, tokenizer=runner.tokenizer, backend="causal", device=device,
                 batch_size=batch, max_length=2048, dtype="bfloat16")
    with applied_teacher(runner.model, teacher, strength, upstream) as teacher_identity:
        result = evaluator.simple_evaluate(model=model, tasks=tasks, samples=samples, batch_size=batch,
            apply_chat_template=False, random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
            fewshot_random_seed=1234, bootstrap_iters=1000, log_samples=True)
    rows, scores = [], {name:[] for name in names}
    for name in individual:
        metric = "acc_norm" if "acc_norm" in result["samples"][name][0] else "acc"
        component_rows = []
        for record in result["samples"][name]:
            score = float(record[metric])
            group = owners[name]
            index = int(record["doc_id"]) if group == name else name + ":" + str(record["doc_id"])
            row = dict(benchmark=group, example_index=index, sample=0,
                       score=score, truncated=False, tokens=0, metric=metric)
            rows.append(row); component_rows.append(score)
        if len(component_rows) != len(samples[name]):
            raise ValueError("Evaluation coverage differs from requested public selection")
        scores[owners[name]].extend(component_rows)
    components = {name:statistics.mean(values)*100 for name,values in scores.items()}
    with output.with_suffix(".generations.jsonl").open("x") as stream:
        # Only per-example metric records, without prompts or generation text.
        for row in rows:
            stream.write(json.dumps(row)+"\n")
    protocol = dict(selected=selected, chat_template=False, max_length=2048, task_metrics={n: "acc_norm" if "acc_norm" in result["samples"][n][0] else "acc" for n in individual})
    report = dict(schema="qwen35-local-likelihood-advisory-v1", model_id=identity["model_id"],
        config_id=config["id"], source_version_id=metadata["id"], validation_sha256=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest(),
        official_manifest_sha256=config["validation_sha256"], generation=dict(kind="native-likelihood", chat_template=False),
        coverage=dict(benchmarks=names, examples=len(rows), full_official_suite=False),
        artifact=dict(deltas_sha256=digest(deltas) if deltas else None,
                      teacher_sha256=digest(teacher) if teacher else None,
                      teacher_engine=teacher_identity, strength=strength, norm_gain=norm_gain),
        validation_score=statistics.mean(components.values()), validation_benchmarks=components,
        protocol=protocol, runtime=provenance["runtime"], lm_eval_version=version("lm-eval"),
        note="Advisory subset likelihood mean; not official validation, not a private test score."
    )
    write_json(output, report)
    return report
