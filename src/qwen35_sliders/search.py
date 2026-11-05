"""Fixed finite search, separate routes, and a selected final-horizon run."""
import json
from pathlib import Path
import subprocess
import sys

from .data import write_json


def search(spec):
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    if len(spec["trials"]) > 12 or spec["final_steps"] > 1600:
        raise ValueError("Declared local search cap exceeded")
    records = []
    for index, trial in enumerate(spec["trials"]):
        config = dict(spec["base"], **trial, output=str(root / f"trial-{index:02d}"))
        if not 1 <= config["steps"] <= 256:
            raise ValueError("Search trial must be bounded to 1..256 steps")
        path = root / f"trial-{index:02d}.json"
        write_json(path, config)
        result_path = Path(config["output"]) / "result.json"
        if not result_path.exists():
            with (root / f"trial-{index:02d}.log").open("a") as log:
                subprocess.run([sys.executable, "-m", "qwen35_sliders.cli", "train-particle", str(path)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(result_path.read_text())
        if result.get("upstream", {}).get("particlegan_version") != "0.9.5" or result.get("metric") != "raw_hidden_rms":
            raise ValueError("Cached search result belongs to another engine; use a fresh output directory")
        # Compare raw held-out hidden errors on the same fixed development rows.
        ratio = result["student"]["on_rms"] / result["student"]["off_rms"]
        records.append(dict(index=index, config=config, result=result, student_relative_rms=ratio))
        write_json(root / "search-results.json", records)
        print(json.dumps(dict(trial=index, student_relative_rms=ratio)), flush=True)
    best = min(records, key=lambda row:row["student_relative_rms"])
    final = dict(best["config"], steps=spec["final_steps"], output=str(root / "final"))
    # Start from the common initial seed; this is a declared final horizon, not a relabelled resume.
    write_json(root / "selected.json", dict(trial=best["index"], criterion="distilled student held-out relative hidden RMS", config=final))
    write_json(root / "final.json", final)
    if not (root / "final/result.json").exists():
        with (root / "final.log").open("a") as log:
            subprocess.run([sys.executable, "-m", "qwen35_sliders.cli", "train-particle", str(root / "final.json")],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
