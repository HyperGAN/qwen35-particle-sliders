"""Public training data and disjoint development/calibration splits."""
import hashlib
import json
from pathlib import Path

import httpx


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_gsm(source, output, *, train=128, dev=32, calibration=32):
    raw = [json.loads(line) for line in Path(source).read_text().splitlines()]
    # Hash order is independent of input ordering, without seed shopping.
    raw.sort(key=lambda r: hashlib.sha256(r["question"].encode()).hexdigest())
    rows = []
    for row in raw[:train+dev+calibration]:
        explanation, answer = row["answer"].rsplit("####", 1)
        rows.append(dict(prompt=row["question"], reasoning=explanation.strip(),
                         answer=answer.strip(), source="openai/gsm8k:train"))
    if len(rows) != train+dev+calibration:
        raise ValueError("Insufficient training data")
    result = dict(schema="qwen35-slider-data-v1", source_sha256=digest(source),
        train=rows[:train], dev=rows[train:train+dev], calibration=rows[train+dev:])
    write_json(output, result)
    return result


def token_rows(tokenizer, rows, *, max_tokens=384, history_tokens=32):
    prepared = []
    for index, row in enumerate(rows):
        prompt = row["prompt"]
        neutral_text = tokenizer.apply_chat_template([dict(role="user", content=prompt)],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        neutral = tokenizer.encode(neutral_text, add_special_tokens=False)
        guidance = "Use this verified worked solution to answer accurately:\n" + row["reasoning"] + "\nAnswer: " + row["answer"]
        positive_text = tokenizer.apply_chat_template([dict(role="system", content=guidance),
            dict(role="user", content=prompt)], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        positive = tokenizer.encode(positive_text, add_special_tokens=False)
        solution = row["reasoning"] + "\n</think>\n\\boxed{" + row["answer"] + "}" + tokenizer.eos_token
        continuation = tokenizer.encode(solution, add_special_tokens=False)
        # Causal prefixes of the same verified continuation, including zero history.
        count = (index * 17) % (min(history_tokens, max(1, len(continuation)-1)) + 1)
        history = continuation[:count]
        if max(len(neutral), len(positive)) + count > max_tokens:
            continue
        prepared.append(dict(ids=neutral+history, positive_ids=positive+history,
                             supervised_ids=neutral+continuation[:max_tokens-len(neutral)],
                             prompt_tokens=len(neutral), source=row["source"], source_row_index=index,
                             neutral_text=neutral_text, positive_text=positive_text, history_ids=history))
    if len(prepared) < 2:
        raise ValueError("Not enough rows within the declared token budget")
    return prepared


def download(url, output):
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        response = client.get(url)
        response.raise_for_status()
    Path(output).write_bytes(response.content)
