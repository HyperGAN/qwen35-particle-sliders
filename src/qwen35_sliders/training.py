"""Upstream GAN callbacks and a separately labelled supervised LoRA baseline."""
from contextlib import contextmanager
import json
from pathlib import Path
import time

import torch
from safetensors.torch import save_file

from .data import digest, token_rows, write_json
from .host import Slider, hidden, distill, linear_deltas


def load_host(workspace, device="cuda"):
    from delta_cli.local import Runner, read_workspace, model_directory
    workspace, identity, config, metadata = read_workspace(workspace)
    with model_directory(workspace, metadata) as (path, provenance):
        runner = Runner(path, device)
    runner.model.eval().requires_grad_(False)
    return runner, dict(workspace=identity, parent=metadata["content_hash"],
                        provenance=provenance, runtime=runner.details)


@contextmanager
def state(slider, values):
    previous = {k: v.detach().clone() for k, v in slider.state_dict().items()}
    slider.load_state_dict(values, strict=True)
    try:
        yield
    finally:
        slider.load_state_dict(previous, strict=True)


@torch.no_grad()
def prepare(model, slider, rows):
    with slider.scaled(0.):
        return [dict(**row, neutral=hidden(model, row["ids"]).detach(),
                     target=hidden(model, row["positive_ids"]).detach()) for row in rows]


@torch.no_grad()
def metrics(model, slider, rows, critic):
    def rms(strength):
        with slider.scaled(strength):
            return float(torch.cat([critic.normalize(hidden(model, row["ids"])) -
                        critic.normalize(row["target"]) for row in rows]).square().mean().sqrt())
    return dict(off_rms=rms(0.), on_rms=rms(1.))


def train_particle(config, *, runner=None):
    from .upstream import load
    api = load(config.get("upstream"))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("Completed run exists; select another output")
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.manual_seed(config.get("seed", 7))
    runner, identity = load_host(config["workspace"], config.get("device", "cuda")) if runner is None else runner
    model = runner.model
    data = json.loads(Path(config["data"]).read_text())
    splits = {key: token_rows(runner.tokenizer, data[key], max_tokens=config.get("max_tokens", 384))
              for key in ("train", "dev", "calibration")}
    slider = Slider(model, api, scope=config.get("scope", "all"))
    start = time.monotonic()
    try:
        training = prepare(model, slider, splits["train"])
        development = prepare(model, slider, splits["dev"])
        target = torch.cat([r["target"] for r in training])
        neutral = torch.cat([r["neutral"] for r in training])
        stamp = api.stamp.as_dict()
        critic_config = {key: stamp["critic_"+key] for key in ("tokens", "width", "layers", "heads", "score_bound")}
        critic, g, d = api.game.build_game(slider, target, critic=stamp["critic"],
                                          critic_config=critic_config, neutrals=neutral)
        critic.noise_decay_steps = stamp["noise_decay_steps"]
        rate = config.get("lr", stamp["g_lr"])
        for group in g.param_groups:
            group["lr"] = rate * (config.get("particle_lr_mult", 10.) if group["role"] == "particles" else 1.)
        for group in d.param_groups:
            group["lr"] = rate * config.get("d_lr_mult", 1.5)
        sampler = api.game.BridgeSampler(len(training), config.get("seed", 7), batch_size=config.get("batch", 4))
        ema = api.game.initialize_ema(slider)
        contract = dict(config=config, identity=identity, data_sha256=digest(config["data"]),
                        formulation=stamp, names=slider.names, rows={k:len(v) for k,v in splits.items()})
        write_json(output / "contract.json", contract)
        initial = metrics(model, slider, development, critic)
        write_json(output / "initial.json", initial)
        completed = 0
        if config.get("resume"):
            packet = torch.load(config["resume"], map_location=target.device, weights_only=True)
            # Output and resume paths are operational; the training recipe must match.
            old = {k:v for k,v in packet["contract"]["config"].items() if k not in {"output", "resume"}}
            new = {k:v for k,v in config.items() if k not in {"output", "resume"}}
            if old != new or packet["contract"]["data_sha256"] != contract["data_sha256"] or packet["contract"]["identity"] != identity:
                raise ValueError("Resume recipe, runtime, data or base differs")
            slider.load_state_dict(packet["slider"], strict=True)
            critic.load_state_dict(packet["critic"], strict=True)
            g.load_state_dict(packet["g"]); d.load_state_dict(packet["d"])
            sampler.load_state_dict({**packet["sampler"], "rngs": {k:v.cpu() for k,v in packet["sampler"]["rngs"].items()}})
            ema = packet["ema"]
            torch.set_rng_state(packet["torch_rng"].cpu())
            torch.cuda.set_rng_state_all([x.cpu() for x in packet["cuda_rng"]])
            completed = packet["step"]
        def checkpoint(step):
            packet = dict(contract=contract, step=step, slider=slider.state_dict(), critic=critic.state_dict(),
                          g=g.state_dict(), d=d.state_dict(), sampler=sampler.state_dict(), ema=ema,
                          torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
            temporary = output / "continuation.tmp"
            torch.save(packet, temporary)
            temporary.replace(output / "continuation.pt")
        with (output / "metrics.jsonl").open("a") as stream:
            for step in range(completed, config["steps"]):
                def predict(index, phase):
                    with slider.scaled(1.):
                        return critic.normalize(hidden(model, training[index]["ids"]))
                record = api.game.update(slider, critic, g, d, target, predict, sampler=sampler, step=step)
                api.game.update_ema(ema, slider)
                record.update(step=step+1, elapsed_seconds=time.monotonic()-start)
                stream.write(json.dumps(record, allow_nan=False)+"\n"); stream.flush()
                if (step+1) % config.get("checkpoint_every", 16) == 0 or step+1 == config["steps"]:
                    checkpoint(step+1)
                    print(json.dumps({k:record[k] for k in ("step", "g_adv", "d_loss", "particle_gan_grad_norm", "elapsed_seconds")}), flush=True)
        raw_metrics = metrics(model, slider, development, critic)
        with state(slider, ema):
            ema_metrics = metrics(model, slider, development, critic)
        # Selection uses disjoint training-source development data, not the public test suite.
        use_ema = ema_metrics["on_rms"] < raw_metrics["on_rms"]
        if use_ema:
            slider.load_state_dict(ema, strict=True)
        save_file({k:v.detach().cpu().contiguous() for k,v in slider.state_dict().items()}, str(output / "particle.safetensors"))
        student, calibration = distill(slider, model, [r["ids"] for r in splits["calibration"]], api,
                                       rows=config.get("calibration_rows", 256))
        save_file(student, str(output / "distilled.safetensors"))
        write_json(output / "calibration.json", calibration)
        names = slider.names
        slider.close()
        slider = Slider(model, scope=config.get("scope", "all"), ordinary=True)
        slider.load_state_dict(student, strict=True)
        student_metrics = metrics(model, slider, development, critic)
        deltas = linear_deltas(slider)
        save_file(deltas, str(output / "deltas.safetensors"))
        result = dict(kind="particle-distilled", steps=config["steps"], raw=raw_metrics, ema=ema_metrics,
                      selected="ema" if use_ema else "raw", student=student_metrics, initial=initial,
                      targets=names, elapsed_seconds=time.monotonic()-start,
                      deltas_sha256=digest(output / "deltas.safetensors"))
        write_json(output / "result.json", result)
        return result
    finally:
        slider.close()


def train_lora(config):
    """Separate supervised comparator. No claim that this is the particle objective."""
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.manual_seed(config.get("seed", 7))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError(output)
    runner, identity = load_host(config["workspace"], config.get("device", "cuda"))
    model = runner.model
    data = json.loads(Path(config["data"]).read_text())
    splits = {key:token_rows(runner.tokenizer, data[key], max_tokens=config.get("max_tokens", 384)) for key in ("train", "dev")}
    slider = Slider(model, ordinary=True, scope=config.get("scope", "all"), rank=config.get("rank", 8))
    optimizer = torch.optim.AdamW(slider.parameters(), lr=config.get("lr", 1e-4), weight_decay=.01)
    write_json(output / "contract.json", dict(config=config, identity=identity, data_sha256=digest(config["data"]), names=slider.names))
    def loss(row):
        ids = torch.tensor([row["supervised_ids"]], device=next(model.parameters()).device)
        labels = ids.clone(); labels[:, :row["prompt_tokens"]] = -100
        with slider.scaled(1.):
            return model(input_ids=ids, labels=labels, use_cache=False).loss
    start = time.monotonic()
    try:
        with (output / "metrics.jsonl").open("a") as stream:
            for step in range(config["steps"]):
                optimizer.zero_grad(set_to_none=True)
                total = 0.
                accumulation = config.get("batch", 4)
                for offset in range(accumulation):
                    row = splits["train"][(step * accumulation + offset) % len(splits["train"])]
                    value = loss(row) / accumulation
                    if not torch.isfinite(value):
                        raise FloatingPointError("Nonfinite supervised loss")
                    value.backward(); total += float(value.detach())
                torch.nn.utils.clip_grad_norm_(slider.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                record = dict(step=step+1, loss=total, elapsed_seconds=time.monotonic()-start)
                stream.write(json.dumps(record)+"\n"); stream.flush()
                if (step+1) % 16 == 0:
                    print(json.dumps(record), flush=True)
            with torch.no_grad():
                development = sum(float(loss(row)) for row in splits["dev"]) / len(splits["dev"])
        save_file({k:v.detach().cpu().contiguous() for k,v in slider.state_dict().items()}, str(output / "lora.safetensors"))
        save_file(linear_deltas(slider), str(output / "deltas.safetensors"))
        result = dict(kind="supervised-lora", steps=config["steps"], dev_nll=development,
                      elapsed_seconds=time.monotonic()-start, targets=slider.names,
                      deltas_sha256=digest(output / "deltas.safetensors"))
        write_json(output / "result.json", result)
        return result
    finally:
        slider.close()
