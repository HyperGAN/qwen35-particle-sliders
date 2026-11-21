"""Scope the shared paper methods to native Qwen3.5 answer predictions."""
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

import torch
from safetensors.torch import load

from .steering_core import CORE_REVISION, load_steering_core


class SteeringPlan:
    """An explicit fitted or calibrated steering plan, loaded before inference."""

    def __init__(self, path, wheel, strength):
        path = Path(path)
        content = path.read_bytes()
        plan = json.loads(content)
        if not isinstance(plan, dict):
            raise ValueError("Qwen3.5 steering plan must be a JSON object")
        self.layer = plan.get("layer")
        self.method = plan.get("method")
        self.intervention = plan.get("intervention")
        evidence = plan.get("evidence_ref")
        if (type(strength) not in (int, float) or not isfinite(strength)
                or type(self.layer) is not int or self.layer < 0
                or self.intervention not in ("residual", "raw_adapter")
                or not isinstance(evidence, str) or not evidence.strip()):
            raise ValueError("Supply finite strength, a decoder layer, intervention and calibration evidence")
        self.core = load_steering_core(wheel)
        self.strength = float(strength)
        self.identity = dict(core_revision=CORE_REVISION, plan_sha256=sha256(content).hexdigest(),
                             method=self.method, intervention=self.intervention, decoder_input_layer=self.layer,
                             evidence_ref=evidence, nominal_strength=self.strength)
        self.probe = self.direction = None
        required = {"method", "intervention", "layer", "evidence_ref"}
        if self.method == "s_psr":
            if (self.intervention != "residual"
                    or set(plan) != required | {"weights", "weights_sha256", "weights_kind", "ema_decay"}):
                raise ValueError("S-PSR requires the residual intervention and a fitted safetensors file and digest")
            if (plan["weights_kind"] != "raw" or plan["ema_decay"] is not None
                    or not isinstance(plan["weights"], str) or not plan["weights"].strip()):
                raise ValueError("S-PSR requires fitted RAW weights with no EMA")
            payload = (path.parent / plan["weights"]).read_bytes()
            fingerprint = sha256(payload).hexdigest()
            if fingerprint != plan["weights_sha256"]:
                raise ValueError("S-PSR fitted weights do not match the plan's digest")
            self.probe = load(payload)
            if set(self.probe) != {"probe.weight", "probe.bias", "direction.weight"}:
                raise ValueError("Use the fitted shared PromptSteering state dictionary")
            dtype = self.probe["probe.weight"].dtype
            if any(x.dtype != dtype or not x.is_floating_point() or not torch.isfinite(x).all()
                   for x in self.probe.values()):
                raise ValueError("S-PSR weights must have one finite floating-point dtype")
            self.identity.update(weights_sha256=fingerprint, weights_kind="raw", ema_decay=None)
        else:
            self._prepare_gate(plan, required)

    def _prepare_gate(self, plan, required):
        methods = ("cfp", "clabs", "random")
        if self.method not in methods or type(plan.get("parameter")) not in (int, float):
            raise ValueError("Provide a CFP threshold, CLABS bias, or matched random firing rate")
        allowed = required | {"parameter", "direction", "seed", "clabs", "preserve_norm", "projection_bounds"}
        if set(plan) - allowed:
            raise ValueError("Unexpected Qwen3.5 steering options")
        self.parameter = plan["parameter"]
        self.seed = plan.get("seed")
        self.clabs = self.core.adaptive.CLABSParameters(**plan["clabs"]) if "clabs" in plan else None
        self.preserve_norm = plan.get("preserve_norm", False)
        self.bounds = plan.get("projection_bounds")
        if type(self.preserve_norm) is not bool:
            raise ValueError("preserve_norm must be Boolean")
        if self.bounds is not None:
            low, high = self.bounds
            if not isfinite(low) or not isfinite(high) or low > high:
                raise ValueError("Projection bounds must be finite and ordered")
        if self.intervention == "raw_adapter" and (self.layer != 0 or self.preserve_norm or self.bounds is not None):
            raise ValueError("Qwen3.5 RAW gating reads layer 0 before all adapter targets")
        if "direction" in plan:
            direction = torch.as_tensor(plan["direction"], dtype=torch.float32)
            norm = direction.norm()
            if direction.ndim != 1 or not isfinite(norm) or norm <= 0:
                raise ValueError("Provide a finite nonzero calibrated direction")
            self.direction = direction / norm
        elif self.method != "random" or self.intervention == "residual":
            raise ValueError("Provide the calibrated direction at the selected decoder input")
        self.new_gate()
        self.identity.update(parameter=self.parameter, seed=self.seed,
                             clabs=asdict(self.clabs) if self.clabs is not None else None,
                             preserve_norm=self.preserve_norm, projection_bounds=self.bounds)

    def new_gate(self):
        return self.core.adaptive.ActivationGate(self.method, self.parameter, seed=self.seed, clabs=self.clabs)


class CachedSteering:
    """Bind the plan to one native model. Each request starts with a new gate."""

    def __init__(self, model, slider, plan, max_tokens):
        text = model.model.language_model
        if text.config.model_type != "qwen3_5_text" or plan.layer >= len(text.layers):
            raise ValueError("Select an existing Qwen3.5 decoder-input layer")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("Adaptive steering requires a positive native token budget")
        self.model, self.slider, self.plan, self.max_tokens = model, slider, plan, max_tokens
        self.layer = text.layers[plan.layer]
        self.width = text.config.hidden_size
        self.fitted = None
        if plan.probe is not None:
            if (any(plan.probe[key].shape != (1, self.width) for key in ("probe.weight", "direction.weight"))
                    or plan.probe["probe.bias"].shape != (1,)):
                raise ValueError("Fitted S-PSR state must match the Qwen3.5 decoder width")
            self.fitted = plan.core.prompt.PromptSteering(self.width, device="meta", dtype=plan.probe["probe.weight"].dtype)
            self.fitted.load_state_dict(plan.probe, assign=True)
            self.fitted.eval().requires_grad_(False)
        elif plan.direction is not None and plan.direction.shape != (self.width,):
            raise ValueError("Calibrated direction must match the Qwen3.5 decoder width")
        if plan.intervention == "raw_adapter" and slider is None:
            raise ValueError("RAW gating requires a nonlinear RAW teacher")
        if plan.intervention == "residual" and slider is not None:
            raise ValueError("Direct residual steering requires the frozen base without a teacher")

    @contextmanager
    def request(self):
        plan = self.plan
        gate = plan.new_gate() if self.fitted is None else None
        trace, applied = [], []
        receipt = dict(plan.identity, trace=trace, effective_strengths=applied)
        scale = ExitStack()
        prefix = cache = position = None

        def before_forward(_model, _args, request):
            nonlocal prefix, cache, position
            if position is not None:
                raise RuntimeError("Qwen3.5 skipped the selected decoder layer")
            ids = request.get("input_ids")
            mask = request.get("attention_mask")
            past = request.get("past_key_values")
            index = len(applied)
            if ids is None or ids.ndim != 2 or not callable(getattr(past, "get_seq_length", None)):
                raise RuntimeError("Qwen3.5 adaptive steering requires native cached token IDs")
            if index == 0:
                prefix, cache = ids.shape[1], past
            previous = prefix + index - 1 if index else 0
            masked = mask is not None and (not isinstance(mask, torch.Tensor)
                or mask.shape != (1, prefix + index) or not prefix or mask[0, -1].item() != 1)
            if (prefix == 0 or index >= self.max_tokens or ids.shape != (1, 1 if index else prefix)
                    or masked or past is not cache or past.get_seq_length() != previous):
                raise RuntimeError("Qwen3.5 adaptive steering saw an unexpected cached position")
            position = index

        def before_layer(_layer, args):
            nonlocal position
            if position is None:
                raise RuntimeError("Qwen3.5 visited the selected decoder layer twice")
            index, position = position, None
            if plan.strength == 0 or (index == 0 and plan.intervention == "raw_adapter"):
                applied.append(0.)
                return None
            hidden = args[0]
            if hidden.shape != (1, 1 if index else prefix, self.width):
                raise RuntimeError("Qwen3.5 decoder residual does not match the native cached position")
            row = hidden[:, -1:]
            record = dict(output_index=index, cache_position=prefix + index - 1)
            edited = row
            if gate is None:
                self.fitted.to(device=row.device)
                edited, coefficient = self.fitted(row, plan.strength)
                effective = (coefficient * plan.strength).item()
                record["coefficient"] = coefficient.item()
            else:
                score = None
                if plan.method != "random":
                    score = plan.core.adaptive.activation_projection(row, plan.direction, strength=plan.strength).item()
                decision = gate.step(score)
                effective = plan.strength if decision.fired else 0.
                record.update(alignment=score, probability=decision.probability, fired=decision.fired)
                if decision.fired and plan.intervention == "raw_adapter":
                    scale.enter_context(self.slider.scaled(plan.strength))
                elif decision.fired:
                    edited = plan.core.adaptive.steer_residual(row, plan.direction, strength=plan.strength,
                        gate=True, preserve_norm=plan.preserve_norm, projection_bounds=plan.bounds)
            if not isfinite(effective) or (edited is not row and not torch.isfinite(edited).all()):
                raise FloatingPointError("Qwen3.5 steering produced a nonfinite edit")
            record["effective_strength"] = effective
            trace.append(record)
            applied.append(effective)
            if edited is not row:
                return (torch.cat((hidden[:, :-1], edited), dim=1) if index == 0 else edited, *args[1:])
            return None

        def after_forward(_model, _args, output):
            scale.close()
            if output is not None and position is not None:
                raise RuntimeError("Qwen3.5 finished a token without reaching the selected decoder layer")

        with ExitStack() as hooks:
            if self.slider is not None:
                hooks.enter_context(self.slider.scaled(0.))
            hooks.callback(self.model.register_forward_pre_hook(before_forward, with_kwargs=True).remove)
            hooks.callback(self.layer.register_forward_pre_hook(before_layer).remove)
            hooks.callback(self.model.register_forward_hook(after_forward, always_call=True).remove)
            hooks.callback(scale.close)
            yield receipt
        receipt["prefix_tokens"] = prefix
        if gate is not None:
            receipt["usage"] = asdict(gate.usage)
