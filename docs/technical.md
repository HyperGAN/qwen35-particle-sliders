# Training with ParticleGAN 0.9.5

Run these commands from the repository root. The command-line API is unchanged;
new particle runs use the shared `particle-sliders-core` 0.13 engine and published
`particlegan==0.9.5`. They use a fresh output directory and cannot resume the
historical 0.6.0 experiment. No new benchmark result is claimed by this update.

## Install and prepare

Use a separate Python 3.12+ environment with a CUDA-capable PyTorch installation
for native model training. Tests use CPU fixtures.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test,eval]'
. .venv/bin/activate
delta competition init qwen35-08b-v1 --output experiment
curl -fL https://raw.githubusercontent.com/openai/grade-school-math/b0bb162abedc65e1fdd8e93ed090fd7598ee68bc/grade_school_math/data/train.jsonl -o gsm8k-train.jsonl
qwen35-slider prepare-gsm gsm8k-train.jsonl data/gsm-v1.json
```

Edit workspace, data and output paths in `configs/search.json` for your machine.
The model's original weights and official validation manifest stay unchanged.
Use the GPU assigned to your workload when setting `CUDA_VISIBLE_DEVICES`.

## Train particle sliders

```sh
qwen35-slider search configs/search.json
```

This runs four 32-update settings, then a fresh 256-update final using the selected
setting. The generator, critic and particle-bank learning-rate settings remain
configurable through the same JSON keys. Selection uses raw hidden-state error
on separate development questions, not benchmark answers. Lower error is only a
training diagnostic; the candidate must separately beat the base on matched
answer-quality evaluation before submission.

The shared `ConditionalPG09` engine supplies ParticleGAN's public
`bispf_optimized` recipe: its adversarial loss, critic penalty, particle-prior
regularizer and optimizers. This replaces the historical RpGAN/b_cap/VIC game.
The routed adapter architecture still shares one 128×4 bank across rank-8
branches in 186 Qwen text attention/MLP projections. Base weights stay frozen
and load at unquantized BF16 precision; trainable adapters remain FP32.

Qwen supplies paired hidden responses, with and without worked-solution context.
The critic receives the paired response error plus noise, conditioned on the
normalized frozen neutral response. Error is divided by the training set's
scalar RMS edit size. The model's full hidden width is retained and explicitly
declared `continuous`; ParticleGAN 0.9.5 supports this without reshaping it or
interpreting it as categorical. The critic emits raw logits.

Each run writes its resolved upstream recipe, exact source identities, data/model
identity, metrics, particle teacher, distilled adapter, projection weight deltas
and continuation checkpoint. Continuation includes the shared game, optimizers,
regularizers, random streams and moving-average adapter state. Old-version or
mismatched-source checkpoints are refused. `particle_grad_norm` includes the
public prior regularizer; it is not labelled an adversarial-only gradient.

## Export and verify

The particle teacher's nonlinear routing cannot merge exactly into model weights.
A linear student is fitted on separate calibration inputs using the shared ridge
fitter. Export merges that student, and checks the exported patch against the
student; it does not prove that the student equals the teacher.

```sh
qwen35-slider build --workspace experiment \
  --deltas runs/particle-search-pg095-v2/final/deltas.safetensors \
  --strength 1 --output runs/particle-pg095-patch
delta validate runs/particle-pg095-patch
qwen35-slider verify-candidate --workspace experiment \
  --deltas runs/particle-search-pg095-v2/final/deltas.safetensors \
  --patch runs/particle-pg095-patch
```

## Compare against the original model

```sh
qwen35-slider evaluate-likelihood --workspace experiment --output runs/eval/base.json
qwen35-slider evaluate-likelihood --workspace experiment \
  --deltas runs/particle-search-pg095-v2/final/deltas.safetensors --output runs/eval/particle.json
delta compare runs/eval/base.json runs/eval/particle.json --output runs/eval/comparison.json
```

The likelihood command uses the pinned public task selection and labels its
limited benchmark coverage. `evaluate-math` also exists; by default it uses only
16 questions, zero-shot chat and a 1,024-token cap. That is a development screen,
not the full suite's five-shot/8,192-token protocol. Report truncation and compare
both models under identical settings. `--teacher` evaluates a saved nonlinear
particle teacher separately from the student in both math and likelihood checks.
Math also accepts `--offset` within the pinned public selection and `--seed`;
these controls are included in the comparison protocol identity. See the
[goal/metric guide](goal-and-metrics.md) and the frozen evaluation-plan JSON for
development versus confirmation use.

No training/export command logs in, uploads weights or submits automatically.
Do not submit without a demonstrated win over the base. The earlier research
PR is closed.

## Historical comparison and tests

`qwen35-slider train-lora configs/lora.json` remains available to reproduce the
separate supervised comparison. It is not part of particle training or a fallback
submission. [Historical details](historical-pg82.md) describe v0.1.0 only.

```sh
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

The CPU tests perform actual shared-engine updates, frozen-base and zero-strength
checks, exact checkpoint continuation, ridge export and merged-model equivalence.
These test mechanics rather than pretrained-model quality.
