# Qwen3.5 Particle Sliders

Train routed particle teachers on the registered Delta League Qwen3.5-0.8B
checkpoint, distill them into ordinary linear adapters, and export exact weight
replacement patches. A separate supervised LoRA provides a practical comparator.

This is a native CUDA experiment in progress. The two-update Qwen smoke run and
exact patch reconstruction have passed; search and final quality evaluation are
pending. No official submission or score is claimed.

The first supervised comparator has completed 128 updates. On the fixed 512-row,
eight-benchmark local likelihood subset, its mean is 56.05% versus 52.93% for the
parent (+3.13 points; paired bootstrap interval about +0.6 to +5.7, sign-test
p=0.023). These are development results; generated-answer comparisons and the
particle final run are pending. The complete aggregate receipt is in
[evidence/lora-likelihood-v1.json](evidence/lora-likelihood-v1.json).

The particle objective is the actual historical formulation from
[mikkel/meta-glimmer-particle-slider](https://github.com/mikkel/meta-glimmer-particle-slider):
one learned 128×4 bank, per-projection rank-8 down/up maps, deterministic
all-particle softmax routing, the gmix critic, paired-edit normalization,
relativistic GAN losses, upstream lazy b_cap penalty, and particle VIC. The
source-checked upstream update executes unchanged. There is no local replacement
GAN implementation or supervised term in particle training.

For a neutral causal history, the frozen positive context contains a verified
worked solution from **GSM8K training data**. Both contexts receive identical
continuation token IDs. The fake critic input is noise plus the normalized
adapted-neutral minus frozen-positive final hidden state; real input is the same
noise. Development and ridge calibration use disjoint rows from that training
split. The public benchmark selection is never used as gradient data.

The host discovers real linear-attention, full-attention and MLP projection
dimensions across all 24 text layers. It excludes vision, embeddings and the
language output head. `scope=attention` additionally excludes MLPs. The frozen
pretrained checkpoint is loaded at its full, unquantized BF16 precision; particle
parameters and critic calculations use FP32. This trains adapters through the
full text model; it does not describe a full-parameter fine-tune.

Nonlinear routed adapters cannot be merged exactly. Export captures actual host
activations on the separate calibration split and calls the reference's unchanged
dual ridge fit. It retains the teacher up projection and fits a linear down
projection. The resulting student is approximate; its held-out behavior must be
evaluated separately. A strength multiplies the resulting linear weight delta.

## Install and initialize

Python 3.12+, Git and a CUDA-capable PyTorch installation are required for the
recorded native experiment. CPU fixtures also run without a GPU.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test]'
. .venv/bin/activate
delta competition init qwen35-08b-v1 --output experiment
curl -fL https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl -o gsm8k-train.jsonl
qwen35-slider prepare-gsm gsm8k-train.jsonl data/gsm-v1.json
```

The data builder records the downloaded source hash and deterministically divides
192 rows into 128 training, 32 development and 32 calibration rows. Preserve the
original `experiment/model/`, `delta-model.json`, evaluation and public manifest.
The upstream loader obtains its own checkout if none is supplied and verifies the
exact commit plus algorithm-file hashes before loading it.

This project deliberately uses the reference's isolated **historical PG82**
dependency, whose package version is 0.6.0. It does not claim to be the modern
PG0.9.4 Glimmer backend: that backend's current source gates refuse native
1024-wide hidden responses. Neither that gate nor the fixture-only admission is
bypassed here. The historical Glimmer package and its exact dependency remain in
their own environment. See [NOTICE.md](NOTICE.md) for pins and attribution.

## Train and export

Edit the absolute workspace/data/output/upstream paths in `configs/` for your
machine. The recorded search is four fixed 32-update trials, followed by one
256-update final run at the selected rates, all with seed 7. It searches generator
learning rate, discriminator ratio and bank learning-rate ratio. Selection uses
held-out distilled-student hidden RMS relative to the same off baseline. This
diagnostic is not a language capability score. Final quality selection needs the
public advisory comparison.

```sh
CUDA_VISIBLE_DEVICES=1 qwen35-slider search configs/search.json
CUDA_VISIBLE_DEVICES=0 qwen35-slider train-lora configs/lora.json
qwen35-slider build --workspace experiment \
  --deltas runs/particle-search-v1/final/deltas.safetensors \
  --strength 1 --output runs/submission
delta validate runs/submission
```

The supervised comparator uses the same source/splits and selected projections,
ordinary rank-8 LoRA, answer-token cross entropy and a 128-update horizon. Its
loss is separate from the particle formulation.

Artifacts stay outside Git. Each particle run saves the teacher, distilled
student, merged projection deltas, contract, metrics, calibration diagnostics
and a continuation checkpoint with optimizers, critic, EMA and random streams.
Resume requires the original recipe, base, data and runtime identity. Export
builds an official `dense-diff`/`replace` manifest, reconstructs it using the
official CLI implementation, and checks every reconstructed tensor against the
expected merged checkpoint. Unchanged tensors and registered dtypes are checked.

`--norm-gain` optionally tests a separately recorded final-logit gain. Qwen's
RMSNorm parameter is an offset, so the implementation scales `(1 + weight)` and
then subtracts one. Default gain is 1; this is not silently added to a particle
result.

## Evaluation and submission

The current evaluation is `public-ladder500-8k-v2` and the immutable original
parent is `0ea62f40-7cd8-4d98-9ebb-484192fa05eb`. The live public download is a
suite manifest (task selection indices, revisions and hashes), whereas the
0.1.1 `delta evaluate` implementation expects benchmark-to-example lists. Preserve
the official file; do not fabricate a compatible file under its official hash.
Local comparisons must be labelled advisory and state their benchmark coverage
and generation overrides. Official server validation and private-test results
remain the authority.

The optional `eval` extra enables a native-likelihood comparison over eight
public benchmarks. It verifies dataset revisions, populations, selected indices
and every selected document hash against the downloaded official manifest, then
executes the pinned lm-eval tasks. It uses `acc_norm` where available and `acc`
otherwise, without a chat template, and labels the result as a subset mean.
It does not cover math generation, IFEval, coding or BFCL.

```sh
uv pip install --python .venv/bin/python -e '.[test,eval]'
qwen35-slider evaluate-likelihood --workspace experiment --output runs/eval/base.json
qwen35-slider evaluate-likelihood --workspace experiment \
  --deltas runs/particle-search-v1/final/deltas.safetensors --output runs/eval/particle.json
delta compare runs/eval/base.json runs/eval/particle.json --output runs/eval/comparison.json
qwen35-slider evaluate-math --workspace experiment --output runs/eval/base-math.json
qwen35-slider evaluate-math --workspace experiment \
  --deltas runs/particle-search-v1/final/deltas.safetensors --output runs/eval/particle-math.json
```

The per-example comparison records contain metric values rather than generated
text. Their protocol hash is separate from the unchanged official manifest hash.
The math command separately verifies the selected public GSM8K document hashes
and generates up to 1,024 tokens on 16 examples by default, with the registered
thinking/sampling settings and seed 1000. It reports truncation explicitly.
`--teacher` can load the nonlinear particle artifact for a teacher/student check.
Raw math generations remain local and are excluded from Git.

The intended final deliverable is a validated submission directory plus its
research record and reproducible source. Before an actual submission, open a
research-only PR under Delta League's contribution rules and attach its URL.
Authentication and an evaluation quota are needed only for actual submission:

```sh
delta login
delta quota
delta submit runs/submission --title 'Qwen3.5 particle-distilled reasoning slider' --pr ACTUAL_RESEARCH_PR_URL
```

The training/export commands never log in, upload weights or consume official
evaluation quota automatically.

## Tests

```sh
QWEN35_PG82_ROOT=/path/to/pinned/particle-sliders python -m pytest -q
```

The fixtures exercise hybrid target discovery, a genuine upstream adversarial
update, frozen base tensors, zero-strength bypass after training, ridge fitting,
and linear forward-versus-merged equivalence. Native training/quality evidence
is recorded separately from fixture conformance.
