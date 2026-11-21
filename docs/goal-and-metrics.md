# Goal: a particle slider that beats the original Qwen model

Train and tune a particle slider for Qwen3.5-0.8B using ParticleGAN 0.11.1 through
`particle-sliders-core` 0.13.0. Keep the original model weights frozen. Find a
promising slider, compare it with the slider OFF on real questions, and produce
a verified weight patch only if the particle-derived model demonstrates a clear,
repeatable quality improvement. Supervised LoRA is a historical comparison,
not a fallback for this goal. Do not submit without that win.

## What is already done?

- The shared ParticleGAN upgrade is on `particle-sliders` main at `3dd672a`.
- This project's adoption is on main at `f2f8dd6`, with that exact dependency pin.
- The shared suite passed 584 CPU tests; 27 were skipped. This project's three
  integration tests passed, including exact continuation and adapter export.
- Historical training and exports exist, but the old particle candidate did not
  demonstrate better answers. Those results are not results for ParticleGAN 0.11.1.

No new 0.11.1 native training or quality evaluation has completed. The earlier
research PR is closed. Native GPU resource coordination is pending with Ember;
use a fresh run directory and the new environment for the next admitted run.

## Development selection and training diagnostics

The search ranks the nonlinear RAW teacher by the mean change in
scored finished answers between strength 1 and OFF. Every development
question counts equally. Trials share the same model, questions, prompt,
sampling settings, seed, and token budget. Truncated runs are unmeasured.
Ties are retained without automatically training a final candidate.

The hidden-response RMS and SSE in `result.json` describe how closely the
raw teacher or distilled student matches intermediate training targets.
These metrics are diagnostics. Each trial still records its exact recipe,
input data, implementation and diagnostic OFF baseline for cache validation.
Compare the actual distilled export against OFF separately on matched finished
answers. Older checkpoints and measured outputs keep their source identity.

## How to check a promising slider against OFF

1. Select settings using the development metric. Freeze the candidate file,
   strength, model/source versions and evaluation plan before confirmation.
2. Check that strength `0` reproduces the unchanged base model. OFF must not
   receive a worked solution or any other extra information that ON lacks.
3. Evaluate the actual nonlinear particle teacher OFF and ON on the same
   questions, with the same prompt, seed, sampling, token budget and scorer.
   Evaluation supplies questions, not their worked solutions, to the model.
4. Compare answer accuracy/task scores and per-question wins/losses. Here,
   **higher is better**: `quality_gain = ON_score - OFF_score`. Report uncertainty,
   task regressions and how often answers hit the token limit.
5. If the teacher gives a convincing improvement, check the distilled student
   separately, then the actual loaded weight patch. A good teacher does not
   guarantee a good student. Verify the exported tensors, parent lineage and
   loaded outputs as well as quality.
6. Confirm the frozen choice on an untouched question set or a predeclared set
   of fresh evaluation seeds. Do not keep tuning on the confirmation results.

With the current `delta compare` tool, a positive result needs **both** a positive
score difference and statistical separation: the 95% interval must be wholly
above zero and the paired sign-test p-value below `0.05`. `separable=true` alone
can also describe a model that is worse. A small positive average with an
interval crossing zero is inconclusive. The comparison remains local evidence;
no official result is implied. Predeclare the primary evaluation and regression
checks, and use enough examples to distinguish a useful improvement.

## Example commands for a promising teacher

These commands are for after a fresh run has produced `particle.safetensors`.
They have not been run for the new engine. Activate the new training environment;
on the current workstation it is `/ml2/deltaleague/.venv-engine`.

```sh
qwen35-slider evaluate-math --workspace experiment \
  --teacher runs/particle-search-pg0111-v1/final/particle.safetensors \
  --strength 0 --limit 16 --offset 0 --seed 1000 --max-tokens 8192 \
  --output runs/eval/engine-teacher-off.json
qwen35-slider evaluate-math --workspace experiment \
  --teacher runs/particle-search-pg0111-v1/final/particle.safetensors \
  --strength 1 --limit 16 --offset 0 --seed 1000 --max-tokens 8192 \
  --output runs/eval/engine-teacher-on.json
delta compare runs/eval/engine-teacher-off.json runs/eval/engine-teacher-on.json \
  --output runs/eval/engine-teacher-comparison.json
```

These are development commands. The [predeclared evaluation plan](../configs/evaluation-plan.json)
reserves selection positions `0..15` for development and `16..63` for
confirmation. After freezing the candidate and strength, use `--offset 16
--limit 48 --seed 2000` for the primary confirmation, with seeds `2001` and
`2002` for separately reported replications. Use a fresh output filename for
every run. Changing the slice or seed changes the protocol identity, so mismatched
reports are refused by `delta compare`.

The prompt remains zero-shot chat. Even with an 8,192-token budget, this does not
reproduce the official five-shot protocol or the complete league score. Report
the selected questions and generation settings explicitly. Do not pool repeated
seeds of the same questions as if they were independent new questions.

For the student, repeat the comparison with `--deltas .../deltas.safetensors`
instead of `--teacher .../particle.safetensors`. Both the math and likelihood
commands now accept nonlinear teachers. Use the latter to check broader task
regressions before and after distillation:

```sh
qwen35-slider evaluate-likelihood --workspace experiment \
  --teacher runs/particle-search-pg0111-v1/final/particle.safetensors \
  --strength 0 --tasks arc_challenge arc_easy boolq commonsense_qa hellaswag openbookqa piqa winogrande mmlu \
  --output runs/eval/engine-teacher-likelihood-off.json
qwen35-slider evaluate-likelihood --workspace experiment \
  --teacher runs/particle-search-pg0111-v1/final/particle.safetensors \
  --strength 1 --tasks arc_challenge arc_easy boolq commonsense_qa hellaswag openbookqa piqa winogrande mmlu \
  --output runs/eval/engine-teacher-likelihood-on.json
delta compare runs/eval/engine-teacher-likelihood-off.json runs/eval/engine-teacher-likelihood-on.json
```

These nine tasks are a local regression check. They do not cover every official
league task. They use the manifest's task metrics and selected documents, not
the training hidden-error metric. Once confirmation results have been inspected,
do not describe that same set as untouched in a later tuning round.

## Completion condition

The goal is complete when a particle-derived candidate has a clear, repeatable
quality win over OFF/base under matched conditions, its actual export preserves
that win and passes reconstruction/load checks, and the evidence is recorded.
A repository, passing software tests, smaller hidden error or a loadable patch
alone does not complete the goal. Keep PR #13 closed. Do not submit unless the
quality requirement has been met.
