# Goal: a particle slider that beats the original Qwen model

Train and tune a particle slider for Qwen3.5-0.8B using ParticleGAN 0.9.5 through
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
  demonstrate better answers. Those results are not results for ParticleGAN 0.9.5.

No new 0.9.5 native training or quality evaluation has completed. The earlier
research PR is closed. Native GPU resource coordination is pending with Ember;
use a fresh run directory and the new environment for the next admitted run.

## How to read the training metric

The current search prints **`student_relative_rms`**. **Lower is better.**
It compares the distilled adapter's internal-state error with the original
model's error, on the same development questions and target states:

```text
student_relative_rms = student ON_RMS / student OFF_RMS
```

| Value | Meaning |
| --- | --- |
| Less than `1` | ON matches the internal training target better than OFF. |
| Exactly `1` | No change in that error. |
| Greater than `1` | ON matches the target worse than OFF. |

For example, OFF RMS `2.0` and ON RMS `1.6` give a ratio of `0.8`: 20% less RMS
error. This is an illustrative example, not a measured result.

`on_rms` and `off_rms` in each run's `result.json` are also lower-is-better.
`raw` is the trained particle teacher, `ema` is its moving-average version, and
`student` is the linear approximation fitted for weight export. `selected` says
which teacher state was saved; EMA is selected only if its development error is
lower. Compare every version with its corresponding OFF baseline.

RMS means root mean squared error. On identical rows and dimensions, raw sum of
squared errors (SSE) gives the same ranking:

```text
SSE = number_of_compared_values * RMS^2
ON_SSE / OFF_SSE = (ON_RMS / OFF_RMS)^2
```

So a ratio of `0.8` in RMS means a ratio of `0.64` in SSE. Compare trials only
when their base model, development examples, target construction, dimensions
and raw metric definition match. Their OFF baseline should agree. Do not compare
absolute errors from the old normalized metric with the current raw metric.
If OFF error is zero, the ratio is undefined and must not be used for ranking.

**A lower hidden-state error is a screening signal, not a benchmark win.**
Generator and critic losses (`g_adv`, `d_loss`) are training diagnostics; their
scales change with the game and they are not scores to minimize across models.

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
on the current workstation it is `/ml2/deltaleague/.venv-pg095`.

```sh
qwen35-slider evaluate-math --workspace experiment \
  --teacher runs/particle-search-pg095-v2/final/particle.safetensors \
  --strength 0 --limit 64 --max-tokens 8192 \
  --output runs/eval/pg095-teacher-off.json
qwen35-slider evaluate-math --workspace experiment \
  --teacher runs/particle-search-pg095-v2/final/particle.safetensors \
  --strength 1 --limit 64 --max-tokens 8192 \
  --output runs/eval/pg095-teacher-on.json
delta compare runs/eval/pg095-teacher-off.json runs/eval/pg095-teacher-on.json \
  --output runs/eval/pg095-teacher-comparison.json
```

The command fixes the sampling seed at `1000` for both runs. It uses the same
public question selection and zero-shot chat prompt. Even with an 8,192-token
budget, it does not reproduce the official five-shot protocol. Treat this as a
development comparison, not final confirmation or the complete league score.
The earlier experiment already inspected some of these public questions.

For the student, repeat the comparison with `--deltas .../deltas.safetensors`
instead of `--teacher .../particle.safetensors`. The current likelihood evaluator
accepts student weight deltas; it does not yet accept nonlinear teachers.
Broader teacher coverage and fresh-seed/untouched-selection controls are remaining
work if needed for the predeclared confirmation plan.

## Completion condition

The goal is complete when a particle-derived candidate has a clear, repeatable
quality win over OFF/base under matched conditions, its actual export preserves
that win and passes reconstruction/load checks, and the evidence is recorded.
A repository, passing software tests, smaller hidden error or a loadable patch
alone does not complete the goal. Keep PR #13 closed. Do not submit unless the
quality requirement has been met.
