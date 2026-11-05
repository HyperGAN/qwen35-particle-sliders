# Particle sliders for Qwen3.5

This project trains a **particle slider**: a learned adjustment to Qwen3.5-0.8B
whose strength can be turned up or down. The aim is to improve the model's
answers to math questions while keeping the original model weights fixed.

**The goal is a particle slider that beats the original model. We have not
achieved that yet. Nothing has been submitted to Delta League, and the research
PR is closed.** The last training run has finished; no training job is currently
running from this project session.

Current training code uses **ParticleGAN 0.9.5 through the shared
[particle-sliders core](https://github.com/HyperGAN/particle-sliders)**. The
results below belong to the earlier v0.1.0 experiment; the dependency update does
not establish a new model result.

See the [current goal and metric guide](docs/goal-and-metrics.md) for the remaining
work, which numbers are lower-is-better, and how to compare a promising slider
ON versus OFF on answer quality.

## What is a slider?

A slider adds small trainable modules, called **adapters**, inside an existing
model. Here, the adapters adjust calculations in the model's text layers.
We train those additions while leaving the original model weights unchanged.

The slider's strength controls how much of the learned adjustment is applied:

| Strength | What it does |
| --- | --- |
| `0` | Uses the original model without the adjustment. |
| `0.5` | Applies half of the learned adjustment. |
| `1` | Applies the full learned adjustment. |

The number controls the size of the adjustment. It does not measure reasoning
ability, and turning it up does not guarantee better answers. The recorded
training used strength `1`; it did not establish useful behavior at every setting.
This repository provides Python code and command-line tools, not a visual slider UI.

## What makes this a particle slider?

The adapters share a bank of 128 learned vectors called **particles**. Each
adapter uses the model's intermediate values to choose how to mix information
from that bank. This lets its adjustment depend on the input, rather than
always applying the same linear transformation.

The shared [particle-sliders](https://github.com/HyperGAN/particle-sliders) engine
uses ParticleGAN's training recipe. The routed adapter architecture was adapted
from [the Glimmer particle-slider project](https://github.com/mikkel/meta-glimmer-particle-slider).
A GAN is a training setup in which one network learns an adjustment and another,
called a **critic**, learns to detect the remaining error.

For this experiment:

1. Take a math question and its worked solution from the GSM8K training dataset.
2. Run the original model with the worked solution supplied as extra context.
   Save its internal representation (the numbers it computes inside the model)
   as the training target.
3. Run the model with the particle slider using the question without that extra
   solution context. Both runs also receive the same short solution prefix,
   which can be empty.
4. Train the particle slider to reduce the difference between those internal
   representations, using the ParticleGAN critic.

The hypothesis is that learning this adjustment will help the model answer
questions when the worked solution is absent. **Matching internal representations
better does not itself prove that answers improve.** In the completed experiment,
that matching improved, but the answer-quality checks did not show a win.
The particle training uses worked solutions as guidance; it does not directly
train next-token answer prediction.

## Why is there also a supervised LoRA experiment?

We ran a separate comparison using **LoRA** (low-rank adaptation), a simpler
adapter made from two small matrices. “Supervised” means that we supply a question and its known
worked answer, then train the adapter to predict the answer's next tokens.
The training error is computed on the worked answer, not the question.

That comparison uses the same source data and model layers, but it has no
particle bank or GAN critic. It is a separate experiment, not a stage in particle
training. The half-strength and full-strength LoRA results come from the same
trained adapter with different strength settings.

**The project remains focused on particle sliders. The supervised comparison
is not a replacement for that goal or a fallback submission.** Its historical
results are retained in the [experiment record](evidence/final-experiment.json).

## What did we train, and did it work?

In the earlier v0.1.0 experiment, we tried four particle-training settings for 32 updates each, then trained a
fresh particle slider for 256 updates using the selected setting. The settings
were selected by internal-representation matching, not by demonstrated answer
quality. The data came from a small GSM8K training subset: 125 retained training
questions, 32 separate development questions, and 31 separate calibration
questions used for export.

For Delta League's weight-patch format, we also fitted a simpler linear adapter
to approximate the particle slider. This conversion is called **distillation**:
the original particle slider is the *teacher*, and the approximation is the
*student*. The student can be merged into model weights; the input-dependent
particle teacher cannot be merged exactly. The scores below are for the student,
so they do not establish the teacher's full performance.

| Model | Average accuracy across nine local benchmark tasks | Correct math answers in a small local check |
| --- | ---: | ---: |
| Original Qwen3.5-0.8B | 52.31% | 4 of 16 |
| Adapter distilled from the particle slider | 50.38% | 1 of 16 |

The nine-task check contains 626 examples. The math check allowed 1,024 generated
tokens per answer; every original-model and particle-student answer reached that
limit. It also used a different prompting setup from the full Delta League
protocol. A separate four-question check of the particle teacher was too small
and similarly truncated to establish its quality.

These results do not demonstrate an improvement over the original model.
The exported files load correctly, but that does not establish model quality.
**We will not submit unless a particle candidate demonstrates a clear win over
the original model under matched evaluation conditions.**

## Code, artifacts, and reproduction

- [Technical details and reproduction commands](docs/technical.md): installation,
  particle training, the separate supervised comparison, export, and evaluation.
- [Complete experiment record](evidence/final-experiment.json): scores, settings,
  source versions, and artifact hashes for all experiments.
- [Experimental model files](https://github.com/HyperGAN/qwen35-particle-sliders/releases/tag/v0.1.0):
  saved particle teacher, distilled adapter, and supervised comparison adapter.
  These are research artifacts; they are not recommended submissions.
- [Closed research PR](https://github.com/ilblackdragon/deltaleague/pull/13): the
  earlier research record and notice explaining why nothing was submitted.

The training code is in [training.py](src/qwen35_sliders/training.py), the slider
modules are in [host.py](src/qwen35_sliders/host.py), and the particle search
settings are in [configs/search.json](configs/search.json). Source attribution
and dependency versions are in [NOTICE.md](NOTICE.md).
