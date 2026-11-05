# Cached Qwen solutions selected by Jev through NanoGPT

The target is one accepted worked solution for each of the 128 training questions.
The original Qwen3.5-0.8B model generates candidate solutions. Jev evaluates them
through NanoGPT's Decisions API; Jev does not generate the solution text.

Qwen receives the training question and the dataset's verified reference solution.
It is asked to write a concise standalone explanation and finish with `#### number`.
This is reference-assisted generation, not a claim that Qwen solved the question
without help. The source is GSM8K's training split. Development, calibration and
benchmark questions are not used to select these generated training solutions.

For every question:

1. Reuse any saved attempts before generating another answer.
2. Reject an empty or truncated response, a missing final-answer marker, a wrong
   numeric answer, or an explanation that will not fit the training token budget.
3. Ask Jev two independent questions together: whether the reasoning is correct,
   and how good it is as a clear, complete worked solution.
4. Accept a solution only when Jev's correctness estimate is at least `0.9` and
   its quality score is at least `3` on the `0..4` rubric. Higher is better for
   both judgments. These are curation thresholds, not benchmark scores or proof
   that Jev is always correct.
5. Sample again if no saved solution passes. Keep the highest-quality passing
   solution among the saved attempts, breaking ties by correctness and then
   attempt number. Stop generating once a suitable positive exists.

The default budget is 16 attempts per question per pass. If a question still has
no accepted solution, it stays pending. The workflow does not quietly drop it or
accept its last answer. Increasing the attempt budget resumes from cached attempts.
The final training dataset is written only after every question has an accepted
positive. Its development and calibration rows remain exactly as in the input.

## What is cached?

Generation and ranking have separate caches:

- A **generation record** is keyed by the frozen teacher/model identity, rendered
  prompt including its reference solution, sampling settings and deterministic
  seed. Every completed attempt is saved, including answers that are later rejected.
- A **Jev record** is keyed by the NanoGPT endpoint, pinned Jev model, candidate
  text, question/reference and exact judgment instructions and rubric. The raw
  typed response and usage are retained.

Changing a ranking rubric or threshold does not regenerate matching teacher
answers. Changing generation inputs creates a different generation key. Cache
records have content hashes, atomic writes and locks so duplicate workers reuse
the first saved result. Corrupt records are refused rather than silently replaced.
API failures never become accepted answers. Credentials are read at runtime and
are never written to cache records, reports or dataset files.

Every selected positive points back to its generation and Jev cache keys. Merging
workers rechecks the saved responses, correctness thresholds and original question
and answer identities. The output records the source-data hash, curator source
hash, teacher identity, generation settings and ranking policy.

## Seeds and paired hidden states

The generation record saves the sampling seed with the exact Qwen prompt,
settings and answer. A seed alone is insufficient to identify a training pair.
Particle training also caches the exact positive and neutral token IDs and both
frozen-model hidden-state tensors **in one record**. It retains the rendered
prompt text, shared continuation prefix, model/runtime identity and source-row
identity. Curated training rows link back to the accepted generation and Jev
records, so the question, answer, seed and resulting pair remain traceable.

These are the final normalized text hidden states used by this training objective,
one 1,024-value vector per side for Qwen3.5-0.8B. The stored float32 values preserve
the original bits. Reuse requires the same model, extractor, input tokens and
provenance. A changed positive, neutral or model gets a different cache entry;
corrupt or nonfinite tensors are refused.

Set `pair_cache` in the particle training configuration to share a persistent
cache across trials. The default is `frozen-pairs` beside the input data file.
Each run saves `prepared-pairs.json` linking its rows to the cached pairs, and
binds that file's hash and the actual training/development tensor hashes into its
training contract. `pair-cache-usage.json` reports hits and newly computed pairs.
Checkpoint continuation rechecks these identities, and search refuses a changed
paired-state manifest.

## Commands

Install the package and set `NANOGPT_API_KEY` in the process environment. Adjust
the paths in [the configuration](../configs/teacher-data.json). An existing private
env file can alternatively be supplied as `key_env_file`; it must contain a
`NANOGPT_API_KEY` assignment. Never commit that file or a key.

```sh
qwen35-slider curate-teachers configs/teacher-data.json
qwen35-slider merge-teachers configs/teacher-data.json
```

For parallel generation, run one process per worker, choosing an available GPU
for each process through `CUDA_VISIBLE_DEVICES`. With four workers, run the
curation command with `--shards 4 --shard 0`, then the same command with shards
`1`, `2` and `3`. After all four finish successfully:

```sh
qwen35-slider merge-teachers configs/teacher-data.json --shards 4
```

Use the merged file as `data` in a fresh particle training configuration and a
fresh output directory. This remains particle/GAN training with the base frozen;
it does not switch to supervised LoRA.

## Positive and neutral prompts

Both contain the identical math question. During particle training the positive
also receives the accepted worked solution as system context:

```text
Neutral:
  User: {question}

Positive:
  System: Use this verified worked solution to answer accurately:
  {accepted Qwen worked solution}
  Answer: {known correct answer}
  User: {same question}
```

Both are rendered with Qwen's chat template with thinking enabled. Both also
receive the same causal solution prefix, from zero to 32 tokens. The target is
the frozen model's internal state on the positive input; the slider is trained
on the neutral input. The actual ON/OFF answer tests give both models only the
question, with no reference solution or answer.

## Native pilot

The initial four-question pilot accepted answers after `1`, `4`, `6` and `1`
attempts. A fresh-output native replay reused all 12 generated attempts and all
nine Jev responses without changing or adding any cache record. This verifies
reuse for that pilot. It is not evidence that a trained slider beats OFF.

The provider request contract is documented by
[NanoGPT Decisions](https://docs.nano-gpt.com/api-reference/endpoint/decisions).
The grading uses TypeSafe's [Score](https://docs.typesafe.ai/primitives/score) and
Noul primitives, served here exclusively through NanoGPT.


### Retrying difficult questions without replacing accepted examples

An optional `retry_generation` object provides different generation settings for
attempt indices 64–127 (`retry_after` defaults to 64). The retry implementation
first checks every cached attempt, including the original recipe. If any cached
answer passes the unchanged Jev and code checks, it reuses the best one. Otherwise
it prioritizes new retry-setting attempts, keeping their actual settings and seed
in the generation cache key. The maximum attempt index is bounded at 127.

Use a fresh output name after changing the curator implementation or retry recipe;
existing generation and Jev records are still reused from the shared cache. The
current retry pass uses presence penalty 0 and temperature 0.6. It changes sampling,
not acceptance: correct final number, correctness >=0.90, quality >=3/4, no truncation,
and the training token budget are still required. A rejected candidate may be
mathematically correct yet fail the separate explanation-quality threshold. Jev is
a fallible filter; its outputs are not proof of correctness or model improvement.
