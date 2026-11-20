# Memory requirements

| Operation/profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
|---|---:|---:|---:|---|
| Particle hidden-response training (RAW planning policy) | 1 | 12 | 12 | scopeall/rank8; accumulated batch 4,≤ 384 tokens; BF16 full model, FP32 adapters; no checkpointing; no KV |
| Separate ordinary LoRA supervised comparator | 1 | 12 | 12 | same384 token profile; real AdamW FP32 parameters+grad+two moments; full logits upper allowance |
| Bounded finished-text sampling | 1 | 8 | 12 | batch 1; ≤ 1024 prompt+1024 new=2048 total; hybrid6KV+18recurrent; BF16 base; external Runner |

These are **calculated planning budgets, not measured peaks or hard runtime ceilings**. GiB means 2³⁰ bytes. The table bounds one operation at a time; do not add sequential phases, and do add concurrently resident models/jobs. Longer contexts, larger batches, reference audio or different backends require recalculation; the package does not have a universal finite maximum for every accepted request. No benchmark, model construction, forward, GPU query or training run was performed for this document.

Frozen BF16/FP16 weights cost `2P`; FP32 costs `4P`. LoRA costs `r × Σ(in + out)`. The root particle bank is counted once (128 × 4 FP32 values). Rank-8 shared RoutedMLP branches have 6,476 parameters each: router 8→16→16→16→4 and network 12→48→48→48→8; graph variants below use this conservative branch allowance. Critic allocations are included. Frozen weights have no parameter gradients or optimizer moments.

For ParticleGAN 0.11.0 `bispf_optimized`, DualNorm has zero momentum: do **not** add Adam's two full-sized moments. We reserve **64 bytes per trainable scalar** for FP32 parameters/gradients, old/displaced parameters and normalization/constraint temporaries, including protected-normal `.double()` buffers; small scalar step/constraint state fits the workspace allowance. This is a conservative transient envelope, not 64 bytes of persistent optimizer state. No model-weight EMA copy is allocated in the supported RAW policy. A separately identified supervised AdamW comparator uses its actual two FP32 moments.

Activation estimates deliberately include FP32 routed branches and two FP32 attention-sized buffers for eager/math attention. With `L` layers, width `H`, FF width `I`, query heads `N`, tokens `T` and effective accumulated batch `B`, the uncheckpointed training allowance is `64BLTH + 12BLTI + 8BLNT²` bytes. With block checkpointing it is `8BLTH + 64BTH + 12BTI + 8BNT²`; hybrid attention uses only its full-attention layer count in the quadratic term, with recurrent state added separately. These are explicit planning allowances for saved tensors/recomputation, not a promise about every allocator/kernel. Sampling prefill uses `16BTH + 8BTI + 8BNT²`, plus actual KV/recurrent state and any materialized logits. Runtime/workspace allowances below cover Python/framework/CUDA workspaces, allocator fragmentation and the small game state; an executor must still enforce live resource checks.

Host budgets count the stated CPU loading peak, tensor caches and serialization copies plus workspace. They exclude an optional **reclaimable OS file cache** of the complete checkpoint: add the checkpoint file size if deliberately retaining that cache in RAM. Model files remaining on disk are not resident GPU weights. Training duration/update count changes checkpoint disk consumption, not this live memory calculation. RAW/FULL serialization can add CPU copies of trainables; those are included in the host allowances.

Model: **Qwen/Qwen3.5-0.8B**, revision **`2fc06364715b967f1860aea9cf38778875588b17`**.

The table rounds each calculated phase sum upward to the next2 GiB. The byte ledger below gives the unrounded calculation; per-GPU split estimates are listed in device order.

| Phase | Allocation ledger (GiB before rounding) |
|---|---|
| Particle hidden-response training (RAW planning policy) | frozen: 1.6269; trainable_transient: 0.3963; activations: 3.9375; recurrent_state_upper: 0.0769; cache: 0.0020; workspace: 4.0000 |
| Separate ordinary LoRA supervised comparator | frozen: 1.6269; AdamW_weights_grad_moments: 0.0806; activations: 3.9375; recurrent: 0.0769; FP32_logits: 1.4209; workspace: 4.0000 |
| Bounded finished-text sampling | frozen: 1.6269; adapter: 0.0248; full_attention_KV: 0.0234; recurrent: 0.0176; prefill_logits: 0.5791; workspace: 4.0000 |

Detailed arithmetic and assumptions:

- **BF16 plus FP32 island bytes**: `1746882752`.
- **scope all LoRA elements**: `5411328`.
- **projections**: `186`.
- **PG generator plus critic allowance**: `6649241`.
- **recurrent formula**: `18 linear-attention layers *16 value heads *128keydim*128valuedim*4 bytes + conv4 state; full-attention KV only6 layers`.
- **legacy EMA extra bytes**: `26465504`.

Scope and limits:

- The model is hybrid:18linear-attention and6full-attention layers, with a frozen vision tower. The ledger counts the complete873,438,784-element checkpoint, not only the text trunk. Runtime Runner is external; verify it uses this resolved model revision and BF16 before applying this profile.
- Legacy train_particle currently constructs/selects an EMA shadow. That is historical implementation behavior and unsupported by the standing RAW/no EMA policy. Its extraFP32 shadow is recorded here only as an allocation caveat, not an instruction to enable EMA; these rounded budgets have room for that shadow. No runtime change is made by this document.
- Ordinary LoRA genuinely uses AdamW and is budgeted separately. ParticleGAN uses DualNorm; its state is not AdamW. The8192-token evaluation plan requires recalculation; the2048-total sampling profile is deliberately bounded.

Source facts (read as text; no model execution):

- [configs/search.json](configs/search.json)
- [configs/lora.json](configs/lora.json)
- [src/qwen35_sliders/host.py](src/qwen35_sliders/host.py)
- [src/qwen35_sliders/training.py](src/qwen35_sliders/training.py)
- [src/qwen35_sliders/upstream.py](src/qwen35_sliders/upstream.py)
- [Pinned model metadata](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17) (architecture/parameter metadata or exact file-size lock as described above).
- ParticleGAN 0.11.0 release `b86f0379b4d887f4cde88d686aedc04661a2b61b`, `recipes.py:bispf_optimized`, `optim/dualnorm.py`, `optim/direction_blend.py` and `optim/constraint_geometry.py`; shared core `routed_projection.py`, `reference.py:RoutedMLP`, `audio.py:AudioResponseCritic`. These optimizer formulas are shared dependencies, not a new optimizer implementation.

Calculated 2026-11-14. Counts, live runtime qualification and memory requirements are separate facts.
