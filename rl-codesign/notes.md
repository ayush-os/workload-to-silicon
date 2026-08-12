# RL Post-Training Codesign — Notes

Working notes for project #6 (`rl-codesign`) — Rollout vs. Training. See
`spec.md` for the full phase plan; see `handoff.md` for current status and
where to pick up next. This file is a running log, appended to phase by
phase — not yet a polished writeup (that consolidation happens at project
completion, matching how `disagg_and_placement_notes.md` and the
sibling `numerics-and-sparse-attn/README.md` were finished).

---

## Phase 0 — Setup

### Reading list, synthesized

**Colocated/resharding:**

- **HybridFlow** (arXiv 2409.19256): 3D-HybridEngine resharding —
  separate weight buffers for train vs. generation mode; transition =
  reload offloaded generation weights to GPU + concurrent intra-group
  all-gathers to re-partition into the target 3D layout. **"Zero memory
  redundancy" is a memory-only claim, not a time claim** — real, measured
  transition cost: naive resharding eats up to **36.4%** of iteration time
  on a 70B model when stages sit on different devices; HybridFlow's engine
  cuts that by 55.2% average (~11.7s), up to 89.1% (~78.2s) — reduced,
  never eliminated.
- **RLHFuse** (arXiv 2409.13221 — found fresh via search, not in the
  original spec's reading list): two distinct colocated bottlenecks —
  generation stragglers (long-tailed response lengths) and training
  pipeline bubbles — fixed via dynamically migrating straggler samples to
  dedicated instances mid-iteration, repurposing freed GPUs. A third
  architecture point, more granular than pure colocated/disaggregated.

**Disaggregated/asynchronous:**

- **RhymeRL** (arXiv 2508.18588): 3 pools (rollout/reward/train), strict
  **one-step staleness** enforced (not more), weight sync via host-memory
  buffer async propagation, speculative decoding using historical rollout
  tokens as drafts (65–79% acceptance).
- **AReaL** — the real staleness-quantification anchor for Phase 4. Formal
  bound `⌊(N_r−1)/B⌋ ≤ i + η`, **η=4 recommended for coding, η=8 for math
  reasoning** (η=0 = fully synchronous). Correction mechanism: "decoupled
  PPO" — importance ratio taken against a *proximal* policy (params just
  before the current update), not the original behavior policy. Reported
  speedup vs. synchronous: 1.49×–2.27× (1.5B→32B models), up to 2.77×
  headline. ~75%/25% inference/training GPU split on a 512-GPU H800 setup.
- **StreamRL**: one-step async (staleness=1), dynamic DP-unit-granularity
  resource rebalancing, skew-aware dispatch.
- **BiDiRL**: the real hybrid, hot-switch end of the spectrum — GPUs
  default to a primary role but can temporarily execute the other's work,
  gated by predicted-time-saved vs. measured switch cost (3.6–7.7s per
  switch, 2B–4B models). **Caveat**: the paper's "idleness is a problem on
  both sides" claim is qualitative (timeline diagrams), not a measured
  utilization table — don't cite it as a quantified number later.

**Workload grounding — DeepSeek-R1 (arXiv 2501.12948):**

- GRPO confirmed: no critic, group-relative advantage = mean/std
  normalization within group (Eq. 3), PPO-style clipped objective +
  explicit KL term (Eq. 1, β=0.001), Schulman k3 KL estimator (Eq. 2).
- **K = 16** (group size), confirmed identical across R1-Zero, R1's first
  RL stage, and the 32B distillation ablation. Batch = 32 prompts × 16
  samples = 512 completions/step.
- **KL penalty is kept, not dropped** — but the reference model is **not
  literally frozen**: swapped for the latest policy every 400 steps.
- Response length: max generation length 32,768 tokens before step 8.2k,
  then **65,536** afterward — grows over training, not a fixed anchor.
- Infra: 512×H800 GPUs, R1's RL stage ≈80 hours. **Training precision not
  stated** anywhere in the paper.

**Miles** (SGLang/RadixArk — found mid-Phase-1 via a direct question, not
in the original reading list, worth folding in retroactively): a real,
current production framework implementing *both* colocated and
disaggregated layouts as a launch-time config choice (Ray placement
groups), not a research pick of one side. Unified BF16/FP8/MXFP8/INT4-QAT
precision spanning rollout+training — real confirmation the
precision-conversion-at-boundary problem is a genuine production concern,
not a hypothetical the spec invented. Weight sync over dedicated NCCL/RDMA
channels, Ray handling only the control path. Fully async mode
(queue-based, no per-iteration blocking) — no explicit staleness bound
given (unlike AReaL's η). **New scope flag, not in the original spec**:
"Rollout Routing Replay" — MoE expert-routing decisions can drift between
rollout (SGLang) and training (Megatron) recomputation, destabilizing the
policy update if uncorrected. Directly relevant since this project's
workload reuses `moe-routing-notes.md`'s formulas (DeepSeek-style MoE +
MLA) — not addressed by any of the six original reading-list papers.
**Not yet scoped in or out — flagged for Phase 2/3.**

### Decision 1: keep the reference model

Settled by R1's own grounding, not a free design choice — GRPO's
objective explicitly includes the KL term. **Kept.**

Real wrinkle carried forward: the reference model is refreshed every 400
steps, not frozen for the whole run — a periodic weight-copy event, same
shape as one training→rollout sync, just far rarer (a footnote for Phase
3, not a new axis).

FLOPs shape: a single forward pass over each full generated response to
get log-probs — **prefill-shaped** (parallel over sequence length), not
decode-shaped like rollout's own generation. Gets its own line item in
Phase 1's accounting, not folded into rollout.

Placement: shares rollout's chip pool by default (both pure inference, no
training state) unless something concrete pushes back on that later.

### Decision 2: fresh treatment for precision (not a direct reuse of decode's §4.1 framework)

Decode's coupled/decoupled framework (`decode_notes.md` §4.1) answers
"does a numerics lever flip *one workload's own* regime" — RL's
weight-sync question is a different kind of question: the one-time cost
of the BF16(training)→FP4(rollout) conversion at the sync boundary, a
transition cost *between* two workloads, not a within-workload lever.
Direction-reversal detail confirmed real: TPU 8i's FP4 peak is achieved
via *native* FP4 compute — storage and compute precision match on the
rollout side (the "coupled" case in decode's own vocabulary), but that's
incidental, not the actual question being asked. **Fresh derivation
needed for Phase 3**, carrying over only the vocabulary/discipline, not
the formula itself.

TPU 8i's real BF16 peak: **not stated** in Google Cloud's primary source
(only FP4=10.1 PFLOPS is given). Provisional estimate via the
halving-per-precision-step pattern documented for the sibling TPU 8t chip
(FP4 12.6→FP8 6.3→BF16 3.15 PFLOPS): applying the same ratio to 8i gives
**≈5.05 PFLOPS FP8, ≈2.525 PFLOPS BF16** — **unconfirmed, needs a real
verification pass before Phase 2 depends on it.**

---

## Phase 1 — Rollout-side roofline

### Mechanism: why rollout is decode-heavy, and what K-way prefix sharing actually does

Two separate effects, not one — worth keeping distinct since it's easy to
conflate them:

1. **Root cause**: response length ≫ prompt length (R1's responses run
   into the tens of thousands of tokens vs. a much shorter prompt) —
   decode-heavy per request, independent of K entirely, same shape as
   `decode_notes.md`'s original prefill-vs-decode argument (many
   sequential decode steps dwarf one parallel prefill pass).
2. **K-way prefix sharing amplifies, doesn't cause, the decode-heaviness**:
   K completions share one prefill pass instead of K independent ones.
   Savings = **(K−1)/K = 15/16 = 93.75%** of prefill's own cost avoided by
   sharing — a clean, R-independent number, and the direct answer to
   Phase 1's stated deliverable question.

### FLOPs: prefill-once vs. K×decode

Formulas reused directly from `disagg_and_placement_notes.md` Phase 2b
(MLA absorbed-decode / naive-prefill, arXiv 2405.04434 §2.1) and MoE
routing's FFN formula — no new mechanism needed, per-layer:

```
Prefill/layer(P)              = 675,938,304·P + 81,920·P²
Decode/layer/completion(R,P)  = 810,156,032·R + 278,528·R·P + 278,528·R(R-1)/2
```

(×60 layers for the full model — DeepSeek-V2's real layer count, layer-1's
dense-FFN exception carried over as this repo's own known small gap)

Assumptions: **P=1,024** (this project's own pick — no R1 source exists
for prompt length, flagged, not sourced), **K=16** (R1-confirmed), two
response-length anchors R=8,192 (this repo's existing Llama-3 native-cap
convention) and R=65,536 (R1's late-training cap).

| | R = 8,192 | R = 65,536 |
|---|---|---|
| Prefill total (once) | 46,683,610,152,960 FLOPs (≈46.7 TFLOPs) | same — R-independent |
| Decode total (K=16) | 17,585,249,672,232,960 FLOPs (≈17.6 PFLOPs) | 643,114,830,806,384,640 FLOPs (≈643.1 PFLOPs) |
| decode : prefill ratio | ≈377 : 1 | ≈13,779 : 1 |
| prefill's FLOPs share of total | ≈0.265% | ≈0.0073% |

N (number of prompts in the batch) cancels out of every ratio above —
holds regardless of batch size.

**Genuinely non-trivial finding**: decode's own accumulated cost across a
full generation is *not* O(R) — the `R(R-1)/2` term makes it **O(R²) for
large R**, because each decode step attends to the growing KV cache.
Crossover where the quadratic term overtakes the linear term: **R\* ≈
5,817 + 2P** (≈**7,865** at P=1,024). Below R\*, decode's growth is closer
to linear; above it, quadratic dominates (89% of decode's cost at
R=65,536 is the quadratic term). This is the standard
quadratic-attention/growing-KV-cache fact — not a new mechanism — but
it's the same one already flagged as an out-of-scope open thread in
`decode_notes.md` ("sparse/linear attention — the natural next lever
after GQA"). This project is the first in the repo where a *single
request's own response length* crosses into that regime without needing
large batch or long prompt context to get there — worth a scope-note
pointer back to that open thread, not fresh territory.
