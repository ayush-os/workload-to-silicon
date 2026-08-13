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
- **RLHFuse** (arXiv 2409.13221 — was already in the original spec's
  reading list as "course reading," just without a citation; found the
  actual arXiv number via search): two distinct colocated bottlenecks —
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

### Decision 2: fresh treatment for precision (not a direct reuse of decode's §4.1 framework) — revised after real search pass

**Original framing (superseded below):** decode's coupled/decoupled
framework (`decode_notes.md` §4.1) answers "does a numerics lever flip
*one workload's own* regime" — RL's weight-sync question looked like a
different kind of question: the one-time cost of a BF16(training)→FP4
(rollout) conversion at the sync boundary, a transition cost *between*
two workloads. That framing assumed BF16 is the real training precision
for this repo's TPU 8i chip.

**Real search pass (post-Phase-1), findings:** checked the primary
Google Cloud TPU 8t/8i deep-dive, the comparative TPU 7x/8t/8i codelabs
writeup, and three secondary sources (Introl, IntuitionLabs, a
Medium/dev.to TPU-history piece). **No BF16 peak FLOPs number exists
publicly for *either* 8th-gen chip** — not a documentation gap on 8i
specifically, a real absence on both:

- **TPU 8i** (inference-optimized): 10.1 PFLOPS FP4/chip (confirmed,
  already in use throughout this repo), 11.6 EFLOPS FP8 at the
  1,152-chip pod level. No BF16 anywhere.
- **TPU 8t** (training-optimized): 12.6 PFLOPS FP4/chip, 121 EFLOPS FP4
  at the pod level. No FP8 or BF16 anywhere either.

**This is the real, load-bearing new fact**: the 8th generation is the
first time Google has split the TPU line into two physically distinct
chips — 8t "built for large-scale pre-training and embedding-heavy
workloads," 8i "built for post-training and inference." Both are
marketed purely around FP4 (with FP8 as 8i's pod-level aggregate, not a
per-chip spec). That's real, sourced evidence against the original
premise, not just a missing number — it suggests this hardware
generation doesn't treat BF16 as the standard training baseline at all,
which is what Decision 2 originally assumed.

**Decision 2, revised**: drop the BF16-training assumption entirely.
Model **training on TPU 8t, at 8t's own native FP4 peak (12.6
PFLOPS/chip)** — the real training-optimized sibling chip, not a
borrowed/derived BF16 number on the inference chip. This reframes the
weight-sync question: it's no longer a BF16→FP4 *precision conversion*
between one chip's two modes, it's an **FP4(8t)→FP4(8i) cross-chip
transfer** — same nominal format on both ends, but still a real
transition worth deriving in Phase 3 (different chips/dies almost
certainly means the sync isn't free even at matched precision — network
transfer, and possibly still a real requantization step if the two
chips' FP4 formats/scaling conventions differ, unconfirmed and worth a
Phase 3 check rather than assuming byte-identical). Carries a second
real implication forward: **the real 8t/8i chip split is itself
evidence for the disaggregated end of Phase 3's spectrum** — Google
ships training and inference as separate silicon by design in this
generation, which argues against "one pool of chips reshards between
modes" being the natural hardware-native shape, at least for this
generation's chips.

**Fresh derivation still needed for Phase 3** for the cross-chip
transfer cost itself — carrying over the vocabulary/discipline from
decode's §4.1 framework, not its formula (that framework answered a
within-chip precision-regime question; this is now a cross-chip
transfer question, an even bigger departure from the original framing
than first thought).

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

### Bytes and wall-clock time

Prefill is compute-bound, decode is memory-bound — the established
convention throughout this repo (`decode_notes.md`, disagg's dense/MoE
work) — confirmed, not assumed, for this project's own numbers.

**Prefill**: at a single isolated prompt (batch=1), AI comes out *below*
TPU 8i's ridge point (≈1,174 FLOPs/byte, = 10.1 PFLOPS / 8.6 TB/s) — would
flip prefill memory-bound, contradicting every prior finding in this
repo. Resolved: crossover only needs **B≈3 concurrent prompts** batched
together to clear the ridge (weight bytes amortize across the batch,
FLOPs scale linearly with B) — trivially satisfied by any real rollout
(R1 itself batches 32 prompts/step). Compute-bound confirmed, not just
assumed.

**Prefill wall-clock = 46,683,610,152,960 FLOPs / 10.1 PFLOP/s ≈ 4.62 ms.**

**Decode bytes — two real modeling forks, not pure formula reuse:**

1. FFN bytes depend on *cross-prompt* batch size (how many distinct
   experts get touched scales with total concurrent decode batch, not
   just K). Reused disagg's real, HBM-capacity-grounded **N=640**
   (8-device EP group, 258.05 MB/device/layer FFN bytes, 99.43%
   expert-table coverage) rather than re-deriving a new saturation curve —
   justified because R1's own real minibatch (32×16=**512**) is close
   enough in scale to 640 for the same near-saturated regime to plausibly
   apply. **Flagged assumption, not free reuse.**
2. KV-cache read bytes — a term FLOPs never needed. Each of the ~640
   concurrent requests reads its own MLA cache (576 elements/token/layer)
   every step. Modeled at steady-state: average context length ≈ P + R/2
   across the batch (matching disagg's own steady-state-averaging
   convention, not per-request staggering).

Per-device decode bytes/layer = 108.17 MB (attention weights, replicated
across the EP group) + 258.05 MB (FFN shard, N=640) + cache (80
requests/device × avg context × 288 bytes):

| | R = 8,192 | R = 65,536 |
|---|---|---|
| Cache bytes/layer/device | 117.96 MB | 778.57 MB |
| Total bytes/layer/device | 484.18 MB | 1,144.78 MB |
| Total bytes/device (×60 layers) | 29.05 GB | 68.69 GB |
| Time/decode-step (÷8.6 TB/s) | 3.378 ms | 7.987 ms |
| **Decode wall-clock (×R steps)** | **27.67 s** | **523.5 s** |

**Combined:**

| | R = 8,192 | R = 65,536 |
|---|---|---|
| Prefill | 4.62 ms | 4.62 ms |
| Decode | 27,672 ms | 523,536 ms |
| **Prefill's share of wall-clock** | **≈0.0167%** | **≈0.00088%** |

Smaller than the FLOPs-only shares — confirms decode's memory-bound
regime makes its real wall-clock dominance even more extreme than raw
FLOPs suggested, because decode's effective throughput is lower than
prefill's near-peak compute-bound execution.

### Key Findings, Phase 1

1. **Rollout wall-clock is completely dominated by decode**, driven
   specifically by KV-cache movement (not FFN weights) at realistic
   response lengths — prefill is ≈0.001–0.017% of total rollout
   wall-clock, 4–5 orders of magnitude below decode.
2. **Mechanism chain**: response length ≫ prompt length (decode-heavy by
   construction, independent of K) → K-way prefix sharing further shrinks
   prefill's already-small share (93.75% savings, R-independent) →
   decode's own accumulated cost is quadratic-dominated in R past
   R\*≈7,865 (growing KV cache, the standard mechanism, already an open
   thread from `decode_notes.md`) → converting FLOPs to real wall-clock
   via the roofline (decode memory-bound, prefill compute-bound) makes the
   dominance more extreme still, because cache-read bytes — not FFN
   weights — become decode's largest wall-clock cost term at long R.
3. **Sensitivity note, deliberately not chased further**: the N=640
   cross-prompt batching assumption is load-bearing for the exact decode
   number, but the qualitative conclusion is robust to a wide margin
   (would need something like a 1,000×+ swing to flip it) — a K=16-alone
   sensitivity run would also be measuring the wrong thing (isolated
   non-batched decode, not a real deployment scenario any RL system would
   actually run), so deliberately not run. Matches this repo's own scope
   discipline (`disagg_and_placement_notes.md`'s "cut it and flag it"
   convention).
4. **Open flag, real but unaddressed**: Miles' "Rollout Routing Replay"
   issue (MoE routing decisions drifting between rollout and training
   recomputation) is a real mechanism specific to this project's MoE+MLA
   workload, not covered by any of the six original reading-list papers,
   and not yet scoped in or out. Needs an explicit decision in Phase 2 or
   3.

---

## Phase 2 — Training-side roofline (in progress)

### Backward-pass FLOPs multiplier: confirmed 4×, not 3× — and why

**Mechanism (sourced, not re-derived from scratch)**: the standard "6ND"
training-FLOPs rule decomposes as 2 (forward) + 2 (grad-wrt-activation)
+ 2 (grad-wrt-weight) per parameter-token pair = 3× forward total,
*without* activation recomputation. This is pure backprop algebra — each
weight matrix participates in exactly one forward matmul and two
same-sized backward matmuls — and it holds per-matmul regardless of
dense vs. MoE routing, **as long as only the actually-activated compute
is counted on both sides** (which this repo's own MoE FFN formulas
already do). So MoE sparsity does not, by itself, break the 2×-backward
ratio — confirms the intuition that this "has to be" 2× mechanically.
(Source: FLOPs-calculus breakdown cross-checked against standard 6ND
literature.)

**What does change it**: with activation recomputation (recompute
forward activations during backward instead of storing them), backward
itself costs ~3× forward (recomputed-forward + grad-activation +
grad-weight, each ≈1×), making the **total 4× forward, not 3×**.

**Is recomputation actually forced here, or just common practice?**
Checked against a real, dedicated memory-analysis paper for DeepSeek-V3's
exact architecture (Zhang & Su, "Memory Analysis on the Training Course
of DeepSeek Models," arXiv 2502.07846) — gives explicit per-layer
activation-memory formulas for both no-recomputation and
full-recomputation cases. **Note: this source uses DeepSeek-V3's own
dims (h=7168, n_h=128, 61 layers), not V2's** (this project's own
convention elsewhere, 60 layers, `576 elements/token/layer` MLA cache
from arXiv 2405.04434) — same model family, not identical; flagged, not
silently blended.

MLA's per-layer, no-recomputation activation memory has a quadratic
term, `5·b·n_h·s²` bytes (the attention-score matrix, one of the terms
selective recomputation specifically targets per Korthikanti et al.
2205.05198). At batch=1, using this project's own two R anchors as s:

| R | quadratic term, 1 layer | vs. TPU 8t's 216 GB HBM |
|---|---|---|
| 8,192 | 42.95 GB | ≈20% of the *entire chip* for one layer's attention scores alone |
| 65,536 | 2,748.8 GB | **≈12.7× the entire chip's capacity**, one layer, batch=1 |

Naively summed across 61 layers with no recomputation at all: ≈2,722 GB
(R=8,192) to ≈168,490 GB (R=65,536) — both far beyond any real chip's
HBM, before even adding FFN/MoE activations, weights, gradients, or
optimizer state. **Conclusion: activation recomputation isn't an
optimization choice at this project's response lengths — it's numerically
required for training to run at all**, most acutely at R=65,536.

**Working number for Phase 2**: use **4× forward FLOPs** for training
step cost (not the naive 3×), on the grounds that recomputation is
forced, not optional. Caveat: DeepSeek-V3's own case study uses
*selective* recomputation in practice (targeting just the
low-FLOP/memory-ratio ops like the attention-score matrix, not the whole
layer) — full recomputation is the cleaner 4× case; selective likely
lands somewhere between 3× and 4×, since only part of the forward pass
gets recomputed. Treating 4× as the current working assumption, flagged
as an upper-bound simplification, not yet refined to selective's real
partial multiplier.

### Adam optimizer state memory

**Real parameter count reused, not borrowed from V3**: this repo already
sourced DeepSeek-V2's own total/activated params directly from the paper
(`moe-routing-notes.md` §3.1) — **236B total / 21B activated per token**.
Using V2 here resolves part of the earlier V2-vs-V3 flag for this
specific building block.

**Why total (236B), not activated (21B)**: Adam's momentum/variance
state must persist per-parameter regardless of per-token routing
sparsity — a real training batch routes tokens across nearly every
expert, so nearly every expert receives a gradient and needs its own
optimizer state every step. MoE's sparsity saves FLOPs, not optimizer
memory. (Same logic already implicit in this repo's MoE weight-memory
accounting — routed experts are all *stored*, just not all *computed
against* per token.)

**Multiplier, sourced from the canonical reference** (Rajbhandari et al.,
ZeRO, SC20): mixed-precision Adam needs **12 bytes/param** — 4 (FP32
master copy of the parameters) + 4 (FP32 momentum) + 4 (FP32 variance).
This is the standard, most-cited convention, distinct from the
weights+gradients themselves (typically 2+2 bytes in BF16, a separate
line item). **Worth being precise about where the 12 comes from**: Adam
itself only tracks 2 numbers/param (momentum + variance) — 8 bytes/param
if both are FP32, matching the naive expectation. The 3rd term (the FP32
master weight copy) isn't part of Adam's algorithm at all — it's a
mixed-precision-training necessity: compute happens in low-precision
weights (FP4 here), and Adam's tiny per-step updates would underflow to
zero if applied directly to a low-precision copy, so a full-precision
master copy accumulates every update and periodically re-quantizes down
to produce the fast compute copy. Easy to misattribute all 12 bytes to
"Adam" when only 8 of them are — the other 4 are a separate, distinct
cost that mixed-precision training bolts on. A more aggressive variant exists — the same DeepSeek
memory-analysis paper used for activation memory (arXiv 2502.07846) ran
an illustrative case with BF16 momentum/variance (**8 bytes/param**: 4
FP32 master + 2 + 2) — but that paper explicitly disclaims its configs
as illustrative, not DeepSeek's real recipe, so 12 bytes/param is the
better default working assumption; 8 bytes/param noted as a real,
sourced alternative if a memory-constrained config is later needed.

| | Total optimizer state (236B params) | vs. TPU 8t's 216 GB/chip |
|---|---|---|
| 12 B/param (canonical) | 2,832 GB (≈2.83 TB) | ≥13 chips, optimizer state alone |
| 8 B/param (reduced) | 1,888 GB (≈1.89 TB) | ≥9 chips, optimizer state alone |

**Real-world corroboration**: R1's own reported training setup uses
AdamW with **optimizer CPU offloading** — independent confirmation that
optimizer state doesn't fit on-device even for a real GPU training
cluster, consistent with this chip-count finding rather than an artifact
of TPU 8t's specific capacity.

**No tension with Decision 2's FP4-native training assumption**: FP4 is
the *compute* precision (forward/backward matmul throughput); the
optimizer's FP32 accumulation is a separate concern about numerical
stability of a long-running moving average, and mixed-precision training
always keeps these two decoupled — fast low-precision compute, high-
precision accumulator state. Not a contradiction to reconcile, just
worth stating explicitly given how much this project has already had to
reason about precision.

**Implication for Phase 2's own accounting**: training cannot be modeled
as fitting on a single TPU 8t chip — optimizer-state sharding (ZeRO
stage 1, "os," at minimum) across multiple chips is structurally
required. Phase 2's eventual per-device budget should follow this
repo's own established convention (Phase 1's N≈640 per-device decode
accounting) rather than a single-chip model.

**Relationship to the activation-recomputation finding — two separate
constraints, not one reinforcing the other.** Worth being precise here:
recomputation *reduces* pressure toward more chips, it doesn't add to
it — it only touches activations (recreatable by rerunning the forward
pass), bringing the no-recompute case's absurd 2.7 TB/layer down to a
small per-layer footprint, which is *why* activation memory stops being
the dominant constraint once recomputation is applied. Optimizer state
is a different memory category entirely — Adam's momentum/variance are
*persistent accumulated state*, nothing to recompute — so no activation-
side technique touches this 2.83 TB floor. Two independent constraints:
activations *would have* forced even more sharding than optimizer state
does (165 TB total, uncapped, vs. 2.83 TB) had recomputation not
defused it; optimizer state has no equivalent defuse and is the one that
actually sets the current chip-count floor.

**AdamW hyperparameters (real, not this project's own choice)**: lr=1e-6,
β1=0.9, β2=0.98, weight decay=0.1, constant schedule — sourced from R1's
reported training config. Not yet independently re-verified against the
arXiv primary text word-for-word this pass (the search result may blend
R1's paper with a secondary description) — flagged for a primary-source
check if these specific values matter later (they don't affect the
memory/FLOPs derivation above, only optimizer *dynamics*).

### GRPO's critic-free saving, quantified against a PPO baseline

**Baseline sourced, not invented**: GRPO's own origin paper (DeepSeekMath,
arXiv 2402.03300) states explicitly why the critic was dropped — "the
critic model is comparable to the policy model in size." Model the PPO
critic as a second full 236B-param MoE/MLA model, not a lightweight
shared-backbone value head.

Reusing every number already derived for the policy (doubles cleanly,
since critic = same architecture/size):

| | Policy alone | + PPO critic | GRPO's saving |
|---|---|---|---|
| Optimizer state | 2.83 TB (≈13 chips) | 5.66 TB (≈26 chips) | 2.83 TB / ≈13 chips, 50% |
| Weights stored | 236B params | 472B params | a full second copy |
| Activation memory | forced into recomputation regime | same regime, doubled | doubled burden avoided |
| Training FLOPs/step | ~4× forward | ~8× forward | ~halves total training compute |

**Net finding**: not "GRPO is simpler" as a qualitative claim — a flat
**~2× reduction across every major training resource** (chips, memory,
FLOPs), because the critic isn't a small side-model, it's a second full
copy of everything already derived for the policy. Not yet refined:
whether real PPO-RLHF critics are ever smaller/shared-backbone in
practice (would shrink this saving) — flagged, not chased, since the
sourced GRPO-paper rationale is the more defensible anchor than a
hypothetical smaller critic.

### Per-device parallelism config: fitting training on TPU 8t

**Corrected mental model first**: DP alone doesn't save memory — it
replicates the full model per replica, and only *enables* ZeRO-style
sharding of that redundancy. EP/TP/PP shard the model directly; SP
shards activations specifically.

**Starting point (EP=8 only, inherited from Phase 1's rollout config)**:
per-device params = 8.95B replicated attention + 31.14B local FFN =
40.09B. At 16.5 bytes/param (0.5 FP4 weights + 4 FP32 grads + 12 FP32
optimizer), that's **661.5 GB/device — 3.06× over TPU 8t's 216 GB**,
before activations. Breaking down *why*: attention's replication waste
is 147.7 GB, but the FFN pool (routed + shared experts) is 513.8 GB —
the dominant cost isn't the "obviously wasteful" replicated attention,
it's simply that there's far more total FFN parameter mass (≈225.5B)
than attention mass (≈8.95B) in this architecture.

**Refined FFN accounting** (routed vs. shared experts, since shared
experts don't shrink no matter how wide EP goes — same kind of fixed
floor as attention's replication): 160 routed experts × 23,592,960
params/expert × 59 MoE layers = 222.7B (this is what actually shrinks
with EP degree); 2 shared experts × same × 59 layers = 2.78B (fixed
floor, replicated on every EP rank regardless of width).

**Solution found: TP=4 (attention) × EP=40 (experts) = 160 devices**:

| TP | EP | attn/dev | ffn/dev | total | GB | vs. 216 GB |
|---|---|---|---|---|---|---|
| 1 | 8 | 8.95B | 30.62B | 39.58B | 653.0 | over |
| 2 | 32 | 4.48B | 9.74B | 14.22B | 234.6 | over |
| 2 | 40 | 4.48B | 8.35B | 12.83B | 211.7 | over (only ~4GB headroom) |
| **4** | **40** | **2.24B** | **8.35B** | **10.59B** | **174.7** | **OK, ~41GB headroom** |

TP=4×EP=40 clears the weights+gradients+optimizer budget with real
margin. Bonus: 160 total devices divides the 160 routed experts evenly
(4/device), no load-balancing awkwardness. **PP turned out unnecessary**
— a genuinely non-obvious result; expected 60 layers to force pipeline
splitting, but a modest TP on attention plus widening the already-
established EP mechanism closed the gap without touching the layer axis.

**Activation memory against the ~41GB headroom — deliberately not
precisely rederived under this exact TP×EP grid**, and flagged as such
rather than chased further: sharding under TP/EP can only *reduce*
per-device activation memory relative to the earlier unsharded estimate
(~7–57GB depending on R, computed with no parallelism applied) — each
device only computes/stores activations for its own attention shard and
its own local experts' tokens. So the headroom check is against a
conservative upper bound already, not a coin-flip estimate; a precise
recompute under this specific grid (using V2's own dims, yet another
instance of the still-open V2/V3 mismatch) would be diminishing-returns
effort for a number already known to fit safely. Matches this project's
own repeated "cut it and flag it" discipline elsewhere (Phase 1's N=640
sensitivity run, Phase 5's own scope note, spec.md's explicit
instruction).

**Validation check: was TP actually necessary, or just one option among
several?** Tested EP alone, maximally widened (EP=160, one routed expert
per device, no TP at all): 216.6 GB — still *over* the 216 GB budget,
with zero headroom. Confirms attention-sharding (TP or an equivalent)
wasn't a convenient add-on, it was structurally required — no amount of
EP-widening alone gets there.

**TP vs. a ZeRO-style alternative for attention — checked, not assumed
equivalent**: the other candidate for de-duplicating attention's 8×
replication was ZeRO-3-style sharding (shard weights+grads+optimizer,
all-gather transiently before compute) rather than TP's distributed-
matmul approach. For a matched degree, the two land at essentially the
same steady-state per-device memory (both are "divide by N," just
different communication mechanics — TP does in-layer all-reduce, ZeRO
does gather-before/scatter-after). TP has a slight edge (no transient
full-weight materialization spike during the gather) and matches the
real reference source's own case study (TP for attention, EP for
experts) — so TP was the better-grounded choice, not an arbitrary pick.

**Reflexive-PP check**: pipeline parallelism is often the "default first
reach" for a many-layer model (60 layers here), but it doesn't fix
replication waste on its own (would still need EP/TP within each stage)
and adds real pipeline-bubble wall-clock cost. TP+wider-EP closed the
gap without it — good discipline confirmation that PP wasn't reached for
reflexively just because the layer count looked big.

**Meta-pattern across Phase 1 vs. Phase 2, worth remembering going into
Phase 3**: Phase 1's constraint was *bandwidth* (decode is memory-
bandwidth-bound — how fast can you move KV-cache bytes). Phase 2's
constraint is *capacity* (does the whole model + optimizer + activations
fit on the chip at all). Same roofline-adjacent instinct, structurally
different axis — worth keeping distinct when Phase 3 starts comparing
rollout and training resource costs directly.

**Phase 2 checkpoint met**: training-step FLOPs (4× forward, recomputation-
forced), bytes/memory (Adam optimizer state, per-device parallelism
config), and GRPO's critic-free saving (flat ~2×) are all derived and
sourced.

---

## Open Threads / Flags carried into Phase 2+

- ~~TPU 8i's real BF16 peak is unconfirmed~~ — **resolved**: no BF16
  number exists publicly for either 8th-gen chip (see Decision 2,
  revised, above). Phase 2 models training on TPU 8t at its own native
  FP4 peak (12.6 PFLOPS/chip) instead.
- **New from the Decision 2 revision**: the 8t↔8i weight-sync cost in
  Phase 3 is now a cross-chip FP4→FP4 transfer, not a BF16→FP4
  conversion — still needs its own derivation (network cost, and an
  unconfirmed question of whether the two chips' FP4 formats/scaling
  conventions actually match byte-for-byte).
- **V2 vs. V3 architecture dims**: this project has used DeepSeek-V2's
  numbers throughout (60 layers, 576-elements MLA cache, arXiv
  2405.04434), but the only real activation-memory source found for
  Phase 2 (arXiv 2502.07846) analyzes V3's dims (61 layers, h=7168,
  n_h=128). Same family, not identical — decide whether to switch the
  whole project to V3 for consistency or find/derive V2-specific
  activation-memory numbers before Phase 2's activation-memory work goes
  further.
- **Selective vs. full recomputation**: DeepSeek-V3's own real practice
  uses *selective* recomputation (targeting specific ops, not the whole
  layer) — the 4× FLOPs multiplier assumes full recomputation, an
  upper-bound simplification. Refine to selective's real partial
  multiplier before Phase 2's training-FLOPs number is treated as final.
- ~~TPU 8t's HBM bandwidth/capacity not yet looked up~~ — **resolved**,
  see new subsection below.

### TPU 8t full spec pass (post-Decision-2-revision)

Same primary Google Cloud deep-dive used for the FP4 numbers has a full
comparison table — sourced directly, not derived:

| | TPU 8t (training) | TPU 8i (rollout) |
|---|---|---|
| Peak FP4 | 12.6 PFLOPS/chip (121 EFLOPS/pod) | 10.1 PFLOPS/chip (11.6 EFLOPS FP8/pod) |
| HBM capacity | 216 GB | 288 GB |
| HBM bandwidth | 6,528 GB/s | 8,601 GB/s |
| On-chip SRAM (Vmem) | 128 MB | 384 MB |
| ICI bandwidth | 19.2 Tb/s/chip (2× prior gen) | 19.2 Tb/s/chip |

**Consistency check**: 8i's 8,601 GB/s matches this repo's existing 8.6
TB/s exactly — confirms that number was already right, not a coincidence
of rounding.

**New: TPU 8t's own ridge point** = 12.6 PFLOPS / 6.528 TB/s ≈ **1,930
FLOPs/byte** — noticeably right of 8i's ≈1,174 FLOPs/byte. Training needs
a higher compute-to-memory ratio to stay compute-bound than rollout does
on 8i. Worth checking explicitly once Phase 2 derives training's own
FLOPs/bytes rather than assuming compute-bound by analogy to prefill.

**New: ICI bandwidth (19.2 Tb/s/chip = 2.4 TB/s/chip)** is stated for
both chips, doubled vs. the prior generation — a real, sourced number for
Phase 3's cross-chip FP4(8t)→FP4(8i) weight-sync derivation, rather than
guessing a network figure. Not yet clear whether weight sync between 8t
and 8i pods routes over ICI directly (same fabric) or over a separate
DCN/RDMA path (Miles used dedicated NCCL/RDMA channels on GPUs, a
different topology) — flagged for Phase 3, don't assume ICI applies
without checking whether 8t and 8i pods are even on the same fabric.
- **P=1,024 (prompt length) is this project's own assumption**, not
  sourced from R1 or any primary source — flagged, revisit if a real
  number surfaces.
- **Rollout Routing Replay / MoE-routing-drift** (from Miles) — real,
  workload-specific mechanism not in the original spec's reading list.
  Decide explicitly in/out during Phase 2 or 3, don't let it silently fall
  through the cracks.
- **N≈640 realistic decode batch size** carried forward from disagg's own
  HBM-capacity grounding — same chip, same model, so the reuse is
  legitimate, but confirm it still holds if Phase 2/3 changes any
  deployment assumptions (e.g. a different EP-group size).
