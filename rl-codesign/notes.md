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

## Phase 3 — Colocated vs. disaggregated architectures (in progress)

**Headline, survives every check run this phase — read this before the
detail below**: rollout is most likely more of a bottleneck than
training; the magnitude of that dominance depends on the length of
rollouts (R), your chip count, and your sharding scheme. "Most likely,"
not "always" — one real exception exists (EP=160-only-no-TP's R≈10⁶
asymptotic crossover, see below), everything else tested gave ratio >1.

### Rollout:train wall-clock split, per full RL step

Shared input both sub-architectures need (colocated's wall-clock split,
disaggregated's chip-ratio methodology) — computed once, first, before
picking either side.

**Rollout wall-clock (full step, not single-request)**: Phase 1's
per-request numbers scaled to the real batch. Prefill: 32 distinct
prompts (not shared via K-sharing, which only applies *within* a
prompt's K=16 completions), batched — still compute-bound (batch≫B≈3
threshold), wall-clock scales linearly with total FLOPs so 32×4.62ms ≈
147.9 ms. Decode: all 512 completions run concurrently in one wave (≤
N=640 capacity), so the full-step decode wall-clock **equals** the
single-request number already derived — no rescaling needed.

| | R=8,192 | R=65,536 |
|---|---|---|
| Prefill (32 prompts) | 147.9 ms | 147.9 ms |
| Decode | 27.67 s | 523.5 s |
| **Rollout wall-clock/step** | **≈27.82 s** | **≈523.65 s** |
| Prefill's share | 0.53% | 0.028% |

**Training wall-clock (full step)**: no formula to reuse wholesale —
training's forward pass is **prefill-shaped over the full sequence**
(same mechanism Decision 1 already established for the reference
model's log-prob pass), so Phase 1's `Prefill/layer(seq)` formula is
reused directly, evaluated at **seq = P+R** (not just P — training
needs gradients through the whole generated response), ×60 layers, ×512
(real batch), then ×4 (Phase 2's forced-recomputation multiplier, not a
generic 3×/8ND estimate).

```
forward_flops_per_layer(S) = 675,938,304·S + 81,920·S²,  S = P+R
training_step_flops = 4 × 60 × 512 × forward_flops_per_layer(P+R)
wall_clock = training_step_flops / (160 devices × 12.6 PFLOPS/chip)
```

| | R=8,192 (S=9,216) | R=65,536 (S=66,560) |
|---|---|---|
| Forward FLOPs/seq (×60 layers) | 7.912×10¹⁴ | 2.447×10¹⁶ |
| Forward FLOPs, batch=512 | 4.051×10¹⁷ | 1.253×10¹⁹ |
| Training step FLOPs (×4) | 1.6205 EFLOPs | 50.125 EFLOPs |
| **Training wall-clock/step** | **≈0.804 s** | **≈24.86 s** |

**Compute-bound verified, not assumed** (same discipline as Phase 1's
prefill check): AI = FLOPs/device ÷ bytes/device (weight reads,
~3× per step for fwd+bwd+recompute-forward) comes out to 637,573
FLOPs/byte (R=8,192) and 19,721,679 FLOPs/byte (R=65,536) — both far
above TPU 8t's ≈1,930 ridge point. Training is solidly compute-bound at
this batch/sequence scale, consistent with Phase 2's activation-
recomputation finding (recomputation trading memory capacity for extra
FLOPs is itself evidence compute is the cheap resource here).

### E2E per RL step and the rollout:train ratio

| | R=8,192 | R=65,536 |
|---|---|---|
| Rollout | 27.82 s | 523.65 s |
| Training | 0.80 s | 24.86 s |
| **Ratio (rollout:train)** | **≈34.6 : 1** | **≈21.1 : 1** |

**Rollout dominates wall-clock at both ends — genuinely non-obvious
given training carries the 4× recomputation penalty and rollout is
"only" memory-bound**: rollout is R *sequential* memory-bound decode
steps; training is one large parallel compute-bound pass spread across
160 chips at once — the parallelism gap outweighs the FLOPs-multiplier
penalty.

**Does the ratio ever cross over as R grows? No — it asymptotes to
≈16.1×, never reaching parity.** Both wall-clocks pick up a
quadratic-in-R term at large R, from the *same* underlying mechanism
(full attention/cache read over a context that grows with R): rollout's
KV-cache-read bytes (`11,520·R` term inside the bytes/layer/device
formula, verified by reconstructing it from Phase 1's own published
table values) and training's `81,920·S²` attention-score compute term
(S≈R at large R). Since both are O(R²), their ratio converges to the
**ratio of the two quadratic coefficients**, not to 1:

```
decode asymptotic coefficient (wall-clock/R²):  8.037e-08
train  asymptotic coefficient (wall-clock/R²):  4.993e-09
ratio → 16.096
```

Verified numerically out to R=10¹⁰ — ratio decreases monotonically from
34.6 (R=8,192) through 21.1 (R=65,536) and flattens at 16.10 by
R≈10⁷, never dropping further. The 34.6→21.1→16.1 decrease is the
non-quadratic terms (rollout's fixed per-step weight-byte floor,
training's linear-in-S term) becoming relatively negligible as R grows
— not a sign of convergence toward crossover.

**Caveat, since revised below**: 16.1× is a property of *this specific,
mismatched device split* (160 chips for training, EP=8/8-device rollout
group) — it answers a **disaggregated**-flavored question (each phase
gets its own independently, capacity-derived pool size — real for
disaggregated systems, which do split unevenly on purpose), not a
colocated one (single shared pool, same chip count both modes). Treating
16.1× as "the" rollout:train dominance number would be wrong — see
below.

### Matched-chip (colocated) comparison — the 16.1× number was the wrong question

Colocated reshards the *same* pool between rollout- and training-mode,
so both phases must run at the same N. Training's ≥160-chip capacity
floor (Phase 2) forces N≥160 for the whole pool — meaning rollout also
gets up to 160 chips during its phase, not just the 8 it was
independently sized for. Recomputing rollout's decode wall-clock at
EP=160 (bytes/layer/device terms — cache, FFN — scale as 8/EP; the
attention-weight term, 108.17 MB, is *replicated* across the EP group
and doesn't shrink with EP width at all, same structural issue that
forced TP into training's own solution):

| | R=8,192 | R=65,536 |
|---|---|---|
| Rollout @ EP=160 (prefill+decode) | 7.41 s | 73.31 s |
| Training @ 160 devices | 0.80 s | 24.86 s |
| **Ratio** | **9.2 : 1** | **2.95 : 1** |

Down from 34.6:1/21.1:1 — matching chip counts closes most of the gap.
But EP-widening alone only gets 3.81×/7.15× speedup from a 20× chip
increase (8→160), not 20× — bounded by the replicated-attention floor
(≈6.18s/49.46s as EP→∞). This is a genuine crossover-enabling change:
at EP=160, decode's own asymptotic (R→∞) quadratic coefficient
(4.02e-9) is now *smaller* than training's fixed one (4.99e-9) — unlike
the mismatched case, there **is** a real crossover here, at **R≈1,000,000**
(bisected numerically). Heavily caveated: ~15× beyond this project's own
R=65,536 anchor, assumes rollout gets no TP (only wider EP), and assumes
N=640 concurrent-request capacity holds unchanged at EP=160 — not
load-bearing for any real conclusion at this project's actual R range,
but a real mathematical consequence worth recording.

### Sensitivity to rollout's own parallelism layout — the magnitude is not a single number

EP=160-alone isn't the only rollout layout available at 160 chips.
Tested against training's own TP=4×EP=40 split (mirroring training's
exact layout) and a scaled-up TP=4×EP=160 (640 chips, both sides). Byte
scaling rule (matches Phase 2's own confirmed pattern): **TP shards
attention weights only** (independent of EP), **EP shards FFN/expert
weights only** (independent of TP) — cache term's TP-sensitivity is
unconfirmed (MLA's cache is attention-adjacent, plausibly TP-shardable,
but no source pins this down), so both conservative (cache: EP-only) and
optimistic (cache also shards with TP) variants computed.

| Config | Devices | Ratio @ R=8,192 | Ratio @ R=65,536 |
|---|---|---|---|
| EP=160 only | 160 | 9.21:1 | 2.95:1 |
| TP=4×EP=40 (matches training) | 160 | 7.45:1 | 4.32:1 |
| TP=4×EP=40, optimistic (cache shards w/ TP) | 160 | 6.20:1 | 2.17:1 |
| TP=4×EP=160, both sides | 640 (each) | 13.77:1 | 5.83:1 |

**Three findings, more load-bearing than any single ratio**:

1. **Rollout dominates in every configuration tested — ratio never below
   ~2:1.** Direction is robust; magnitude isn't (2.2×–13.8× range).
2. **Optimal rollout layout is R-dependent, not fixed**: at R=8,192,
   TP=4×EP=40 beats EP=160-alone (attention's fixed floor is a bigger
   fraction of a smaller total, worth sharding via TP); at R=65,536 it
   flips (cache dominates so heavily that EP-width for spreading it
   matters more than shrinking the now-relatively-small attention term).
   No single best rollout parallelism strategy holds across this
   project's own two R anchors.
3. **Adding proportionally more chips to both sides doesn't preserve the
   ratio — it widens it in training's favor** (7.45→13.77 at R=8,192
   going 160→640 devices, same TP:EP ratio). Training is ideal
   compute-bound strong-scaling (no floor, clean 4× speedup from 4×
   devices). Rollout has structural floors (whatever EP/TP don't reach)
   that cap how much it benefits from added parallelism. "Just add more
   hardware to both" does not equalize rollout and training — it makes
   training pull further ahead.

**Working conclusion for Phase 3's colocated wall-clock split**: rollout
dominates training's wall-clock by roughly 2×–14× depending on
parallelism layout and R, direction robust, magnitude genuinely
sensitive to config — not a single clean number the way Phase 1/2's
findings were. The disaggregated-style 16.1× floor (mismatched pools,
EP=8 vs 160) remains valid as an input to disagg's own chip-ratio
methodology (each phase's independently-sized pool), just not as "the"
colocated dominance number.

**Not modeled, flagged**: K-way prefix sharing (Phase 1's 93.75% prefill
saving) is *not* reused for training's forward — each of the 512
completions gets its own independent backward graph; sharing prefix
activations across a K-group during backward is a real technique but
adds real complexity (gradient accumulation through a shared subgraph)
not derived here. Also assumes **DP=1** (160 chips = the entire step,
no additional data-parallel replicas) — Phase 2 only established the
per-replica footprint, DP degree was never decided.

### Colocated on a single physical chip — the 8t/8i heterogeneity finding

**Colocated fundamentally needs one physical chip type, not just matched
parallelism degrees.** Matching TP/EP layout between modes (e.g. same
TP=4×EP=40 for both) removes the *weight-reshard* cost — devices sit at
the same coordinate holding the same shard in both modes, no all-gather
needed. But it doesn't remove the deeper problem: training is modeled on
TPU 8t, rollout on TPU 8i — **physically different silicon** (Decision
2's revision), not two modes of one chip. HybridFlow's colocated pool
works because GPUs are homogeneous; TPU 8t/8i isn't. So colocation
forces a choice of *which* chip hosts both roles, and whichever role
doesn't get its native chip pays a real, recurring (every step, not
one-time) performance tax:

- All on 8t: rollout pays (8t's 6.528 TB/s vs 8i's 8.6 TB/s bandwidth —
  hurts decode, the bandwidth-bound term).
- All on 8i: training pays (8i's 10.1 PFLOPS vs 8t's 12.6 PFLOPS peak —
  hurts the compute-bound forward/backward).

**Two options going forward, both real**: (a) a genuinely homogeneous
chip capable of both roles (e.g. Blackwell) sidesteps this tradeoff by
design — flagged as a real open thread, not pursued (would require
sourcing new hardware specs and re-deriving this project's formulas on
it, disproportionate scope for one branch of the comparison, and this
project has stayed TPU-8i/8t-anchored throughout). (b) Pick one TPU chip
and caveat the tax — **chosen: 8i**, because rollout dominates step
wall-clock in every config tested, so keeping the dominant term native
and taxing the minority term (training) minimizes total damage,
vs. taxing the dominant term by choosing 8t.

**Capacity re-check on 8i (288 GB/chip vs 8t's 216 GB) — training's
device-count floor shrinks.** Same search as Phase 2's TP×EP grid, new
budget:

| TP | EP | devices | GB/dev | headroom |
|---|---|---|---|---|
| 1 | 40 | 40 | 285.4 | 2.6 GB — **rejected, knife-edge, no room for activations** |
| 2 | 32 | 64 | 234.5 | 53.5 GB |
| **2** | **40** | **80** | **211.6** | **76.4 GB — chosen, more margin than 8t's own 41GB** |

**Colocated on 8i, TP=2×EP=40=80 devices, same layout both modes (no
reshard, no cross-chip tax on the dominant term)**:

| | R=8,192 | R=65,536 |
|---|---|---|
| Rollout | 7.537 s | 119.671 s |
| Training (2.50× tax vs. native 8t/160dev — exactly 2× fewer chips × 1.248× lower peak) | 2.006 s | 62.035 s |
| **Ratio** | **3.76 : 1** | **1.93 : 1** |
| Rollout's share of step | 79.0% | 65.9% |

Tightest colocated picture derived — down from the original 34.6×/21.1×
mismatched estimate. Asymptotic check for this specific config: quad
coefficient ratio ≈1.29 (still >1) — **no crossover** this time, unlike
the EP=160-only-no-TP case's R≈10⁶ crossover. Different parallelism
choice, different long-run behavior — reinforces that neither the
16.1× floor nor any single asymptotic behavior is "the" answer; both are
config-specific.

### Pool-size ambiguity — N is a free variable, not derivable in isolation

Training's capacity floor (≥80 devices on 8i, ≥160 on 8t) is a
**minimum**, not a mandate — nothing stops a larger colocated pool, and
since rollout dominates wall-clock, more chips seem like they should
help rollout more. **They don't, in relative terms**: swept N from 80 to
2,560 (various TP/EP splits, training on native 8t formula scaled to
each N) — total wall-clock (rollout+train) monotonically improves with
N (181.7s→20.1s at R=65,536), but the **ratio gets worse, not better**
(3.76×→23.26× at R=8,192 across the same sweep). Training is ideal
compute-bound strong-scaling (no floor); rollout has structural floors
(whichever term TP/EP don't reach). So "minimize the gap" and "minimize
total latency" are **different objectives that point in opposite
directions** as N grows — there is no principled single N without either
a hardware-budget constraint (never modeled in this project) or an
external anchor. **Resolution**: match colocated's N to disaggregated's
own total chip budget (below) rather than picking N in isolation.

### Disaggregated chip-ratio — throughput-balance methodology

**Real derivation, distinct from the earlier mismatched (8-vs-160)
wall-clock comparison** — that number answered "what's the ratio at
independently-capacity-chosen pool sizes," not "what ratio balances
throughput." Disaggregated runs two *concurrent* pools (rollout
generates step i+1 while training consumes step i), each on its own
**native** chip — no cross-chip tax at all, a real structural advantage
over colocated. Balance condition: `rollout_time(N_r) = train_time(N_t)`
(train on native 8t, 12.6 PFLOPS) — so neither pool idles waiting on the
other.

**Starting from N_r=8** (the one real, disagg-sourced anchor, not
arbitrary): balancing N_t comes out *smaller* than N_r —

| R | rollout_time | N_t needed | ratio N_r:N_t |
|---|---|---|---|
| 8,192 | 27.82 s | 4.62 → 5 | 1.60:1 |
| 65,536 | 523.65 s | 7.60 → 8 | 1.00:1 |

Counter to naive intuition (bigger model → more training chips) —
rollout is so much slower per-chip that a tiny training pool keeps pace.

**Sweep across N_r (pure EP, TP=1) shows the ratio is not fixed — it
climbs with N_r**:

| N_r | ratio @ R=8,192 | ratio @ R=65,536 |
|---|---|---|
| 8 | 1.73 | 1.05 |
| 32 | 2.91 | 1.35 |
| 40 | 3.31 | 1.45 |
| 80 | 5.27 | 1.95 |
| 160 | 9.21 | 2.95 |

**Real-world sanity check**: AReaL's own reported ratio is ~75%/25%
(3:1) inference:training on a 512-GPU H800 cluster. Our derived ratio
**crosses 3:1 right around N_r≈32–40 at R=8,192** — close convergence at
a plausible scale, resolving what initially looked like a real
discrepancy at N_r=8. At R=65,536, reaching the same 3:1 needs a much
larger N_r (still only 2.95× at N_r=160) — **a new, real finding: longer
responses need proportionally far more rollout chips to sustain the same
inference:training balance**, because rollout's wall-clock grows
quadratically with R while the *balance point* itself shifts outward.
Best explanation for AReaL's own lower N_r-scale match: AReaL's models
(1.5B–32B, Phase 0 reading) are far smaller than this project's 236B
MoE workload, and likely used shorter responses than R1's 65,536-token
late-training cap — both push toward AReaL needing comparatively fewer
training chips. Directionally explained, not quantitatively verified
(AReaL's own exact model size within their range and response-length
distribution aren't sourced here).

**Robust vs. config-dependent findings, worth keeping separate**:

- **Robust** (holds in *every* tested config, independent of TP/EP
  split): ratio **shrinks as R grows**, at fixed N_r/N. Both rollout's
  bytes formula and training's FLOPs formula carry a quadratic-in-R term
  from the same cause (attention/cache over growing context) — training's
  own quadratic term keeps closer relative pace, narrowing the gap. True
  for any fixed config, so it doesn't depend on resolving the TP-scaling
  question below.
- **Config-dependent, weaker** (verified only for TP-held-fixed sweeps):
  ratio **widens as N grows**. This one's real mechanism is genuinely
  more efficient training (chip-seconds constant, `FLOPs/peak`,
  independent of N) vs. genuinely less efficient rollout (chip-seconds
  `N×ro(N)` grows with N because rollout's speedup is sub-linear, capped
  by whichever term isn't sharded). **Walked back a stronger claim**:
  earlier framed as "training has no structural bottleneck ever,
  rollout's efficiency decays toward a floor" — that's only established
  for the *specific configs tested* (TP held fixed while EP widened).
  Untested: whether scaling TP *proportionally* alongside EP removes
  rollout's floor too. **Deliberately not chased** — even a clean
  idealized-model answer wouldn't be trustworthy, since (a) this project
  has never modeled TP/EP communication overhead (all-reduce/all-to-all)
  anywhere, and high-TP is exactly where that cost would start to bind
  in reality, and (b) DeepSeek-V2's real attention head count (the actual
  ceiling on TP width) isn't sourced in this project — same open V2/V3
  mismatch flagged since Phase 2. Matches this project's own "cut it and
  flag it" discipline (selective recomputation, N=640 sensitivity, exact
  TP×EP activation-memory rederivation were all similarly flagged and not
  chased).

**Correction, caught while setting up item 4 — the balance-point N_t
values above are infeasible.** The sweep solved `train_time(N_t) =
rollout_time(N_r)` purely for wall-clock equality, without checking it
against Phase 2's own training capacity floor (**≥160 devices on native
8t**, TP=4×EP=40, just to *fit* weights+gradients+optimizer state — a
hard requirement, not a speed consideration). The balance-point N_t
values above (as low as ≈5) don't have anywhere near enough combined HBM
to hold the model; they solve the equation correctly but aren't
buildable systems.

**Corrected picture**: since disaggregated keeps training on its native
chip (8t, avoiding the cross-chip tax — the whole point of
disaggregating), N_t is **fixed at 160**, not a free variable. At
N_t=160, train_time is fixed too: **0.80 s (R=8,192) / 24.86 s
(R=65,536)** (Phase 2/3's own numbers). Compare against rollout's own
*best-possible* wall-clock — its structural floor as N_r→∞ (EP-only,
TP=1, matching this sweep's own convention): **≈6.18 s / ≈49.46 s**.
Training's fixed floor-time is smaller than rollout's floor in both
cases (0.80 < 6.18, 24.86 < 49.46) — **training can never be the
bottleneck in disaggregated, at any N_r, at either R anchor**, once the
real capacity constraint is respected. The ratios tabulated above are
still useful as the *hypothetical, capacity-unconstrained* balance
point, but not as a buildable chip-ratio recommendation.

**Practical consequence for item 4**: N_t=160 is a fixed cost;
N_r is the only real free variable. Disaggregated's real per-step-
equivalent cost, with sync verified hideable within a single pipeline
cycle (checked explicitly — sync's 2.75 s stays under 45% of cycle time
even at rollout's own structural floor, comfortably inside the 1-step-
staleness slack every real system in this space already uses), is:

```
disagg_cost(N_r) = max(rollout_time(N_r), train_time(160))
disagg_total_chips(N_r) = 160 + N_r
```

**Session's working synthesis, stated at the right confidence level**:
rollout:training chip ratio (disaggregated) is >1 in every configuration
tested — not a proven universal law, but true across everything checked
— and it **shrinks as R grows** (robust finding, holds regardless of
parallelism split) while it **widens as pool size grows** (real, but
tied to the untested/unmodeled TP-scaling-ceiling question, so held at
lower confidence). Colocated's own number (rollout dominates 3.76×–9.2×
depending on config, tightest at TP=2×EP=40/80-device/8i) is a separate,
now well-triangulated finding from a different mechanism (the 8t/8i
physical chip split forcing a cross-role tax), not directly comparable
to the disaggregated ratio without matching total chip budgets first —
still an open synthesis step (see `handoff.md`).

### Weight-broadcast topology and sync-boundary transfer cost (disaggregated) — items 1+2

**The question, precisely**: when training finishes a step, its updated
weights (sharded across N_t devices, training's own parallelism layout)
need to reach every rollout worker (sharded across N_r devices, rollout's
own, generally different, layout) before rollout can generate with the
new policy. Two sub-questions bundle together: what topology carries
this traffic (point-to-point / broadcast / tree), and what does it cost
in wall-clock time.

**Real search pass, findings — 8t and 8i do not share a fabric.**
Checked the primary Google Cloud TPU 8t/8i deep-dive directly (re-fetched
twice, see methodological note below) plus independent corroboration
(ServeTheHome writeup, a Jeff Dean tweet on the launch). Confirmed:

- **TPU 8t**: retains **3D torus** topology, scales to 9,600 chips/
  superpod, own **Virgo** scale-out fabric (east-west, TPU-to-TPU within
  the 8t fleet — 47 Pb/s bisection bandwidth across 134,000+ chips).
- **TPU 8i**: **Boardfly**, hierarchical — **4-chip Building Block** (full
  mesh) → **8 BBs = 32-chip Group** (full mesh via copper) → **36 groups
  = 1,152-chip Pod** (via Optical Circuit Switches), 7-hop max diameter
  (down from 16 in a 3D torus at the same 1,024-chip scale). Internal
  consistency check: 36 × 32 = 1,152, matches the stated pod total —
  this is the check that caught the error below.
- **No shared fabric or topology between 8t and 8i pods** — not
  documented anywhere in the primary source. The only cross-pod hint:
  8t racks reach "compute and storage services" via **Jupiter**, Google's
  general-purpose north-south DCN, not ICI. Real, sourced per-host
  bandwidth for the current Jupiter generation: **400 Gb/s/host**
  (standard modern DCN NIC speed, from Google's own Jupiter-network
  evolution writeup — not TPU-8t/8i-specific, but the right order of
  magnitude, and the user's own "probably just standard ethernet"
  intuition was right).
- **Practical conclusion**: the 19.2 Tb/s/chip ICI number flagged
  earlier as a "candidate" for this transfer does **not** apply — ICI is
  pod-internal only (different topologies per chip type, not just
  different pods). The real cross-pool hop is the much slower Jupiter
  DCN link.

**Methodological correction, worth recording explicitly**: an earlier
pass of this derivation used "288 chips" for Boardfly's Group size,
sourced from a WebFetch summary — this was wrong, a fetch-tool
conflation with TPU 8i's real 288 GB HBM capacity (a different number,
already used elsewhere in this project), not an actual topology figure.
Caught by the user, re-verified via direct re-fetch + independent
cross-check (36 groups × chips/group must equal the stated 1,152-chip
pod total — 32 satisfies this, 288 does not by a wide margin). Real
Group size is **32 chips** (8 BBs × 4 chips/BB), not 288. Lesson worth
keeping: sanity-check any tool-summarized hierarchical count against a
stated total before using it, especially when a suspiciously-similar
number already exists elsewhere in the project.

**Topology model, three legs:**

1. **Training-side fan-in**: training's weights are sharded across 160
   devices (TP=4×EP=40, Phase 2). For one egress chip to hold the full
   118 GB payload before it can cross the pool boundary, the other 159
   devices' shards (118 − 5.295 = **112.7 GB**) must be gathered onto it
   first, over 8t's own fabric (3D torus/Virgo, not Boardfly — that's
   8i-only). Bound by 8t's 19.2 Tb/s/chip ICI (best-case, no hop-count
   model — unlike Boardfly, no sourced hierarchical breakdown exists for
   8t at the 160-chip scale, flagged as a real gap, not chased further
   since it's already a small term).
2. **Cross-pool hop**: single Jupiter DCN link, 400 Gb/s.
3. **Rollout-side fan-out**: Boardfly, from the one receiving chip to the
   rest of the N_r-device rollout pool. Hop count depends on whether N_r
   fits inside one 32-chip Group: **N_r=8 and N_r=32 fit (≈2 hops)**;
   **N_r=40, 80, 160 — including the colocated-on-8i config (N_r=80) —
   all exceed one Group** and need Pod-level OCS routing between groups,
   not precisely characterized for partial-pod subsets. Conservative,
   sourced fallback: the pod's own **7-hop max diameter**.

**Transfer cost, single-link baseline** (weight payload 118 GB = 236B
params × 0.5 B/param FP4, already sourced):

| Leg | Bandwidth | N_r=80 (7 hops) | N_r=8/32 (2 hops) |
|---|---|---|---|
| Fan-in (8t, 112.7 GB) | 19.2 Tb/s/chip | 0.047 s | 0.047 s |
| Cross-pool (118 GB) | 400 Gb/s | 2.36 s | 2.36 s |
| Fan-out (8i, 118 GB) | 19.2 Tb/s/chip | 0.344 s | 0.098 s |
| **Total** | | **≈2.75 s** | **≈2.46 s** |

**Headline finding**: compared against Phase 3's own per-step wall-clocks
(training 0.80 s / 24.86 s, rollout 27.82 s / 523.65 s at R=8,192/65,536):

| Sync cost (2.75 s) as % of... | R=8,192 | R=65,536 |
|---|---|---|
| Training step | **≈344%** | ≈11% |
| Rollout step | ≈9.9% | ≈0.5% |

**At R=8,192, the weight-sync step alone costs over 3× the entire
training step.** Not a rounding-error tax on disaggregated's design —
at short response lengths, sync can be a bigger bottleneck than training
itself. At R=65,536 it's comparatively negligible, since rollout so
thoroughly dominates everything by then. Real, load-bearing addition to
disaggregated's total cost model, not a footnote — must be included in
item 4's synthesis, not assumed away.

**Architecture is a real design choice here, not a fixed cost — three
points on a tradeoff curve, not one number:**

| Architecture | Cross-pool time | What it costs |
|---|---|---|
| **1 link, gather→send→broadcast** (baseline above) | 2.36 s | Simple, one physical link. Serializes all 118 GB through one pipe. |
| **N_t=160 parallel links, matched layout** | 118GB/160 ≈ 0.106 s | Needs 160 dedicated cross-pool links (real infra cost) *and* rollout's shard layout must exactly match training's (TP=4×EP=40) — the same "same-layout, no-reshard" trick already used for colocated-on-8i, just applied cross-pool. |
| **N_t parallel links, mismatched layout** | ~0.106 s transfer + **unmodeled reshard cost** | Same infra cost, but rollout must reshuffle 160 incoming pieces into its own layout. Real anchor for how bad this gets: HybridFlow's own measured reshard cost for a much smaller 70B *dense* model was up to 36.4% of iteration time naively, ~11.7 s average even with their optimized engine, up to 78.2 s worst case — for this project's 236B MoE model, plausibly *worse* than the 2.75 s single-link baseline, not better. |

**The catch on the fast option**: matched-layout parallel transfer
requires N_r=N_t=160 — but disaggregated's whole point (Phase 3's own
pool-size-ambiguity finding, above) is that rollout and training get
*independently* right-sized pools (e.g. N_r≈32–40 for the AReaL-matched
throughput balance, not 160). Forcing N_r=160 to get the cheap transfer
collides directly with that. No free lunch: every route to "faster"
either costs more physical links, or gives up independent pool-sizing,
or reintroduces resharding.

**Working decision for Phase 3's synthesis**: carry the single-link
**≈2.75 s** as the default sync cost — needs no extra infra assumptions
and no reshard modeling. Flag the matched-layout-parallel option as a
real, cheaper alternative *contingent on* accepting N_r=N_t, left for
item 4's synthesis to weigh rather than resolved here (same treatment
the pool-size ambiguity itself already got).

### Colocated resharding cost — scoped down, not derived (item 3)

**Why this doesn't need a full derivation**: the strongest colocated
number already in hand (above, "Colocated on one physical chip") is the
**same-layout-on-8i** choice — TP=2×EP=40=80 devices, identical
parallelism grid for both rollout- and training-mode. Devices sit at the
same (TP, EP) coordinate in both modes, so switching modes means
switching *what compute runs locally*, not *redistributing which shard
lives where*. That's a genuinely zero-reshard transition, not just a
cheap one — HybridFlow's own "zero memory redundancy" claim (Phase 0,
`notes.md` top) turns out to hold in both the memory *and* time sense
for this specific config, precisely because there's nothing to reshard.
This is already the number carried forward into item 4 — no reshard
cost needs to be added on top of the 3.76×/1.93× colocated ratios.

**The alternative, flagged but deliberately not chased**: a
*different*-layout colocated setup (e.g. wide-EP-only for rollout mode,
TP+EP for training mode — the layouts Phase 3's own sensitivity sweep
showed were R-dependently optimal) would need a real HybridFlow-style
reshard step between modes, paid every time the pool switches roles.
Real anchor for that cost, already sourced (Phase 0): HybridFlow's own
measured numbers on a 70B *dense* model — up to **36.4%** of iteration
time naively, **~11.7 s average** even with their optimized resharding
engine, up to **78.2 s** worst case. For this project's 236B *MoE*
model, no scaled version derived — plausibly worse (more parameters,
more shards to redistribute) but not quantified.

**Why not derive it further**: the same-layout variant is already the
better answer in hand (avoids the cost entirely rather than minimizing
it), and deriving the different-layout reshard cost precisely wouldn't
change the working colocated number carried into item 4 — it would only
produce a second, deprioritized data point for a config already known to
be worse. Matches this project's own repeated scope discipline
(selective recomputation, N=640 sensitivity, exact TP×EP activation
rederivation — all flagged and not chased for the same reason: the
answer already in hand is sufficient for the checkpoint).

### Item 4: the colocated-vs-disaggregated comparison

**Rollout's optimal TP/EP split is a real 2D surface (N_r, R), not a
fixed choice — correction to the earlier "R-dependent" finding.**
Working the tradeoff out algebraically: `total(TP) = A/TP + C(R)·TP/N_r`,
where A=108.17 MB is attention's replicated-weight term (shrinks with
TP, independent of N_r) and C(R) = 2064.4 + 640·(1024+R/2)·288/1e6 MB
bundles FFN's fixed coefficient with cache's R-dependent one (both
shrink with EP = N_r/TP, so every device TP takes is a device EP loses).
This has a real minimum, not a monotonic slope: **TP\* = √(A·N_r/C(R))**.
Explains the earlier empirical finding cleanly — TP\* grows with N_r
(more EP capacity to spare, worth paying into TP) and shrinks with R
(cache's growing weight in C(R) makes EP more precious at long
sequences). What was previously read as "optimal layout is R-dependent"
is really **one continuous surface in (N_r, R), not two separate
observations** — the earlier sensitivity table was just a few scattered
points on it.

Applied to the three disaggregated chip-budget points (R=8,192):
TP\*≈0.54 at N_r=8, TP\*≈1.2 at N_r=40 (both round to **TP=1**, pure EP
still wins) — but TP\*≈2.4 at N_r=160, where **TP=2×EP=80 beats pure
EP=160** (91.7 MB/layer/device vs. 126.97, a real ~28% improvement).

**Correction to the "structural floor" claim from the pool-size-
ambiguity work above**: the ≈6.18 s/≈49.46 s (R=8,192/65,536) floor as
EP→∞ was computed under a **fixed TP=1** assumption — real, but not an
absolute floor. Once TP is also allowed to scale optimally with N_r
(as just derived), attention gets sharded away too, and rollout time
falls as **O(1/√N_r) without a floor**, not asymptoting to a fixed
value. The 6.18s/49.46s number should be read as "the floor *given*
TP=1 is fixed," not "the floor, full stop."

**N_t=160 is optimal, not just a minimum — a real consequence of
rollout dominating, not a coincidence.** Training is compute-bound with
clean strong-scaling (no structural floor the way rollout has
memory-bound terms) — more N_t directly buys more training speed in
isolation. But `disagg_cost(N_r) = max(rollout_time(N_r),
train_time(N_t))`, and `train_time(160)` (0.80 s / 24.86 s) already
sits below rollout's achievable range at every N_r this project tests —
training was never the binding constraint, so N_t beyond 160 would only
add idle, wasted chips with zero effect on `disagg_cost`. Given the
newly-uncapped rollout scaling above, there's a real (if extreme)
crossover where this stops holding: rough back-of-envelope, solving
`rollout_time(N_r) = train_time(160)` under optimal-TP scaling gives
**N_r ≈ 10,000** at R=8,192 — ~60× the largest N_r this project tests
(640). Real, flagged, not chased — nowhere near any config actually
used here, same discipline as the earlier TP-scaling-ceiling flag.

**Corrected disaggregated rollout, using optimal TP per N_r:**

| N_r | TP\* | Config | R=8,192 | R=65,536 |
|---|---|---|---|---|
| 8 | 0.54→1 | EP=8 (TP=1) | 27.82 s (unchanged) | 523.65 s (unchanged) |
| 40 | 1.2→1 | EP=40 (TP=1) | 10.63 s (unchanged) | 144.42 s (unchanged) |
| 160 | 2.4→2 | TP=2×EP=80 | **5.39 s** (was 7.41 s) | **72.27 s** (was 73.31 s) |

**The chip-budget-matched comparison** (disaggregated total = 160+N_r;
colocated matched to the same total, same-layout-on-8i, TP=2×EP=N/2 —
colocated's TP=2 stays fixed here since it's driven by *training's*
capacity constraint at the reference N=80 config, not re-optimized for
rollout the way disaggregated's is; whether TP=2 is still strictly
required for capacity at larger N is flagged, not re-derived):

| Total chips | Disaggregated `max(rollout,train)` | Colocated (serial) | Colocated wins by |
|---|---|---|---|
| 168 (N_r=8) | 27.82 s / 523.65 s | 6.24 s / 99.57 s | **4.5× / 5.3×** |
| 200 (N_r=40) | 10.63 s / 144.42 s | 5.76 s / 87.15 s | **1.85× / 1.66×** |
| 320 (N_r=160) | 5.39 s / 72.27 s | 4.82 s / 64.08 s | **1.12× / 1.13×** |

**Headline finding, properly conditioned (see Decision 3 below before
treating this as a takeaway)**: colocated wins at every chip budget
tested, at both R anchors — margin shrinks sharply with scale
(4.5×→1.85×→1.12×), converging toward near-parity, but never actually
crossing over in the tested range. **This is the opposite of RLinf's
claimed pattern** (disaggregation wastes compute small-scale, colocation
stalls large-scale) — a real result for *this model's* own very lopsided
rollout:train FLOPs ratio (rollout dominates by 2×–35× per the earlier
wall-clock-split findings), not a universal law. Mechanism: disaggregated
pays a fixed 160-device training tax that sits almost entirely idle
(train finishes in under a second while rollout is still running), a
real structural waste that shrinks only as total budget grows large
enough for the fixed tax to become a small fraction of it; colocated
never pays that tax because the same chips serve both roles.

**This finding is conditional on fully on-device optimizer state — see
Decision 3 immediately below.** The 160-device training floor driving
the entire mechanism above is ~70%-composed of optimizer state bytes;
offloading that (as this project's own anchor workload, R1, actually
does) would shrink the floor and directly undercut this result. Read the
headline as "colocated wins, given neither architecture offloads
optimizer state" — not as an unconditional recommendation.

**What's not yet chased, flagged for completeness**:
1. Whether colocated's own TP should also float with N (rather than
   staying pinned at TP=2) — if it can, colocated might get the same
   O(1/√N) rollout scaling disaggregated now has, which would keep
   colocated's asymptotic advantage rather than letting the margin
   converge toward parity. Not derived — would also require re-checking
   whether TP=2 remains *required* for training's capacity at larger N,
   or becomes optional.
2. The N_r≈10,000 training-becomes-bottleneck crossover, above.
3. RLinf's own claim may hold at different model sizes or rollout:train
   ratios than this project's specific 236B MoE / R∈{8192,65536}
   configuration — not tested, the finding above is scoped to this
   project's own numbers, not a general refutation.

**Decision 3: optimizer offloading — out of scope for quantitative
derivation, resolved directionally, not left as a dangling caveat.**
Made explicitly, Phase-0-style, rather than discovered reactively mid-
synthesis (worth noting the difference — this same question surfaced
through dialogue *after* the headline numbers above were already
computed, which is exactly the failure mode this decision is meant to
correct going forward: state scope boundaries up front, don't let them
accumulate as caveats bolted onto a finished conclusion).

*Why out of scope*: modeling this for real needs TPU 8t's host-device
interconnect bandwidth (this project has never touched host-side
anything — every number so far is accelerator-to-accelerator), CPU-side
Adam compute throughput (a different hardware category entirely), and
the prefetch/overlap strategy real offload systems use to hide that
latency (comparable in complexity to the weight-sync topology work
above). That's a fourth genuinely new research axis — `spec.md`'s own
scope note already flags that two new axes (training-side memory in
Phase 2, staleness in Phase 4) is already a full load, and warns
explicitly against a third.

*Why not silent, though*: the direction is derivable without the full
derivation. Optimizer state is ~70% of the per-device byte footprint
that sets N_t's capacity floor (12 of 16.5 bytes/param). Offloading it
to host memory would shrink that floor toward the throughput-balance
point — directly attacking the exact idle-capacity mechanism the
colocated-wins finding rests on. **The headline finding above should be
read as conditional on fully on-device optimizer state — an assumption
this project's own anchor workload (R1) is known to violate** (Phase 2
already noted R1's real training used CPU optimizer offloading). Not a
minor caveat: this is plausibly the single biggest lever that could
narrow or reverse the result, and it's a real, common production
technique, not a hypothetical edge case.

*Considered and rejected*: an "optimistic bookend" (assume offloading is
free, N_t shrinks to the throughput-balance point, see who wins) was
considered but rejected — it isn't actually a clean bound, since
colocated's own capacity floor (80 devices, also driven substantially by
on-device optimizer state) would plausibly shrink under the same
assumption, requiring new unmodeled assumptions about colocated's
response too. Two stacked idealizations don't bound the truth cleanly;
they produce a number that looks more precise than it is. The
directional statement above is the honest resolution.

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
