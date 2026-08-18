# Speculative Decoding Codesign — Notes

Working notes for project #9 (`speculative-decoding`) — tree-structured
speculative decoding, draft/verify/rollback. See `spec.md` for the full
phase plan; see `handoff.md` for current status and where to pick up
next. This file is a running log, appended to phase by phase — not yet a
polished writeup (that consolidation happens at project completion,
matching how `prefill_notes.md`/`rl_codesign_notes.md` were finished).

---

## Phase 0 — Setup

Three decisions are already recorded directly in `spec.md`, made before
this session, not re-litigated here: tree-structured over chain
(Decision 1), standalone draft model over self-speculation (Decision 2),
static tree shape with dynamic explicitly deferred to the build phase
(Decision 3).

### Reading list, synthesized

**Core mechanism (Leviathan et al. 2023; Chen et al. 2023, DeepMind):**
accept a draft token w.p. `min(1, p_target/p_draft)`; on reject, resample
from the residual `norm(max(0, p_target − p_draft))`. Exact distribution
preservation, not approximate — the accept branch contributes exactly
`min(p,q)` mass, the resample branch exactly the leftover `max(0,p−q)`,
summing to `p_target`. Chen et al.'s real numbers (Chinchilla-70B target
/ 4B draft): 2–2.5× speedup, pass@1 unchanged within noise (47.0% vs.
45.1%) — the exactness claim held empirically, not just on paper. This
is the mechanism every tree-structured method below extends, not
replaces.

**Draft & Verify** — real bf16/fp8/fp4/nf4 acceptance table, flat at
≈0.91 across all four (confirmed: 0.910/0.911/0.913/0.910). **Flagged as
not directly reusable for this project's Phase 5**: the setup is
self-speculative — one shared physical backbone with layers skipped for
drafting — so "quantizing the draft" there means quantizing the whole
shared backbone both draft and verify run through, correlating
draft/target error. Says nothing about independently-quantized,
differently-weighted standalone drafts (this project's Decision 2 setup)
without a fresh argument.

**SpecInfer** (Miao et al. 2023) — multiple draft models' outputs merged
into one token tree (shared prefixes collapse to shared nodes),
verified in one target pass via a "topology-aware causal mask." Raises
verification success ~57%→~97% vs. naive sequential rejection sampling
via tree-generalized speculative sampling per-branch. 1.5–3.5× speedup
depending on setup.

**Medusa** (Cai et al. 2024) — direct kernel precedent, confirmed from
source (`medusa/model/utils.py`). Mask = per-node ancestor-inclusion
bitmask (`mask[node, ancestor] = 1` iff `ancestor` is on `node`'s root
path), concatenated with the ordinary causal mask over the shared
prefix, one forward pass. **Position IDs are non-sequential across the
flattened tree** — every node at tree-depth `i` gets the same position
id (`prefix_len + i`), so RoPE treats sibling branches as the same
positional "slot." Real tree shape isn't raw Cartesian product — a
greedy tree-grower over a calibration dataset, 64-node sparse tree in
their main config (5 heads). Fixed/regular tree ⇒ mask + position-id
buffers are static, precomputed once at init
(`generate_medusa_buffers`), never rebuilt per step. Reported: Medusa-2
speedup 2.3–3.6×, acceleration rate (avg accepted tokens/pass) 3.47–3.51.
Paper's own toy worked example: head-1 top-2 × head-2 top-3 → 6 leaf
paths, but flattened *node* count (what actually goes in the
verification buffer, including the 2 head-1 nodes) = 8.

**EAGLE-2** (Li et al. 2024) — dynamic tree via per-node
confidence-product path score, expand top-k, global rerank, keep top-m,
flatten with the same ancestor-mask structure as Medusa. 3.05–4.26×
speedup, 20–40% over EAGLE-1's static tree. **Never quantifies the
expansion+reranking overhead itself** — leaves Decision 3's deferred
question (does a dynamic tree need a structurally different kernel, or
just host-side construction on the same masking primitive?) genuinely
unanswered in the literature — confirms it has to be settled by
building in Phase 3, not by citing EAGLE-2.

**Sequoia** (Chen et al.) — tree shape via a DP optimizer trading real
measured verification time `t(n)` against draft/target time ratio.
Empirical result directly relevant to Phase 1: optimal tree size is
compute-vs-bandwidth-regime dependent — on-device/compute-underutilized
setups (e.g. A100 batch=1) pick *small* trees (64–128 nodes, depth
6–10); memory-bandwidth-bound offloading setups pick much larger ones
(768 nodes, depth 18). A real external cross-check for whatever crossover
Phase 1 derives independently.

**vLLM** (confirmed from `vllm/v1/core/kv_cache_manager.py`) — chain
rejection is genuinely just a scalar: `num_computed_tokens` truncates to
the real accepted length, blocks beyond it get reclaimed by ordinary
refcount-to-zero on the next allocation cycle. No per-block "free on
reject" call exists — cut-point-plus-trailing-reclaim, exactly as
trivial as `spec.md` predicted for the chain case.

**PipeInfer** — closer primitive than vLLM, still doesn't solve tree
rollback: each cache cell carries a *set* of sequence-IDs (multi-owner
metadata exists), but the real free/copy operations only ever act on
whole contiguous partitions per speculative run, never at
individual-node granularity.

**Concrete gap for Phase 4** (grounded by the reading, not solved):
neither system has a primitive for evicting a non-contiguous,
branch-scoped subset of cache while a sibling branch in the *same*
storage container survives, nor a per-branch cut point — vLLM has one
integer per request, PipeInfer has one FIFO partition per run. Tree
rollback needs simultaneous per-branch accept-depths plus mixed
retain/evict within what's currently a single free/keep unit in both
systems.

---

## Phase 1 — Verification-regime roofline

### Mechanism, first-principles

Why speculative decoding wins despite the target model still running a
full attention/forward pass: plain autoregressive decode is memory-bound
because `seq_len_q=1` — FLOPs per step are tiny, but the entire model's
weights plus the KV cache still have to stream from HBM to produce one
token, so that huge fixed read cost is barely amortized. A forward
pass's weight-read cost barely changes whether it processes 1 query
token or K — weights are read once per layer and reused across however
many tokens are in the pass (the same stationary/streamed reuse logic as
GQA's own K/V reuse, generalized from "reused across query heads" to
"reused across query tokens"). Speculative decoding exploits exactly
this: guess K candidate tokens cheaply via a small draft model, then
verify all K in *one* target-model pass, paying roughly the same fixed
weight-read cost as one ordinary decode step but walking away with up
to K accepted tokens instead of one. The verification pass isn't
something spec decoding avoids running — it's the mechanism that cashes
in the amortization. This project's attention-sublayer-only roofline
below captures the KV-cache-specific version of that same story (the
full-model-weights version is out of scope, matching this portfolio's
established attention-sublayer-only scoping used throughout
prefill/decode).

### FLOPs and bytes, generalized for asymmetric seq_len_q ≠ seq_len_kv

Verification's shape: `seq_len_q = tree_len` (flattened tree token
count, small — tens of tokens), `seq_len_kv` = context length already
cached (can be large). Genuinely asymmetric — unlike prefill's
`seq_len_q = seq_len_kv` and decode's `seq_len_q = 1`.

**Bytes** (compulsory, fused, int8, GQA):

| Term | Formula | Note |
|---|---|---|
| Q | `batch × num_q_heads × tree_len × d_k` | scales with tree_len |
| K | `batch × num_kv_heads × seq_len_kv × d_k` | **pinned to context length, independent of tree_len** — read once, shared/reused across every tree token's attention (K/V-stationary dataflow, same mechanism as GQA's own reuse, now generalized across tree positions, not just query heads) |
| V | same as K | |
| O | same as Q | attention-output activation, same shape convention as prefill §1.2's output row — **not** the accept/reject decision, which is a downstream LM-head/sampling step, out of scope for this sub-layer roofline |

Total bytes = `2·batch·d_k·(num_q_heads·tree_len + num_kv_heads·seq_len_kv)`

**FLOPs:**

- QK^T = `2·batch·num_q_heads·tree_len·seq_len_kv·d_k` (contraction dim = `d_k`)
- P·V = `2·batch·num_q_heads·tree_len·seq_len_kv·d_k` (contraction dim = `seq_len_kv`) — same formula as QK^T despite different M/N/K role assignments, the same coincidence prefill §1.1 found, confirmed to hold in the asymmetric case too (both matmuls always reduce to `2×M×N×K` with the same three values, just relabeled)

Total FLOPs = `4·batch·num_q_heads·tree_len·seq_len_kv·d_k`

**Validated against known cases**: at `seq_len_q = seq_len_kv = 8192`
this exactly reproduces prefill's 2⁴⁶ FLOPs; at `seq_len_q = 1` it
exactly reproduces decode's 2³³.

**Softmax FLOPs negligibility, re-checked for the asymmetric case**:
matmul FLOPs and softmax FLOPs share the same `seq_len_q × seq_len_kv`
(= |P|, the P-matrix element count) factor — it cancels completely in
their ratio, leaving `ratio ≈ 4·d_head/c` (c = softmax's small
per-element op count: subtract-max, exp, sum, divide), independent of
`seq_len_q`, `seq_len_kv`, or the gap between them. Prefill's ~128×
negligibility argument carries over unchanged — neither strengthened nor
weakened by decoupling `seq_len_q` from `seq_len_kv` — confirmed via the
algebra, not assumed.

### AI(tree_len)

`batch` and `d_k` cancel completely out of the ratio:

**AI(tree_len) = 2·num_q_heads·tree_len·seq_len_kv / (num_q_heads·tree_len + num_kv_heads·seq_len_kv)**

Batch size has zero effect on regime — confirmed by direct algebraic
cancellation. Only `tree_len`, `seq_len_kv`, `num_q_heads`,
`num_kv_heads` matter.

### Concrete numbers

Continuity workload: `num_q_heads=64`, `num_kv_heads=8` (GQA),
`seq_len_kv=8192`, ridge point `C≈480.5` FLOPs/byte (TPU v5e, int8) —
same chip/workload prefill and decode used.

Two Medusa-sourced (arXiv:2401.10774) tree shapes:

- **8 nodes** (paper's own toy example: head-1 top-2, head-2 top-3,
  Cartesian — flattened node count = 2+2×3=8, not the 6 leaf-path
  count): **AI ≈ 127.0** → memory-bound (127 << 480.5)
- **64 nodes** (Medusa's real deployed sparse-tree config, 5 heads):
  **AI ≈ 963.6** → compute-bound (964 >> 480.5)

**Key finding — boundary validation, not planned in advance.**
`AI(tree_len)` is the *exact same continuous curve* that decode and
prefill sit at the two endpoints of:

- `tree_len = 1` (literal decode shape) → AI ≈ 16.0, matches
  `decode_notes.md`'s real GQA decode AI (≈15.98) almost exactly.
- `tree_len = 8192` (literal prefill shape, `seq_len_q = seq_len_kv`) →
  AI ≈ 14,564, matches `prefill_notes.md` §1.3's fused GQA AI (≈14,564)
  exactly.

So `tree_len` is literally the lever that interpolates verification
continuously between decode's memory-bound regime and prefill's
compute-bound regime — not just a qualitative analogy, an exact
quantitative match at both boundaries using the same formula.

### Not yet done / open for next session

- **Exact symbolic crossover** (`AI(tree_len) = 480.5`, solved for
  `tree_len`) — deliberately deferred. Agreed to do concrete plug-ins
  first (matches `spec.md`'s literal phrasing: "compute AI at that
  shape" for "a couple of concrete... tree shapes") and only generalize
  to the exact threshold if the concrete numbers make it interesting.
- **Unfused case** (P-matrix round-trip to HBM, matching prefill's
  fused-vs-unfused pair) — not yet derived for verification's shape.
- **Batch-vs-tree-size tradeoff, spec.md's explicit deliverable
  phrasing** — technically already answered (batch cancels, doesn't
  affect regime at all) but not yet written up as its own explicit
  statement against that phrasing.

Neither open item blocks moving to Phase 2 — see `handoff.md`.

---

## Phase 2 — Draft/target chip placement under synchronous coupling

### Scope decision: widen beyond attention-sublayer-only, for this phase only

Phase 1 (and decode/prefill before it) deliberately excluded weight-read
bytes/FLOPs — fine there because those phases only ever asked *ratio*
questions (AI, compute-vs-memory-bound regime), and weight bytes are a
fixed additive term that doesn't shift with anything those phases varied
(tree_len, batch), so omitting them never changed a conclusion. Phase 2
is the first phase asking for **absolute latency**, and the first phase
where **model size is itself a comparison variable** (draft vs. target) —
exactly where that omission stops being free, since a smaller draft
model's real-world speed advantage is dominated by *fewer weight bytes to
stream*, not by shrinking `num_q_heads`/`num_kv_heads`/`d_head` alone.

**Explicitly not retroactively widening Phase 1 or decode/prefill**:
their omission was scope-consistent (no cross-model-size comparison,
so nothing they concluded was distorted), and widening them now would
break the real, already-validated boundary-matching finding
(`AI(tree_len=1)`≈decode's real AI, `AI(tree_len=8192)`≈prefill's real
AI) — that match only holds because all three used the identical
attention-only formula. Widening is scoped to exactly where it's
load-bearing: Phase 2's draft/target comparison.

Decided: dense (not MoE) draft model, same-family as the Llama-3-70B/GQA
target (continuity + Decision 2's "8B-class, same family" framing).
SwiGLU MLP (matches real Llama-3 architecture, not a simplified 2-matrix
MLP) — `d_ff ≈ 3.5·d_model`, and this ratio turned out to be Llama-3's
*real* published number, not an approximation (confirmed once concrete
specs were pulled: 8B's `d_ff=14336`, `d_model=4096`, ratio exactly 3.5).

### Per-layer weight/projection formulas, derived

`W_Q`, `W_O` shapes simplify to `d_model²` (using the workload's own
`num_q_heads·d_head = d_model` convention); `W_K`, `W_V` stay narrower at
`d_model·num_kv_heads·d_head` — GQA shrinks K/V specifically, same
mechanism as every prior project's GQA reuse argument, now showing up in
weight bytes instead of just KV-cache bytes.

**Weight bytes/layer (int8)**: `2·d_model² + 2·d_model·num_kv_heads·d_head`
(attention) `+ 3·d_model·d_ff` (SwiGLU MLP).

**Projection FLOPs/layer** (standard `2×M×N×K` matmul form, `M=batch·seq_len_q`):
`4·batch·seq_len_q·d_model² + 4·batch·seq_len_q·d_model·num_kv_heads·d_head`
— same `2·d_model² + 2·d_model·num_kv_heads·d_head` shape as the weight-byte
formula, just scaled by `2·batch·seq_len_q` instead of being a static
param count (same four matrices, storage vs. compute).

**MLP FLOPs/layer**: `6·batch·seq_len_q·d_ff·d_model` — all three SwiGLU
matrices (gate, up, down) contribute *identical* FLOPs despite different
shapes (`(d_model,d_ff)` vs. `(d_ff,d_model)`), the same `2×M×N×K`
role-relabeling coincidence notes.md already flagged for `QK^T` vs. `PV`.

### Term 1 — K sequential draft-model decode steps

Draft model: **Llama-3-8B real specs** — `d_model=4096`, `d_ff=14336`,
`num_layers=32`, `num_q_heads=32`, `num_kv_heads=8`, `d_head=128`.
Workload stays continuity (`batch=32`, `seq_len_kv=8192`, int8).

Per-step total bytes (weights + Q/K/V/O activations, summed over all 32
layers) ≈ **24.17 GB** (≈6.98 GB weights, ≈17.19 GB activations —
activations dominate here because `seq_len_kv=8192` pins K/V-cache reads
regardless of model size).

Confirmed memory-bound by construction (decode's textbook case; FLOPs
side not computed — explicitly skipped as not worth the time given the
huge expected margin).

`K` = tree **depth** (number of sequential draft rounds), not `tree_len`
(flattened node count) — a real distinction, not interchangeable: `K` is
how many sequential passes the *draft* model needs to build the tree;
`tree_len` is what the *target* processes in one flattened verification
pass. Picked **K=5**, paired with Phase 1's 64-node Medusa "real
deployed" tree shape (5 heads → depth 5), to keep the whole round-latency
chain anchored to one concrete, already-grounded tree rather than mixing
shapes. (Sequoia's real depth-6–10/depth-18 numbers were the other
grounded option, not used, noted for future reference.)

### Term 2 — Handoff cost

What's transferred is the flattened tree's **token IDs** (+ small
tree-structure/position metadata), not activations — confirmed, not
assumed: the target model re-embeds and recomputes everything from its
own weights, so a draft-model activation (e.g. attention output `O`)
couldn't be fed in directly even if it were architecturally sensible —
dimensions don't even match across differently-sized models
(`num_q_heads·d_head` differs by model). This is *why* verification is
trustworthy ground truth rather than a continuation of the draft's own
reasoning.

Real ICI numbers (not assumed negligible): **≈1 μs/hop**, TPU v5e 2D
torus (4 neighbors/chip); the newer Boardfly topology caps worst-case
system-wide latency at **7 hops**. Round trip (candidates out, accept/
reject decision back — both directions matter under sync coupling, since
round N+1's draft can't start until round N's verify resolves) even at a
pessimistic 7-hop placement: `2×7×1μs ≈ 14 μs`. Against the ~ms-to-hundred-
ms-scale draft/verify terms, a 3–4 order-of-magnitude gap — negligible,
now a derived conclusion, not an assumption.

### Term 3 — Verification pass (target chip)

Target model: **Llama-3-70B real specs** — `d_model=8192`, `d_ff=28672`
(again exactly `3.5×d_model`), `num_layers=80`, `num_q_heads=64`,
`num_kv_heads=8`, `d_head=128` — matches continuity workload's head
counts exactly. `tree_len=64` (Phase 1's 64-node tree).

Total FLOPs, one verification pass, all 80 layers (projections + `QK^T`+`PV`
+ MLP) ≈ **3.244×10¹⁴**.

**Regime re-check under the widened formula** (not inherited from Phase
1's attention-only AI≈963.6): widened AI ≈ **2,844** — still ≫ ridge
(480.5 on v5e), compute-bound survives with a ~6× margin even once
weights/projections/MLP are counted, not just attention. Memory-time
alone (bytes/bandwidth ≈139.3ms) sits well under compute time
(≈823.2ms via FLOPs/throughput), confirming FLOPs/throughput is the right
number to trust for latency, not an assumption inherited from Phase 1.

### Capacity problem — real finding, forced a chip change

Llama-3-70B at int8 ≈ **70GB**, TPU v5e HBM = **16GB/chip** — the target
model cannot physically reside on one v5e chip at all (off by ~4.4×),
independent of the draft-model question entirely. This had been silently
assumed away since decode/prefill (a one-chip abstraction that's free for
*ratio* questions, since ratios are scale-invariant to ideal sharding,
but breaks for absolute multi-chip placement questions — same pattern as
the attention-sublayer-only scoping issue, now showing up on the capacity
axis).

Two real fixes considered: (1) model real tensor-parallel sharding on
v5e — keeps chip continuity but opens a genuinely new, deep sub-problem
(collective-communication/all-reduce cost between shards) that risks
scope creep past this phase's actual deliverable; (2) switch to a
bigger-capacity real chip. **Chose (2)**: **TPU 8i ("Zebrafish")**, a
real 2026 inference-focused TPU generation — confirmed real specs:
**288GB HBM at 8.6 TB/s**, 384MB on-chip SRAM, 10.1 FP4 petaflops, same
Boardfly ICI topology. `70GB + 8GB ≈ 78GB ≪ 288GB` — both models fit on
one chip with room to spare, which also removes a confound sharding would
have introduced (target-forced-multichip tangling with the actual
draft/target placement question).

**int8 compute not found as a confirmed spec** — one search snippet
implied FP8≈FP4 for this generation, which contradicts the standard
halve-bits-double-throughput pattern and looks like a synthesis artifact,
not trusted. Used a stated, flagged **estimate**: int8 ≈ FP4/2 ≈ **5,050
TOPS**. New ridge point: `5050/8.6 ≈ 587` FLOPs/byte (vs. v5e's confirmed
480.5).

### Recomputed on TPU 8i

| Term | v5e | TPU 8i |
|---|---|---|
| Draft (K=5) | 147.5 ms | 14.05 ms |
| Handoff | ~0.014 ms | ~0.014 ms |
| Verify | 823.2 ms | 64.23 ms |
| **Total** | **970.8 ms** | **≈78.3 ms** |
| Draft % | 15.2% | 17.9% |
| Verify % | 84.8% | 82.0% |

Bandwidth improved `8600/819≈10.5×`, compute improved `5050/394≈12.8×` —
close but not identical, which is *why* the draft/verify split shifted
slightly rather than staying frozen (a perfectly-matched scaling ratio
would have left the percentages exactly invariant; the small drift is a
direct, visible signature of compute outscaling bandwidth between these
two chip generations). Verify dominates the round on both chips by a
similar margin regardless of absolute hardware generation.

### Headline finding — the real placement answer

Under strict synchronous coupling, **colocating draft and verify on one
chip weakly dominates separating them onto two**, on pure latency
grounds: no overlap is possible either way (verify N cannot start until
draft N finishes, full stop — that's the coupling's definition), so
separation adds nothing but the handoff term, and that term is negligible
on any real interconnect (derived twice, not assumed once). TPU 8i's real
capacity also means separation is no longer *forced* by weight-capacity
the way v5e forces multi-chip sharding for the target alone — so if
separation happens, it has to be earned by something other than latency
or necessity.

Two real reasons remain, and this phase can only ground one of them:

- **Specialization** (derivable now, real precedent): draft is
  memory-bound, verify is compute-bound; a chip built to be excellent at
  both is a more expensive/over-provisioned part than two chips each
  matched to one regime — Groq's real production stack (SRAM-resident LPU
  drafts, GPU verifies) is a live instance of exactly this.
- **Fleet-level pipelining** (real, but explicitly out of this phase's
  scope): interleaving many concurrent requests' rounds across
  independent chip pools is a genuine throughput argument, but it's a
  scheduling-level phenomenon invisible to a single-round latency model —
  `spec.md` names this exact question as deliberately left for the build
  phase, not something this roofline derivation resolves.

**Amdahl's-law sharpening of the specialization argument, plus a real
capacity check on it.** Since draft is only ~15–18% of the round
(15.2% v5e, 17.9% TPU 8i), making drafting arbitrarily fast (SRAM
residency, effectively free) is capped by Amdahl's law at
`1/(1−draft_fraction)` ≈ **1.18×** (v5e split) to **1.22×** (8i split)
maximum total-latency improvement — the interesting lever for
SRAM-residency isn't drafting at all, it's whether it could also
compress verify's dominant ~82%+ term (the "target weights also
SRAM-resident, verification itself nearly free" thread already flagged
in `spec.md`'s Groq reading item).

**It can't, for this target model, and not for a latency reason —
a capacity one, the same theme recurring a third time in this project.**
Groq's real SRAM is ~230–500MB/chip. At int8: draft (Llama-3-8B, ~8GB)
needs `8000MB/500MB ≈ 16 chips` for weight capacity alone — tractable.
Target/verify (Llama-3-70B, ~70GB) needs `70000MB/500MB = 140 chips`
minimum, before KV-cache/activation headroom — real large-model Groq
deployments do run on hundreds of chips for exactly this reason. So the
real LPU/GPU split isn't only regime-matching (memory-bound draft ↔
bandwidth chip, compute-bound verify ↔ compute chip) — there's an
independent, capacity-forced reason underneath it: a 70B-class verify
model is essentially unhoused on SRAM-only hardware at any sane chip
count, while an 8B-class draft model is small enough that SRAM
residency is genuinely viable. Same pattern as the v5e HBM-capacity
finding above, now recurring one level down at SRAM granularity — this
project keeps finding that a scoping choice free for a regime/ratio
question turns out capacity-load-bearing once real hardware is asked to
actually hold the model.

**Checkpoint met**: round-latency model derived (symbolic + two concrete
real-chip runs); real, non-hedged stated answer — synchronous coupling
reverses disagg's default "separate chips" instinct on latency grounds
alone, leaving specialization as the one derivable justification for
separation and fleet-level pipelining as the other, correctly left open
rather than force-resolved here.

### Not yet done / open for next session

- TPU 8i's int8 compute figure is a stated estimate (FP4/2), not a
  confirmed spec — re-verify if a real number surfaces later.
- `K=5` is a concrete worked example (paired with the 64-node tree), not
  a swept/general result across tree depths — Sequoia's real depth-6–10
  vs. depth-18 numbers remain an available, ungrounded-here alternative.
- Tensor-parallel sharding + collective-communication cost on v5e was
  named as a real, deliberately not-chased path (chose TPU 8i instead) —
  a legitimate future thread if v5e-specific numbers are ever needed.

Neither open item blocks moving to Phase 3 — see `handoff.md`.
