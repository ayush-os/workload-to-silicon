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
