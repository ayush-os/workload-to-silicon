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
