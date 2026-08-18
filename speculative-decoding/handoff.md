# Handoff — Speculative Decoding Codesign (speculative-decoding)

**Read order for a fresh chat**: `spec.md` (full phase plan; Phase 0's
three decisions are already recorded there) → `notes.md` (everything
derived so far — formulas, numbers, findings) → this file (current
status, what's next).

## Status

Phase 0 (reading), Phase 1 (verification-regime roofline), and Phase 2
(draft/target chip placement under synchronous coupling) are all done to
the point of moving on. Phase 2's checkpoint is met: a real round-latency
model (K sequential draft steps + handoff + verification pass), derived
symbolically and run concretely on two real chips, plus a real, stated
answer on the phase's core open question — see below. Phase 3 (the real
Triton tree-attention-mask kernel) is next and not yet started.

## What's done: Phase 0

Reading list complete (Leviathan/Chen, Draft & Verify, SpecInfer, Medusa,
EAGLE-2, Sequoia, vLLM `spec_decode` source, PipeInfer, plus Groq's real
production spec-decode stack and MatX One's confirmed hybrid SRAM+HBM
architecture) — full synthesis in `notes.md`. The three Phase 0
decisions were already recorded directly in `spec.md` before this
project started and weren't re-litigated: tree-structured over chain,
standalone draft model over self-speculation, static tree with dynamic
explicitly deferred.

## What's done: Phase 1 (verification-regime roofline)

Full derivation in `notes.md`. Headline formula:

`AI(tree_len) = 2·num_q_heads·tree_len·seq_len_kv / (num_q_heads·tree_len + num_kv_heads·seq_len_kv)`

Two concrete Medusa-sourced tree shapes checked against ridge=480.5 (TPU
v5e int8): 8-node toy tree → memory-bound; 64-node real deployed tree →
compute-bound. Real, unplanned finding: the formula's two boundary cases
(`tree_len=1`, `tree_len=8192`) exactly reproduce decode's and prefill's
real AI numbers — verification sits on the same continuous curve
connecting this portfolio's two other regimes, not just an analogy.

Two loose ends flagged, deliberately not chased, neither blocking:
symbolic crossover point, unfused (P-matrix round-trip) case.

## What's done: Phase 2 (draft/target chip placement under synchronous coupling)

Full derivation in `notes.md`. Real round-latency chain built from three
terms, all on the 64-node Medusa tree (`tree_len=64`) and `K=5` draft
depth (paired to that same tree's 5-head structure):

- **Draft (K=5 sequential Llama-3-8B decode steps)**: 147.5 ms (v5e) /
  14.05 ms (TPU 8i)
- **Handoff (token IDs only, real ICI numbers)**: negligible either way
  (~14 μs) — derived from real TPU hop-latency numbers, not assumed
- **Verify (one flattened-tree pass, Llama-3-70B)**: 823.2 ms (v5e) /
  64.23 ms (TPU 8i)

**Real, forced mid-phase pivot**: Llama-3-70B int8 (~70GB) doesn't
physically fit on one TPU v5e chip (16GB HBM) — a real capacity finding,
not a hypothetical, that had been silently assumed away since
decode/prefill. Chose to switch to **TPU 8i** (real 2026 inference TPU,
288GB HBM/8.6TB/s, confirmed specs; int8 compute is a stated ~5,050 TOPS
estimate, not confirmed) over modeling tensor-parallel sharding on v5e,
to avoid opening a collective-communication sub-problem that would have
exceeded this phase's scope.

**Headline answer**: under strict synchronous coupling, colocating draft
and verify on one chip weakly dominates separating them, on pure latency
grounds — no overlap is ever possible either way, and handoff is
negligible on any real interconnect. Separation is therefore only
justified by **hardware specialization** (derivable now — draft is
memory-bound, verify is compute-bound, real Groq LPU/GPU precedent) or
**fleet-level pipelining across concurrent requests** (real, but
explicitly out of this phase's scope per `spec.md` — a build-phase
scheduling question, not a roofline one).

**Specialization sharpened further, post-checkpoint**: Amdahl's law caps
SRAM-residency's max benefit to draft alone at ~1.18–1.22× (draft is
only 15–18% of the round) — the real lever would be compressing verify's
dominant ~82%+ term instead. But that's capacity-blocked, not just
underexplored: Groq's real ~230–500MB/chip SRAM houses the 8B draft
model in ~16 chips but would need ~140+ chips for the 70B target's
weights alone — the same HBM-capacity theme from below, recurring one
level down at SRAM granularity. Full numbers in `notes.md`.

Also scope note carried forward: Phase 2 deliberately widened past
Phase 1's attention-sublayer-only formula (to include weight bytes/FLOPs
via full projection+SwiGLU-MLP terms) because model size became a live
comparison variable for the first time — Phase 1/decode/prefill were
deliberately **not** retroactively widened, since their omission never
distorted a ratio-only conclusion and widening them would break the
Phase 1 boundary-matching validation.

## What's next: Phase 3 (real Triton kernel — tree-attention verification mask)

Per `spec.md`: build a real Triton kernel implementing the ancestor-path
attention mask for the **static** tree shape (Decision 3), matching
project #5's real-hardware discipline (real A100, not simulated),
verified against a golden PyTorch reference across several concrete tree
shapes. Direct precedent already read and confirmed in Phase 0: Medusa's
real mask mechanism (`medusa/model/utils.py`) — per-node ancestor
bitmask + non-sequential position IDs (same tree-depth = same position
id).

The real, deferred question from Decision 3 gets resolved here by
building, not assuming: does a dynamic (EAGLE-2-style) tree need a
structurally different kernel, or is the masking primitive the same with
the real cost sitting entirely in host-side tree construction? Attempt
the dynamic-tree variant after the static one works.

**This project's 64-node tree shape (Phase 1/2's concrete example)** is
a natural first candidate to validate the kernel against, for continuity
with everything derived so far — though `spec.md` asks for several
concrete shapes, not just one.

## Workflow note for a fresh session

This project runs Socratic: the user does the actual derivations/math —
that's explicitly where the learning is — and the assistant checks/
corrects precisely against source docs rather than deriving unprompted.
Arithmetic/formula-assembly/numeric-plug-in work only gets done directly
when **explicitly, imperatively delegated** ("plug in the numbers," "you
can just do X") — not just because the user seems frustrated with the
pace of getting somewhere. This held throughout Phase 2 as well:
concrete numeric plug-ins (draft-step bytes, verification FLOPs, the
TPU 8i re-run) were only computed after explicit delegation; symbolic
setup (matrix shapes, FLOPs formulas) stayed the user's own derivation
work, checked/corrected rather than pre-solved.

One live, recurring pattern worth carrying into Phase 3: this project
keeps surfacing places where a scoping choice that was free for a
*regime/ratio* question (attention-sublayer-only bytes, one-chip
capacity) turns out load-bearing once the question shifts to *absolute*
numbers (latency, real capacity). Worth checking early in Phase 3 whether
the same trap exists there before trusting any inherited formula.

## Key numbers carried forward

- Continuity workload: `batch=32`, `num_q_heads=64`, `num_kv_heads=8`
  (GQA), `d_head=128`, `d_model=8192`, int8, `seq_len_kv=8192`.
- Ridge point: `C≈480.5` FLOPs/byte (TPU v5e, int8); TPU 8i's ridge
  `≈587` FLOPs/byte (int8 compute estimated, not confirmed).
- Target model: Llama-3-70B real specs — `d_model=8192`, `d_ff=28672`,
  `num_layers=80`, `num_q_heads=64`, `num_kv_heads=8`, `d_head=128`.
- Draft model: Llama-3-8B real specs — `d_model=4096`, `d_ff=14336`,
  `num_layers=32`, `num_q_heads=32`, `num_kv_heads=8`, `d_head=128`.
- Two grounded tree shapes: 8 nodes (memory-bound), 64 nodes / depth 5
  (compute-bound) — the 64-node/depth-5 pair is what Phase 2's
  round-latency numbers above are built on.
- TPU 8i real specs: 288GB HBM @ 8.6TB/s, 384MB SRAM, 10.1 FP4 PFLOPS
  (int8 ≈5,050 TOPS, estimated).

---

Not done, deliberately: root `README.md` not touched — repo convention
is to update it once a project reaches its own final phase, not
mid-project.
