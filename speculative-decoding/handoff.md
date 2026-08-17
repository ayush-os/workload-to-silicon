# Handoff — Speculative Decoding Codesign (speculative-decoding)

**Read order for a fresh chat**: `spec.md` (full phase plan; Phase 0's
three decisions are already recorded there) → `notes.md` (everything
derived so far — formulas, numbers, findings) → this file (current
status, what's next).

## Status

Phase 0 (reading) and Phase 1 (verification-regime roofline) are both
done to the point of moving on. Phase 1's checkpoint is met: two
concrete tree shapes checked against the ridge point, plus an unplanned
bonus finding — the derived `AI(tree_len)` formula exactly reproduces
decode's real AI at `tree_len=1` and prefill's real AI at
`tree_len=8192`, meaning `tree_len` is a literal, continuous lever
connecting this portfolio's two other established regimes, not just an
analogy between them. Two Phase 1 loose ends (exact symbolic crossover,
the unfused case) are flagged but deliberately not chased — see
`notes.md`'s "Not yet done" note. Neither blocks Phase 2.

## What's done: Phase 0

Reading list complete (Leviathan/Chen, Draft & Verify, SpecInfer,
Medusa, EAGLE-2, Sequoia, vLLM `spec_decode` source, PipeInfer) — full
synthesis in `notes.md`. The three Phase 0 decisions were already
recorded directly in `spec.md` before this session (tree-structured,
standalone draft model, static tree with dynamic deferred) and weren't
re-litigated.

Real findings from the reading, beyond what `spec.md` already states:
- Medusa's actual mask mechanism, confirmed from source: per-node
  ancestor bitmask + non-sequential position IDs (same tree-depth = same
  position id) — direct precedent for Phase 3's kernel.
- Sequoia's real hardware-aware tree sizing (small trees ~64–128 nodes
  for on-device/compute-underutilized setups, large trees ~768 for
  memory-bandwidth-bound offloading) — an external cross-check for
  whatever crossover Phase 1 eventually derives exactly.
- The Phase 4 gap is now concretely characterized, not just asserted:
  neither vLLM nor PipeInfer has a primitive for non-contiguous,
  branch-scoped eviction with a surviving sibling in the same container,
  or per-branch cut points.
- Draft & Verify's ≈0.91-flat acceptance number is confirmed real but
  flagged as **not** directly reusable for Phase 5 — self-speculative
  setup, correlated draft/target error — doesn't transfer to this
  project's standalone-draft (Decision 2) setup without a fresh
  argument.

## What's done: Phase 1 (verification-regime roofline)

Full derivation in `notes.md`. Headline formula:

`AI(tree_len) = 2·num_q_heads·tree_len·seq_len_kv / (num_q_heads·tree_len + num_kv_heads·seq_len_kv)`

`batch` and `d_k` cancel completely — batch size has zero effect on
regime, confirmed algebraically, not assumed.

Two concrete Medusa-sourced tree shapes checked against ridge=480.5
(TPU v5e int8, continuity workload `num_q_heads=64` /
`num_kv_heads=8` / `seq_len_kv=8192`):
- 8-node toy tree (Medusa's own top-2×top-3 Cartesian example): AI≈127.0
  → memory-bound.
- 64-node real deployed tree (Medusa's main config): AI≈963.6 →
  compute-bound.

**Real finding, not planned going in**: the formula's two boundary cases
exactly reproduce this portfolio's other two projects' real numbers —
`tree_len=1` gives AI≈16.0 (matches `decode_notes.md`'s GQA decode
AI≈15.98), `tree_len=8192` gives AI≈14,564 (matches `prefill_notes.md`
§1.3's fused GQA AI exactly). Verification sits on the exact same
continuous curve connecting decode and prefill, with `tree_len` as the
parameter — a genuinely unifying result across three separate projects
in this portfolio, not just a qualitative analogy.

Open, deliberately not chased this session: the exact symbolic
crossover point (`AI(tree_len)=480.5` solved for `tree_len`), and the
unfused (P-matrix round-trip) case. Neither blocks moving to Phase 2 —
see `notes.md`'s "Not yet done" note for the reasoning on why concrete
plug-ins were done first.

## What's next: Phase 2 (draft/target chip placement under synchronous coupling)

Per `spec.md`: derive the real round-latency chain (K sequential
draft-model decode steps on the draft chip + handoff cost + one
flattened-tree verification pass on the target chip, using Phase 1's
own shape). Then the real open question this phase turns on: does
putting draft and target on **separate** chips still make sense under
the synchronous constraint (verify N can't start until draft N
finishes; draft N+1 can't start until verify N resolves), or does
separation just add pure serial round-trip latency with none of
disagg's real pipelining upside — since, unlike disagg's async
prefill/decode pools, there's no independent overlappable work to hide
it behind? A real, derived answer is expected either direction, not
assumed by default.

## Workflow note for a fresh session

This project runs Socratic: the user does the actual derivations/math —
that's explicitly where the learning is — and the assistant checks/
corrects precisely against source docs rather than deriving unprompted.
Arithmetic/formula-assembly/numeric-plug-in work only gets done directly
when **explicitly, imperatively delegated** ("plug in the numbers,"
"you can just do X") — not just because the user seems frustrated with
the pace of getting somewhere. Concrete miscalibration this session,
worth not repeating: after a "let's just calculate it, I don't see why
we have to speculate" comment (frustration at premature Socratic
hedging about direction), the assistant read that as full delegation
and did the actual compute-vs-memory-bound arithmetic itself — the
user's real reaction was "face palm, I'm supposed to do the actual math,
that's where the learning is." Dropping unnecessary hedging once the
approach is settled is correct; handing over arithmetic that answers the
open technical question is not, unless it's a direct imperative. Default
to checking mode; wait for an explicit imperative before computing
anything that itself answers an open question.

## Key numbers carried forward

- Continuity workload: `batch=32`, `num_q_heads=64`, `num_kv_heads=8`
  (GQA), `d_head=128`, `d_model=8192`, int8, `seq_len_kv=8192` — same as
  prefill/decode.
- Ridge point: `C≈480.5` FLOPs/byte (TPU v5e, int8) — same chip as
  prefill/decode.
- `AI(tree_len) = 2·num_q_heads·tree_len·seq_len_kv / (num_q_heads·tree_len + num_kv_heads·seq_len_kv)`.
- Two grounded tree shapes: 8 nodes (Medusa toy example, memory-bound),
  64 nodes (Medusa real deployed config, compute-bound).

---

Not done, deliberately: root `README.md` not touched — repo convention
(see `disagg_and_placement_notes.md`'s and `rl_codesign_notes.md`'s own
history) is to update it once a project reaches its own final phase, not
mid-project.
