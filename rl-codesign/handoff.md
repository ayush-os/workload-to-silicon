# Handoff — RL Post-Training Codesign (rl-codesign)

**Read order for a fresh chat**: `spec.md` (full phase plan) → `notes.md`
(everything derived so far — decisions, formulas, numbers, findings) →
this file (current status, what's next).

## Status

**Phase 0 (setup) and Phase 1 (rollout-side roofline) are complete.**
Phase 2 (training-side roofline) has not started.

## What's done

- **Phase 0**: reading list read (HybridFlow, RLHFuse, RhymeRL, AReaL,
  StreamRL, BiDiRL, DeepSeek-R1), plus Miles (SGLang/RadixArk) found and
  read mid-project. Two decisions made and defended: (1) keep the
  reference model — GRPO's KL term is real in R1, refreshed every 400
  steps, not frozen; (2) precision-conversion needs fresh Phase 3
  treatment, not a direct reuse of decode's coupled/decoupled framework —
  it's a different question (transition cost vs. within-workload
  regime-flip).
- **Phase 1**: rollout-side roofline fully derived — FLOPs, bytes, and
  wall-clock, all three. Headline: **decode completely dominates rollout
  wall-clock** (prefill ≈0.001–0.017% share), driven by KV-cache movement
  once converted from raw FLOPs to real wall-clock time. Full mechanism
  chain and all formulas/numbers are in `notes.md`.

## What's next: Phase 2 (training-side roofline)

Per `spec.md`: forward+backward FLOPs (verify the ~3× forward-FLOPs rule
holds for MoE's sparse activation pattern specifically, don't assume it
transfers unchanged from dense), Adam optimizer state memory (source the
real multiplier, don't guess the constant), activation memory during
backprop (scales with seq_len × batch, derive explicitly rather than
assuming it's negligible). Then quantify GRPO's critic-free saving (no
critic weights, no critic optimizer state, no critic activation memory)
against a stated PPO baseline, using this project's own MoE/MLA
parameter-count numbers.

**Before Phase 2 leans on it**: TPU 8i's real BF16 peak FLOPs is still
unconfirmed — only a provisional estimate exists (≈2.525 PFLOPS via a
halving-pattern inferred from the sibling TPU 8t chip, not stated directly
for 8i in the primary Google Cloud source). Worth a real search/source
pass before building Phase 2's roofline on top of it.

## Open threads to keep in view

- **Rollout Routing Replay** (Miles' MoE-routing-drift fix) — real,
  workload-specific, not yet scoped in or out. Decide during Phase 2 or 3.
- **P=1,024 prompt length** is an unsourced assumption (Phase 1's own
  pick) — revisit if a real anchor turns up.
- **N≈640 decode batch size** reused from disagg's own HBM-capacity
  grounding — re-verify if Phase 2/3 changes deployment assumptions (e.g.
  a different EP-group size).

## Key numbers carried forward

K=16 (R1-confirmed group size), P=1,024 (assumption), R∈{8,192, 65,536}
(two response-length anchors), 60 layers (DeepSeek-V2's real count),
N≈640 (realistic concurrent decode batch, reused from disagg), TPU 8i:
10.1 PFLOPS FP4 (confirmed), ≈2.525 PFLOPS BF16 (provisional), 8.6 TB/s
HBM bandwidth, ridge point ≈1,174 FLOPs/byte.

## Not done, deliberately

Root `README.md` has not been touched — this repo's convention (see
`disagg_and_placement_notes.md`'s own history) is to update it once a
project reaches its own Phase 5 completion, not mid-project. rl-codesign
is only through Phase 1 of 5, so this is intentional, not an oversight.
