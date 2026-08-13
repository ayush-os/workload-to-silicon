# Handoff — RL Post-Training Codesign (rl-codesign)

**Read order for a fresh chat**: `spec.md` (full phase plan) → `notes.md`
(everything derived so far — decisions, formulas, numbers, findings) →
this file (current status, what's next).

## Status

**Headline finding so far**: rollout is most likely more of a
bottleneck than training; the magnitude of that dominance depends on
the length of rollouts (R), your chip count, and your sharding scheme.
Full derivation and the one known exception in `notes.md`'s Phase 3
section.

**Phase 0, Phase 1, and Phase 2 are complete. Phase 3 is well underway**
— the shared rollout:train wall-clock/chip-ratio question is derived
from multiple angles (mismatched, matched-chip, cross-chip-tax,
throughput-balanced disaggregated), and items 1–3 (weight-broadcast
topology, sync-boundary transfer cost, colocated resharding) are now
resolved this session too. See "What's next" below — the final
synthesis (item 4) is the one remaining piece.

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

## What's done: Phase 2 (training-side roofline)

**Forward+backward FLOPs multiplier resolved**. The 2×
backward-vs-forward ratio holds regardless of MoE sparsity (pure backprop
matmul algebra, confirmed not just assumed) — but activation
recomputation is numerically forced at this project's own R (a single
MLA layer's attention-score activation term alone is ~12.7× TPU 8t's
entire 216 GB HBM at R=65,536, batch=1), which pushes the real multiplier
to **~4×**, not the naive 3×. Full derivation and sourcing in `notes.md`.
Two open refinements flagged there: (1) the source paper for this used
DeepSeek-V3's dims, this project has used V2's elsewhere — reconcile;
(2) 4× assumes *full* recomputation, real practice uses *selective* —
narrow this before treating the number as final.

**Also done**: Adam optimizer state memory. 236B total params (V2's own
real count, `moe-routing-notes.md`) × 12 bytes/param (ZeRO's canonical
mixed-precision Adam: 4 FP32 master + 4 FP32 momentum + 4 FP32 variance)
= **2.83 TB total optimizer state — ≥13 TPU 8t chips just to hold it**,
before weights/gradients/activations. Total params used, not activated
(21B) — Adam state must persist per-parameter regardless of per-token
routing sparsity. Real-world corroboration: R1's own reported training
used optimizer CPU offloading, consistent with this not fitting
on-device. **Distinct from the activation-recomputation finding, not
reinforcing it**: recomputation only touches activations (recreatable by
rerunning forward) and *reduces* pressure toward more chips; optimizer
state is persistent accumulated state with no recompute equivalent, so
it sets its own independent chip-count floor (≥13). Full derivation in
`notes.md`.

**Also done**: GRPO's critic-free saving, quantified against a PPO
baseline. GRPO's own paper (DeepSeekMath, arXiv 2402.03300) states the
critic is "comparable to the policy model in size" — modeled as a second
236B-param copy. Result: a flat ~2× reduction across every major
training resource (optimizer state 2.83→5.66 TB, ≈13→26 chips; weights
doubled; activation-recomputation burden doubled; training FLOPs/step
~4×→~8× forward-equivalent). Full table in `notes.md`.

**Also done**: per-device parallelism config for training on TPU 8t.
Starting from EP=8 alone (inherited from Phase 1's rollout config), a
device needs 661.5 GB — 3.06× over the 216 GB budget, before
activations. Landed on **TP=4 (attention) × EP=40 (experts) = 160
devices**, which clears weights+gradients+optimizer with ~41 GB
headroom, and divides the 160 routed experts evenly. Non-obvious
result: **PP turned out unnecessary** — expected 60 layers to force
pipeline-splitting, but TP+wider-EP closed the gap without it.
Activation memory against that headroom deliberately not precisely
rederived under this exact grid — sharding can only shrink it relative
to the already-computed unsharded estimate, so the headroom check is
already against a conservative upper bound; flagged rather than chased
further, matching this project's own established discipline. Full
table and reasoning in `notes.md`.

**Phase 2 checkpoint met**: training FLOPs, bytes/memory, and GRPO's
critic-free saving are all derived and sourced. Phases 1–3 (once Phase 3
is done) will stand alone as this project's complete fallback artifact
per `spec.md`. The TP=4/EP=40 choice was validated, not just accepted —
`notes.md` has the check confirming EP alone (maxed to EP=160) still
fails (216.6 GB, no headroom), plus a TP-vs-ZeRO equivalence comparison
showing TP was the better-grounded pick, not an arbitrary one. Also
recorded there: a meta-pattern worth carrying into Phase 3 — Phase 1's
constraint was *bandwidth* (decode's HBM-bandwidth-bound), Phase 2's is
*capacity* (does it fit at all) — same roofline instinct, different axis.

**Note for Phase 3**: training's EP degree (40) is now different from
rollout's (8, from Phase 1) — the "N≈640 decode batch size, re-verify if
Phase 2/3 changes EP-group size" open thread below is no longer
hypothetical, it's now confirmed to have happened. Also worth noting:
training now spans 160 devices for one model "replica" (before DP is
even applied) — a real input for Phase 3's colocated-vs-disaggregated
resource-allocation comparison.

**Resolved since Phase 1**: TPU 8i's BF16 peak was searched for directly
(primary Google Cloud source, comparative codelabs writeup, three
secondary sources) and confirmed **not to exist publicly for either
8th-gen chip** — not a documentation gap on 8i, a real absence on both 8i
and 8t. Bigger finding underneath it: this is the first TPU generation
Google has split into two physically distinct chips, 8t
(training-optimized, FP4-native) and 8i (inference-optimized,
FP4-native) — real evidence against a BF16-training baseline for this
hardware, and against "colocated single chip pool" being the
hardware-native shape at all. **Decision 2 revised** (see `notes.md`):
Phase 2 now models training on TPU 8t at its own native FP4 peak (12.6
PFLOPS/chip), and Phase 3's weight-sync question is reframed as a
cross-chip FP4(8t)→FP4(8i) transfer, not a BF16→FP4 conversion.

**Also resolved**: TPU 8t's HBM bandwidth/capacity, sourced from the same
primary Google Cloud table — 216 GB capacity, 6,528 GB/s bandwidth, own
ridge point ≈1,930 FLOPs/byte (right of 8i's ≈1,174 — training needs a
higher compute:memory ratio to stay compute-bound). Also found: 19.2
Tb/s/chip ICI bandwidth, stated for both chips — a real candidate number
for Phase 3's cross-chip weight-sync derivation, though it's not yet
confirmed whether 8t/8i pods even share a fabric for that traffic
(flagged in `notes.md`, don't assume without checking).

## What's done so far in Phase 3

All in `notes.md`'s Phase 3 section, this session's work — read that
section in full before continuing, this summary is a pointer, not a
substitute.

1. **Rollout:train wall-clock split derived from four angles**, each
   answering a genuinely different question — don't collapse them into
   one number:
   - **Mismatched pools** (EP=8 rollout vs. 160-device training, each
     independently capacity-sized): 34.6×→21.1×, asymptotes to **16.1×**
     as R→∞, never crosses over. This is the *disaggregated*-flavored
     input (separate, unevenly-sized pools by design), not colocated.
   - **Matched-chip colocated** (both phases forced to the same N, since
     training's capacity floor sets the pool size): tested across
     multiple parallelism layouts (EP=160-alone, TP=4×EP=40, TP=4×EP=160)
     — ratio ranges **2.2×–13.8×** depending on config and R, direction
     robust, magnitude isn't. Two real sub-findings: optimal rollout
     layout is R-dependent (TP helps more at short R, wide EP helps more
     at long R), and adding proportionally more chips to *both* sides
     widens the ratio in training's favor, it doesn't close it.
   - **Colocated on one physical chip** (the real constraint colocation
     forces, since TPU 8t/8i are physically different silicon, not modes
     of one chip): chose 8i (rollout stays native, training eats a
     2.50× tax) with matched TP=2×EP=40=80-device layout for both modes
     (no weight-reshard needed) — tightest colocated number yet,
     **3.76× (R=8,192) / 1.93× (R=65,536)**.
   - **Disaggregated, real throughput-balanced chip ratio** (not the
     mismatched-pool number above — this solves for N_t that equalizes
     service time with a chosen N_r, on each phase's own native chip, no
     cross-chip tax at all): ratio climbs with N_r, converges to AReaL's
     real ~3:1 empirical split around N_r≈32–40 at R=8,192; needs a much
     larger N_r to reach the same ratio at R=65,536 (real, new finding —
     longer R needs proportionally more rollout chips to sustain a given
     balance ratio).
2. **Pool size (N) is a free variable, not derivable in isolation** —
   minimizing the rollout:train ratio and minimizing total step wall-clock
   are *different objectives* that move in opposite directions as N
   grows. Resolved by deferring colocated's "right" N to whatever total
   budget the disaggregated derivation lands on (not yet finalized — see
   step 4 below).
3. **One claim walked back mid-session**: "training has no structural
   bottleneck, ever" was overstated — only verified for the specific
   configs tested (TP held fixed while EP widened). Whether scaling TP
   *proportionally* removes rollout's floor too is genuinely unknown,
   and deliberately not chased (no TP/EP communication-overhead modeling
   anywhere in this project, and V2's real attention head count — the
   actual TP ceiling — isn't sourced). Don't re-assert the strong version
   of this claim without addressing those two gaps first.

## What's done this session: items 1–3, one item left

**Items 1+2 resolved** (`notes.md`'s new "Weight-broadcast topology and
sync-boundary transfer cost" subsection, end of Phase 3 section): 8t and
8i confirmed to share **no fabric** (3D torus vs. Boardfly, no shared ICI
— the earlier "candidate" 19.2 Tb/s ICI number does *not* apply
cross-pool). Real path is Jupiter (Google's general DCN), 400 Gb/s/host.
Single-link baseline transfer cost (fan-in + cross-pool + fan-out)
**≈2.75 s** — headline finding: this exceeds training's own per-step
wall-clock by **~3.4×** at R=8,192 (11% at R=65,536), a real, load-bearing
cost, not a footnote. Architecture-tradeoff space also mapped (single
link vs. parallel-matched-layout vs. parallel-mismatched-with-reshard) —
single-link ≈2.75 s is the working default carried into item 4; faster
options exist but are contingent on giving up disaggregated's
independent pool-sizing or eating an unmodeled reshard cost (HybridFlow's
own 70B-dense reshard numbers are the real anchor for how bad that could
get). One methodological note recorded there too: an earlier pass
mis-sourced Boardfly's Group size as 288 chips (a WebFetch conflation
with TPU 8i's 288 GB HBM capacity) — caught, corrected to the real 32
chips/Group (8 BBs × 4 chips), re-verified against the stated 1,152-chip
pod total.

**Item 3 resolved this session too** (`notes.md`'s new "Colocated
resharding cost — scoped down, not derived" subsection): same-layout-on-
8i (TP=2×EP=40=80 devices, both modes) means devices never move shards
between modes at all — a genuinely zero-reshard transition, not just a
cheap one, so no reshard cost gets added on top of the 3.76×/1.93×
colocated ratios already carried forward. The different-layout
alternative (real HybridFlow-style reshard, anchored to their own 70B-
dense numbers: up to 36.4% naive, ~11.7s avg optimized, 78.2s worst case)
is flagged as a deprioritized data point, not derived for this project's
236B MoE scale — the same-layout answer already in hand is better, so
deriving the alternative wouldn't change what's carried into item 4.

## What's next: one remaining item

1. **The actual comparison — Phase 3's real checkpoint.** Now that
   disaggregated's complete cost is in hand (chip ratio + broadcast +
   sync transfer, ≈2.75 s single-link default), synthesize: under what
   conditions (model size, cluster size, rollout:train FLOPs ratio, R)
   does colocated vs. disaggregated win? This is the deliverable
   `spec.md` actually asks for — everything above is an input to it, not
   the answer itself. This is also where the pool-size-ambiguity open
   item (above) gets resolved: match colocated's N to whatever total chip
   budget the disaggregated derivation settles on, for a fair comparison.

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
N≈640 (realistic concurrent decode batch, reused from disagg).

TPU 8i (rollout chip): 10.1 PFLOPS FP4, 288 GB HBM capacity, 8.6 TB/s HBM
bandwidth, 384 MB Vmem, ridge point ≈1,174 FLOPs/byte — all confirmed.

TPU 8t (training chip, Phase 2's anchor): 12.6 PFLOPS FP4/chip (121
EFLOPS/pod), 216 GB HBM capacity, 6.528 TB/s HBM bandwidth, 128 MB Vmem,
ridge point ≈1,930 FLOPs/byte — all confirmed. No BF16 number exists for
either chip (confirmed absent, not just unsourced — see Decision 2
revision).

Both chips: 19.2 Tb/s/chip ICI bandwidth (2× prior gen) — **resolved,
does not apply to the cross-pool weight-sync transfer** (8t/8i pods
share no fabric; ICI is pod-internal only). Real cross-pool path is
**Jupiter** (Google's general-purpose north-south DCN), **400 Gb/s/host**
— see below. TPU 8i's own Boardfly topology: 4-chip Building Block (full
mesh) → 32-chip Group (8 BBs, full mesh via copper) → 1,152-chip Pod (36
groups, via OCS), 7-hop max diameter.

**This session's Phase 3 numbers**: colocated-on-8i's parallelism config
is **TP=2×EP=40 = 80 devices** (211.6 GB/216... i.e. /288 GB 8i budget,
76.4 GB headroom) — distinct from Phase 2's training-on-8t config
(TP=4×EP=40=160 devices, 174.7 GB/216 GB). Training's cross-chip tax
running on 8i instead of native 8t: **2.50×** slower (2× from half the
chips × 1.248× from 8i's lower FP4 peak). Weight bytes for the Phase 3
sync-transfer derivation: **≈118 GB** (236B params × 0.5 B/param FP4).

**Weight-broadcast topology + sync cost (this session, items 1+2)**:
disaggregated's cross-pool weight sync, single-link baseline = fan-in
(0.047 s, 8t-internal) + cross-pool (2.36 s, Jupiter @ 400 Gb/s) +
fan-out (0.344 s, Boardfly @ 19.2 Tb/s/chip, 7-hop bound for N_r>32) ≈
**2.75 s total**. Exceeds training's own per-step wall-clock by **~3.4×**
at R=8,192 (11% at R=65,536) — real, load-bearing cost for item 4's
synthesis, not a footnote. Faster architectures exist (parallel
per-device links) but are contingent on either accepting N_r=N_t=160
(collides with independent pool-sizing) or an unmodeled reshard cost
(HybridFlow's 70B-dense numbers are the anchor for how bad that could
get). Single-link ≈2.75 s is the working default. Full derivation,
including the Boardfly Group-size correction (32 chips, not 288 — a
WebFetch/HBM-capacity conflation caught mid-session), in `notes.md`.

## Not done, deliberately

Root `README.md` has not been touched — this repo's convention (see
`disagg_and_placement_notes.md`'s own history) is to update it once a
project reaches its own Phase 5 completion, not mid-project. rl-codesign
is only through Phase 2 of 5, so this is intentional, not an oversight.
