# Project Spec: RL Post-Training Codesign — Rollout vs. Training

**Continuity note:** project #6. Reuses decode's roofline formulas
(`decode_notes.md`), MoE's FLOPs/bytes and interconnect-topology reasoning
(`moe-routing-notes.md`), MLA's real cache formula (`disagg`'s own §2c.3),
and — lightly, at the end — #5's real measured kernel throughput. This is
also the first project with genuinely new derivation content none of the
prior five touched: training-side FLOPs/memory (only inference has been
derived so far), and staleness (a real axis with no prior analog).

**The real tension motivating this project**, sourced from current
literature (Phase 0 reading, not invented): every RL post-training step has
two phases — **rollout** (generate completions with the current policy —
pure inference, decode-heavy, your existing domain) and **training** (the
actual gradient update). Real systems split on how to allocate resources
between them, and it's a live, unresolved debate: HybridFlow keeps one pool
of chips and **reshards** parameters between rollout-optimal and
training-optimal layouts; RhymeRL/AReaL/StreamRL run **fully disaggregated**
pipelined rollout/train worker pools with asynchronous, deliberately-stale
updates; RLinf's own paper states the tradeoff explicitly — full
disaggregation wastes compute at small scale, full colocation stalls at
large scale, hence a hybrid. This project derives both ends of that real
spectrum, rather than picking one the way #5 picked one sparsity mechanism —
the comparison itself is the point, mirroring how disagg compared dense vs.
MoE rather than settling on one.

**Legend:** 🔧 = boilerplate/setup. 🧠 = your job.

---

## Phase 0 — Setup (🔧, two real 🧠 decisions)

### Reading (🔧)

- **Colocated/resharding**: HybridFlow (arXiv 2409.19256) — single/
  multi-controller hybrid, the 3D-HybridEngine's actual resharding mechanism
  (claimed zero memory redundancy — verify what that actually means
  mechanically, don't take the claim at face value). RLHFuse (course
  reading, read fresh — not covered in the search above).
- **Disaggregated/asynchronous**: RhymeRL (arXiv 2508.18588 — dedicated
  rollout/reward/train workers, streaming pipeline), AReaL (staleness-aware
  PPO variant — read specifically for how it *quantifies* acceptable
  staleness, direct input to Phase 4), StreamRL, BiDiRL (arXiv, hot-switch
  runtime + bidirectional scheduler — the actual hybrid end of the
  spectrum, real evidence resource idleness is a genuine problem on both
  sides, not just an inference-side concern).
- **Workload grounding**: DeepSeek R1 Technical Report — GRPO's real
  formulation (group-relative advantage, no critic), and whether R1 itself
  keeps a KL-penalty/reference model or drops it (DAPO-style variants often
  do — confirm which this project models, don't assume).

### 🧠 Decision 1: reference model — keep it or drop it

A frozen reference model (for a KL-divergence penalty against the original
policy) is a real third workload if kept — pure inference, prefill-shaped
over each generated response, no training cost, but real chip time and real
placement question (can it share chips with rollout, since both are
inference-shaped, or does it need its own pool?). Some real GRPO variants
drop it entirely. Decide after the DeepSeek R1 read, state the reasoning —
this changes how many distinct "roles" Phase 1 needs to account for.

### 🧠 Decision 2: precision split across the loop

TPU 8i's FP4 peak (10.1 PFLOPS, used throughout this repo) is realistic for
**rollout** — it's inference. It is not realistic for **training** — real
training runs at higher precision (BF16 is the standard baseline; some
systems use FP8 for parts of the pipeline). This means **weight sync isn't
just moving bytes, it's a real precision-conversion step** (training
precision → rollout precision), the same coupled-vs-decoupled question #5's
Phase 2 already built a framework for (§4.1) — decide now whether to reuse
that framework directly here or whether RL's version is different enough to
warrant fresh treatment (the direction of the conversion is reversed from
KV-cache quantization, which might matter). Source TPU 8i's real BF16 peak
FLOPs before Phase 2 — don't assume it's a simple multiple of the FP4
number.

**Checkpoint:** reference-model decision made and defended; precision split
decided; DeepSeek R1's real GRPO formulation (group size K, KL term or not)
confirmed from the primary source, not assumed.

---

## Phase 1 — Rollout-side roofline (🧠)

**Reuse directly:** decode's FLOPs/bytes formulas and MLA's real cache
formula (`576 elements/token/layer`, disagg §2c.3, DeepSeek-V2's own
hyperparameters) — rollout generation *is* decode-heavy inference, the
exact workload shape you've already derived twice.

**The genuinely new wrinkle**: GRPO's group sampling generates **K
completions per prompt** (real parameter — read DeepSeek R1's actual value,
don't guess), all sharing the same prompt prefix. This is a real,
derivable prefix-sharing opportunity — the exact mechanism the prefix-
caching idea from your own brainstorm would have covered, now showing up
naturally inside this project instead of needing its own one. Derive: how
much does K-way prefix sharing reduce the *effective* prefill cost of a
rollout batch (K completions, one shared prompt) versus K independent
requests? This is closed-form, not a new mechanism to build — a direct
extension of formulas you already have.

**Deliverable**: rollout-phase FLOPs/bytes/wall-clock per RL step, as a
function of (number of prompts, K, response length distribution) — your own
call on what response-length distribution to assume, grounded in something
real (R1's own reported generation lengths are a reasonable anchor) rather
than an arbitrary pick.

---

## Phase 2 — Training-side roofline (🧠, genuinely new territory)

**No formula to reuse here — this repo has only ever derived inference.**
Standard building blocks, real and sourceable, not to be reinvented from
scratch: forward+backward FLOPs (the well-established ~3× forward-FLOPs
rule — 1× forward, 2× backward — verify this holds for MoE's sparse
activation pattern specifically, don't assume the dense-model rule
transfers unchanged), optimizer state memory (Adam: real, standard
multiplier on parameter count for momentum+variance — source it, don't
guess the constant), activation memory during backprop (real, and unlike
rollout's KV cache, this scales with sequence length × batch in a way
that's worth deriving explicitly, not assuming it's negligible).

**The real, specific question GRPO changes**: no critic model means no
critic weights, no critic optimizer state, no critic activation memory —
quantify this saving explicitly against what a PPO-style critic would have
cost, using your own already-derived MoE/MLA parameter-count numbers. A
real, concrete number, not just "GRPO is simpler."

**Checkpoint:** training-step FLOPs/bytes/memory derived, GRPO's
critic-free saving quantified against a stated PPO baseline.

---

## Phase 3 — Two architectures: colocated-resharding vs. disaggregated-pipelined (🧠)

### Colocated (HybridFlow-style)

One pool of chips, switching between rollout-mode and training-mode
parallelism layouts. **No chip-ratio question here** — genuinely different
resource-allocation shape than every prior project in this repo (disagg,
MoE) — instead it's a **wall-clock split**: what fraction of each RL step's
time goes to rollout vs. training, and what does the resharding transition
itself cost (real time, not free — HybridFlow's "zero memory redundancy"
claim needs to be checked against whether it also means *zero time* cost,
which is a different, stronger claim).

### Disaggregated (RhymeRL/AReaL/StreamRL-style)

Reuses disagg's own chip-ratio methodology directly — service time →
throughput → ratio, the same chain disagg's Phase 2 built. **What's
different**: the "handoff" is weight broadcast, not KV cache. Weights are
vastly bigger than any KV-cache handoff this repo has derived (disagg's
dense handoff was 40 MiB/request; a full policy weight sync is the entire
model, GBs to tens of GBs). This needs its own real derivation: is it a
point-to-point transfer or a broadcast (all rollout workers need the same
update — reuse MoE's own mesh-vs-switch topology reasoning, since broadcast
to many workers is a genuinely different traffic pattern than either MoE's
dispatch or disagg's point-to-point handoff, and deserves its own
mesh/switch/tree-topology check rather than assuming one of the two prior
answers transfers unchanged).

**Precision conversion at the sync boundary** (per Phase 0 Decision 2): if
training runs BF16 and rollout runs FP4, weight sync needs a real
requantization step. Reuse #5's coupled-vs-decoupled framework here if
Decision 2 concluded it transfers; derive fresh if not.

### The comparison

Hand-derive, don't assume: under what conditions (model size, cluster size,
rollout:train FLOPs ratio from Phases 1–2) does one architecture win? RLinf's
own stated tradeoff (disaggregation wastes compute small-scale, colocation
stalls large-scale) is a real claim to check your own derivation against,
not to import unverified.

**Checkpoint:** both architectures' cost models derived; a stated,
mechanistically-explained answer for which wins under which conditions.

---

## Phase 4 — Staleness (🧠, a genuinely new axis, explicit scope boundary)

**Real and derivable, systems side**: asynchronous/disaggregated
architectures let rollout run ahead of training (generating with a
slightly-stale policy) to avoid idle time — AReaL and RhymeRL both do this
for real, measurable throughput gains. Derive the throughput-vs-staleness
tradeoff at the systems level: how much does allowing N steps of staleness
reduce synchronization wait time, given Phase 1–3's own throughput numbers.

**Explicitly out of scope, stated why**: whether a given amount of
staleness actually *hurts training quality* (sample efficiency, convergence)
is a real ML research question, not a systems-derivable one from first
principles — this repo's methodology (hand-derive FLOPs/bytes/roofline) has
no purchase on it. Read AReaL's own reported staleness tolerance as a real,
empirically-grounded anchor rather than trying to re-derive it yourself —
the same "use the real number, don't reinvent it" move as every prior
project's reference-reading phase.

---

## Phase 5 — Synthesis (🧠, includes the optional real-number substitution)

**Cross-project connections, stated explicitly:**
- Rollout:train resource split is the direct structural analog of disagg's
  prefill:decode chip ratio — same methodology, different roles.
- Weight-broadcast topology reuses MoE's mesh-vs-switch reasoning for a
  genuinely different traffic pattern (broadcast, not dispatch).
- The precision-conversion-at-sync question reuses #5's numerics framework.

**The optional close-the-loop step**: substitute #5's real measured
decode-kernel throughput (not idealized roofline) into Phase 1's rollout
cost, the same move #5's own Phase 4 Q1 already made once against disagg.
Check whether Phase 3's colocated-vs-disaggregated conclusion changes under
real vs. idealized rollout throughput. **Do this only if it's a genuine
one-line substitution using data you already have** — if it starts needing
new infrastructure (a new simulator, new kernel work) to make the numbers
comparable, that's the signal to cut it and flag it as a future thread
instead, the same discipline #5's own Decision 3 (MLA) and Q2 (abandoned)
already modeled.

---

## Note on scope

This project has more genuinely new content than any prior one (Phase 2's
training-side derivation, Phase 4's staleness axis) — resist the urge to
also chase a full ML-quality staleness model (explicitly out of scope,
Phase 4) or to build a new discrete-event simulator for Phase 5 (explicitly
optional, and only as a cheap substitution, not a rebuild). Two new axes is
already a real amount of breadth for one project; a third would repeat the
mistake this repo has caught and corrected multiple times before.

## Fallback

Phases 1–3 (both roofline derivations plus the colocated-vs-disaggregated
comparison) stand alone as a complete artifact — the real headline
question (which architecture wins, and why) is answered there. Phase 4
(staleness) and Phase 5 (synthesis/close-the-loop) are real additions, not
prerequisites for a complete project.
