# Project Spec: Speculative Decoding Codesign — Draft, Verify, Rollback

**Continuity note.** Reuses prefill's own compute/byte roofline formulas
(`prefill_notes.md`, project #1) for the verification-regime derivation,
decode's roofline formulas (project #1) for the draft phase, and disagg's
chip-ratio methodology (`disagg_and_placement`, project #3 — service time →
throughput → ratio) for draft/target placement, though the synchronous
coupling here breaks disagg's core async-independent-pool assumption and
has to be worked through fresh, not imported. #5's real Triton-kernel/
A100-hardware discipline (`numerics-and-sparse-attn`) carries over directly
for the verification-mask kernel. This is the **first project in the
portfolio with a non-monotonic KV cache** — every prior project (#3, #6)
assumed append-only growth; rejected draft tokens force real eviction, a
genuinely new mechanism.

**The one real fork this spec turns on, decided going in**: **tree-
structured** drafting (Medusa/EAGLE-2/SpecInfer/Sequoia-style branching
candidates) is the primary target, not chain-structured (the original
Leviathan/Chen single-candidate-sequence formulation). This is a real,
deliberate scope increase over the simpler option — tree-structured is
what makes the verification-mask kernel (item 3 below) genuinely novel
work (each candidate token attends only to its own ancestor path, not
sibling branches — a structurally different sparse-attention pattern from
anything #5 built) and what makes KV-cache rollback (item 4) a real design
problem rather than a one-line truncation. **Chain-structured is the
explicit fallback**, not a discarded option — see Fallback below.

**Legend:** 🔧 = boilerplate/setup. 🧠 = your job.

---

## Phase 0 — Setup (🔧, three real 🧠 decisions)

### Reading (🔧)

- **Leviathan et al. 2023** and **Chen et al. 2023** (DeepMind) — the
  original speculative sampling formulation. Read even though chain isn't
  the primary path: the correctness argument (draft-then-verify preserves
  the target's exact output distribution via rejection sampling) is the
  mechanism tree-structured methods extend, not replace.
- **SpecInfer** (Miao et al. 2023) — tree attention's origin: multiple
  draft models' outputs aggregated into a token tree, verified in one
  target-model pass.
- **Medusa** (Cai et al. 2024) — the concrete tree-attention mask
  mechanism (each row attends only to its own ancestors; a *fixed,
  regular* tree shape, chosen specifically because it lets the mask be
  precomputed rather than built fresh every step). Read the mask
  construction closely — this is the direct precedent for Phase 3.
- **EAGLE-2** (Li et al. 2024) — the dynamic-tree alternative: draft trees
  built per-step from draft-token confidence rather than a fixed shape.
  Read for *why* dynamic trees help (higher acceptance per verification
  call) and what it costs (real per-step tree-construction overhead) —
  input to Decision 3, not resolved yet.
- **Sequoia** (Chen et al.) — hardware-aware tree sizing/shaping; read for
  the real tradeoff between tree depth and verification cost, relevant to
  Phase 1's roofline.
- **vLLM's real paged-KV-cache spec-decode mechanics** (docs +
  `spec_decode` source) and **PipeInfer** (real per-cell KV metadata
  approach) — both real, current, working systems for the rollback
  question (Phase 4). Read as grounding, not as an answer to copy —
  vLLM's real mechanism is chain-oriented (free-on-rejection, promote-on-
  acceptance for one linear sequence); the tree case is genuinely harder
  and neither source solves it directly.
- **Draft & Verify** (self-speculative decoding paper) — real, measured
  acceptance-rate-vs-quantization data (bf16/fp8/fp4/nf4, acceptance rate
  flat at ≈0.91 across all four) — a real anchor for Phase 5, though from
  a *self-speculative* setup, not a standalone-draft-model one. Flag this
  now: the number needs re-verification for this project's different
  setup before being reused, not just cited.
- **Groq's real, current production spec-decode stack, and the Groq 3
  LPX architecture** (2026, paired with NVIDIA Rubin GPUs) — direct,
  current grounding for Phase 2, not a hypothetical: an SRAM-resident LPU
  generates drafts (fast specifically because SRAM residency makes
  drafting cheap), a GPU verifies. Real evidence that SRAM residency
  doesn't eliminate speculative decoding's value — it changes which side
  of the draft/verify split benefits, and can make verification itself
  nearly free (target weights also SRAM-resident). Read this closely; it's
  close to a live precedent for exactly the synchronous draft/verify
  placement question Phase 2 derives from scratch.
- **MatX One's real, confirmed hybrid SRAM+HBM architecture** (Feb 2026
  Series B announcement) — worth confirming explicitly, since it's the
  #1 target company: MatX is *not* SRAM-only. HBM specifically carries
  KV cache and long-context; weights get the SRAM-first treatment. The
  bandwidth tension Phase 1 derives is directly live for this specific
  target, not a stale GPU-era assumption.

### 🧠 Decision 1: tree-structured, chain as fallback (recorded, not
re-litigated)

Already decided above. Reasoning restated for the record: tree-structured
is what makes items 3 and 4 real novel work rather than straightforward
extensions of existing mechanisms; the added complexity is real and
acknowledged, which is why chain is kept as an explicit, fully-scoped
fallback rather than an implicit safety net.

### 🧠 Decision 2: draft-model pairing — standalone smaller model, not
self-speculation

Two real options: a **standalone smaller model** from the same family
(e.g., an 8B-class draft for the portfolio's existing Llama-3-70B/GQA
target — continuity with every prior project's workload) generating
candidates that get organized into a tree, versus **self-speculation**
(EAGLE-style: a lightweight head consuming the target's own hidden states,
one shared backbone). **Recommend standalone smaller model.** The direct
reason: self-speculation collapses item 2's chip-placement question (draft
and target are the same model, so there's no real "which chip does the
draft run on" decision left) — and that placement question is real,
named, portfolio-relevant content this spec exists partly to cover.
Self-speculation is also real ML training work (fitting a feature-level
head) this project has no reason to take on. Standalone model preserves
both the placement question and a clean drafting mechanism, at the cost
of not chasing EAGLE-2's specific SOTA acceptance numbers — a real,
stated tradeoff, not an oversight.

### 🧠 Decision 3: tree shape — start static, leave dynamic for the build
phase

Two real options, both read above: Medusa's **fixed, regular tree**
(precomputable mask, simpler) versus EAGLE-2's **dynamic, per-step tree**
(built from draft-confidence, real construction overhead, higher real
acceptance). **Don't resolve this now.** Start Phase 3 against the static
tree — it's the tractable first target and the one with a clean, citable
mask precedent. Whether a dynamic tree meaningfully changes the *kernel*
itself, or is mostly a host-side tree-construction cost sitting on top of
the same masking primitive, is a real, discoverable question — leave it
for Phase 3, per the explicit design principle for this project (don't
over-resolve on paper what building will actually answer).

**Checkpoint:** tree-vs-chain and draft-pairing decisions recorded with
real stated reasoning; tree-shape question explicitly deferred, not
resolved, going into Phase 1.

---

## Phase 1 — Verification-regime roofline (🧠)

**Reuse directly, don't re-derive:** prefill's own FLOPs/bytes/AI
formulas from `prefill_notes.md` — verification is a single forward pass
over a short, prefill-*shaped* sequence (item 1's framing), so the
formula machinery transfers; what's new is the shape it's evaluated at.

**The real new shape**: with tree-structured drafting, the sequence being
verified isn't K+1 tokens in a line — it's the **flattened tree**, whose
token count depends on branching factor and depth (Medusa's own worked
example: two heads × top-2 each = 4 candidate paths; a third head takes it
to 8). Derive the real flattened-tree token count as a function of depth
and branching factor for a couple of concrete, Medusa/EAGLE-2-sourced
tree shapes, then compute AI at that shape using the reused formulas.
Disagg already found prefill is only marginally compute-bound at
seq_len=512 (project #3) — at a flattened-tree length in the tens of
tokens, this is a real, currently undetermined question. Derive it, don't
assume the answer inherits from either #1's 8192-token prefill or #3's
512-token case.

**The genuinely new axis prior projects didn't have**: real serving
batches many concurrent verification requests simultaneously — this is
neither #1's single very-long sequence nor #3's batched single-token
decode; it's *batched short sequences*, a real new point in the roofline
space. Derive how batch size and tree size trade off for this specific
shape.

**Deliverable**: verification-step FLOPs/bytes/compute-or-memory-bound
determination, as a function of (tree depth, branching factor, batch
size) — a real, derived answer, not assumed from either prior project's
regime.

**A real second sub-question this roofline framing forces, worth making
explicit rather than assumed either way**: speculative decoding's benefit
on GPU-class hardware is usually attributed to one mechanism — amortizing
HBM bandwidth across more tokens per forward pass. But there's a second,
distinct mechanism underneath it: batch-1 decode leaves compute arrays
underutilized regardless of memory technology (tiny matmuls), and
verifying a flattened tree in one pass is a bigger matmul than one token
at a time, independent of where weights live. **Derive both mechanisms
separately** against this project's own roofline numbers, then check what
happens to each under an SRAM-resident regime (Groq's real ~230–500MB/
chip SRAM, or Cerebras's real 44GB on-die, no-HBM case): does the
bandwidth-amortization term vanish (plausible — SRAM bandwidth is
real-world ~40-150 TB/s+ versus HBM's ~3-22 TB/s, per the reading above)
while the compute-utilization term survives, or does something else
happen? A real, derived split, not an assumed "SRAM makes this moot" or
"SRAM doesn't change anything" — both are unearned until this phase
actually runs the numbers. MatX's own confirmed hybrid SRAM+HBM design
means this project's primary target doesn't even require resolving the
fully-SRAM edge case to stay relevant — but Groq and Cerebras are real,
current, worth deriving explicitly rather than leaving as a hand-wave.

---

## Phase 2 — Draft/target chip placement under synchronous coupling (🧠)

**Structurally like disagg's own chip-ratio chain** (service time →
throughput → ratio), but the core assumption disagg's Phase 2 relied on —
independent, asynchronously-pipelined pools — doesn't transfer. **Draft
and verify are synchronously coupled**: verification for round N can't
start until drafting for round N has fully finished (K sequential
autoregressive draft-model steps, reusing decode's own roofline formulas
at the smaller draft model's parameter count), and drafting for round N+1
can't start until round N's verification resolves which tokens survived.

**Derive the real round-latency chain**: sum of K sequential draft-model
decode steps (draft chip) + handoff cost (draft candidates → target chip)
+ one flattened-tree verification pass (Phase 1's shape, target chip).
Then the real open question: does putting draft and target on **separate**
chips still make sense under this synchronous constraint, or does
separation just add pure serial round-trip latency with none of disagg's
real pipelining upside (since there's no independent, overlappable work to
hide it behind, unlike disagg's genuinely async prefill/decode pools)? A
real, derived answer — don't assume co-location wins by default just
because the coupling sounds unfavorable to separation, and don't assume
separation wins just because #3 established chip-ratio thinking as the
default move.

**Explicitly left open for the build phase** (per the design principle for
this project): whether round N+1's *draft* can start speculatively before
round N's verification fully resolves, given only some later requests in
a batch are still pending — a real scheduling-level question, not a
roofline-level one, and a legitimate build-phase discovery rather than
something to force an answer to here.

**Checkpoint:** round-latency model derived; a real, stated answer on
whether synchronous coupling changes disagg's separate-chip conclusion —
possibly "it depends on X," stated concretely, not hedged vaguely.

---

## Phase 3 — Real Triton kernel: tree-attention verification mask (🔧 build, 🧠 design)

**The real novel component.** None of #5's kernels (sliding-window +
attention-sink, spatially contiguous masks) touched an ancestor-path mask
— tree-attention is structurally different: each candidate token attends
to its own root-to-node path through the tree and nothing else, including
not its sibling branches, even though all candidates ride through one
flattened sequence and one forward pass (the real mechanism Medusa's own
figure documents).

**Build**: a real Triton kernel implementing this mask for the **static**
tree shape from Decision 3, matching #5's own real-hardware discipline
(real A100, not simulated) — verified against a golden PyTorch reference
across several concrete tree shapes, not one test case.

**The real, deferred question from Decision 3, resolved here by
building, not by assuming**: does a dynamic (EAGLE-2-style) tree actually
require a structurally different kernel, or is the masking primitive the
same and the real cost is entirely in host-side tree construction sitting
on top of it? Attempt a dynamic-tree variant after the static one works;
report the real answer, including if it's "the kernel barely changes" —
that's a legitimate, useful finding, not a disappointing one.

**Deliverable**: a real, verified, real-hardware-validated Triton kernel
for tree-attention verification masking, plus a stated, derived answer on
static-vs-dynamic tree cost.

---

## Phase 4 — KV-cache rollback (🧠 design, deliberately left open going in)

**The first non-monotonic cache in this portfolio.** #3 and #6 both
assumed append-only growth; this project's entire premise is that rejected
draft tokens' cache entries need real eviction. **Chain-structured
rollback is close to trivial** (a single pointer truncates backward — the
real mechanism vLLM's own paged-block free-on-rejection already does).
**Tree-structured rollback is a real, harder problem**: after
verification, one path through the tree is accepted (the shared prefix
plus one winning branch); every other candidate branch's cache entries
need eviction, and they are *not contiguous* with each other in the
straightforward case.

**Real design question, grounded in the reading, not pre-solved**: a
shared-prefix-block plus per-branch delta-block layout (candidate
branches kept as separate, cheaply-discardable allocations until one is
accepted) versus simpler full per-branch duplication (wasteful but
trivial). Reuse vLLM's real paged-block mechanics and PipeInfer's real
per-cell metadata approach as grounding for whichever layout gets chosen,
rather than inventing a scheme from nothing.

**Worth naming explicitly, not chasing**: the shared-prefix-block idea is
a real, direct structural cousin of the still-open fleet-wide
prefix-caching idea on the roadmap (`03_skills_gaps_and_roadmap.md`, idea
3) — this project's version is single-request, single-round, and much
narrower. Name the connection; don't let it pull this phase into building
the fleet-wide version.

**This phase is intentionally underspecified here.** Per the explicit
design principle for this project (stated by the person directly after
#8's retrospective): the rollback mechanism is exactly the kind of real,
discoverable question that should surface during the build, not get fully
resolved on paper first the way #8 over-resolved its epilogue circuit.

**Checkpoint:** a real, working eviction mechanism for the accepted-path/
rejected-branches split, with the layout choice stated and defended after
building — not before.

---

## Phase 5 — Draft-model precision (🧠, optional, cheap)

**A real, cheaply derivable connector to #5/#8**, not a mandatory phase.
Since draft errors get corrected by verification regardless, does the
draft model tolerate more aggressive quantization than the target model
ever could? The Draft & Verify paper's real bf16/fp8/fp4/nf4 table
(acceptance rate flat at ≈0.91 across all four precisions) is a strong
real anchor — **but it's from a self-speculative setup** (Decision 2's
rejected option), not this project's standalone-draft-model setup. Per
the portfolio's single most repeated discipline: re-verify whether that
number's *mechanism* (verification catches errors regardless of their
source) transfers to a genuinely separate, smaller model, rather than
just citing the number as-is.

**If it transfers cleanly**: substitute a more aggressive draft precision
into Phase 2's round-latency model and check whether it changes Phase 2's
separate-vs-colocated chip conclusion — a one-line-ish substitution, the
same move #6's own Phase 5 made against #5's real kernel numbers. **Cut
it if it starts requiring new infrastructure** (a new kernel, a new
quantization scheme) to make the comparison real — that's the signal to
flag it as a future thread instead, not to build it out here.

---

## Phase 6 — Synthesis (🧠, earned by a real open question, not a default closer)

Two real things to close out, both genuinely undetermined until Phases
1–5 actually run:

1. **Was tree-structured worth it over chain?** Decision 1's real,
   acknowledged bet was added complexity (Phases 3–4) in exchange for more
   novel, more current-SOTA-aligned work. State the real answer,
   mechanistically — did the tree-attention kernel and branch-pruning
   rollback surface genuinely new findings the chain fallback wouldn't
   have, or did the added complexity turn out disproportionate to what it
   bought? Either real answer is a legitimate finding.
2. **Direct relevance, stated plainly**: speculative decoding sits
   squarely in inference-serving-architecture territory relevant to every
   target company on the list (MatX, Etched, OpenAI, Anthropic) — say
   concretely what this project demonstrates (a real novel kernel, a real
   non-monotonic cache design, a real synchronous-placement derivation)
   and what it doesn't (no draft-model training, no systematic tree-shape
   search — see Note on Scope).

---

## Note on scope

Resist: (1) training anything (an EAGLE-style feature head, a Medusa-style
multi-head target) — Decision 2 already ruled this out; stay a systems/
architecture project, not an ML training one. (2) A systematic,
automated search over tree shapes/branching factors/hyperparameters —
real, legitimate future work, but a tuning project, not this one. (3)
Building the fleet-wide prefix-caching system Phase 4's shared-prefix-block
idea connects to — name the connection, defer the project. (4) Evaluating
whether draft-model quantization or staleness actually hurts *output
quality* — a real ML-research question with no purchase from this
portfolio's roofline-and-real-hardware methodology, the same boundary #6's
own Phase 4 drew around training-quality effects of staleness.

**A fifth, explicitly flagged rather than added**: whether tree-attention
masking or KV-rollback eviction warrant dedicated hardware support (a
mask-generation circuit, a hardware-level eviction mechanism), the way #8
found a real gap for dynamic exponent computation. **Not added here** —
ISA design and arithmetic-circuit design are already 🟢 in the skills
table (#7, #8), so this wouldn't close a gap the way #8's did, and #9
already carries two genuinely new axes (the tree-mask kernel, non-
monotonic rollback); a third repeats the mistake #6 explicitly declined to
make. Whether a real, narrow hardware gap even exists here (versus fully
decomposing into masking logic #1/#7's hardware already has) is itself
unconfirmed — a small, standalone feasibility check, #8-style, would need
to happen before this becomes a real project candidate, not as part of
#9.

## Fallback

**Two real levels.** First, phase-wise: Phases 1–2 (verification-regime
roofline, synchronous placement) stand alone as a complete artifact
regardless of chain-vs-tree, since both derivations only need *a* short
verification shape, not specifically a tree. Phase 3 (the kernel) is a
real, separate deliverable once built. Phase 4 (rollback) is the
highest-risk, most novel phase, consistent with it being the one
deliberately left open going in.

**Second, and specific to this project's Decision 1**: if tree-structured
drafting proves too complex to close out in Phases 3–4 — the real risk
Decision 1 explicitly acknowledged — **degrade to chain-structured**
rather than stall. Chain is not a lesser version of this project; it's
the original, real, well-precedented formulation (Leviathan/Chen), with a
close-to-trivial mask (standard causal, no ancestor-path logic needed) and
a close-to-trivial rollback (single-pointer truncation, matching vLLM's
own real mechanism directly). A chain-structured version of Phases 1–4
is still a complete, real, shippable project — this is a deliberate,
named safety net, not a failure state.
