# Disaggregated Serving + Placement — Workload → Silicon (Phases 0–5, Complete)

Third of three sequential codesign studies in this repo, one level up from both:
attention ([`prefill_notes.md`](prefill_notes.md)/[`decode_notes.md`](decode_notes.md))
derived single-chip microarchitecture; MoE routing ([`moe_routing_notes.md`](moe_routing_notes.md))
derived multi-chip interconnect topology. This project is the memory-hierarchy layer
underneath both: given a serving system split into prefill and decode pools, what
actually lives in SRAM vs. HBM vs. gets transferred, and how does data move as
prefill hands off to decode. No off-the-shelf tool fits this question, so Phase 4
hand-builds a discrete-event simulator from scratch (SimPy) rather than reusing
Timeloop/ASTRA-sim. Supersedes two working-log documents (`spec.md`→`spec_v2.md`,
`notes.md`) kept live during derivation, mirroring how the sibling projects were
each refactored from a live log into a finished writeup — self-corrections and real
reversals are kept where they carry mechanistic signal, pure process narration is cut.

---

## 0. Workload, Chip, and Simulator Design

**Chip: TPU 8i, homogeneous across prefill *and* decode.** Reached after trying,
then deliberately reversing, a heterogeneous TPU 8i (prefill) / Groq (decode)
split — Groq's SRAM-only architecture (500MB/chip, no HBM at all) looked like
the natural fit for decode's ~30–240× memory-bound roofline margin
(`decode_notes.md`), but Llama-3-70B's 70GB of weights can't fit Groq's per-chip
SRAM at all — a real deployment needs ~140+ pipelined chips, which breaks the
"one decode machine = one atomic unit" abstraction the whole simulator design
depends on. Reworking that abstraction would smuggle a chip-parallelism problem
into a disaggregation project — the exact shallow-breadth failure mode this
repo's own scope philosophy exists to avoid. Flagged as legitimate material for
a future, focused fourth project; not chased here. **Precision: FP4 uniform**,
both pools, both TPU 8i's native format — no requantization boundary at handoff.

**Simulator design (the one genuine 🧠 twist in an otherwise 🔧 setup phase):**
discrete-event, not closed-form — queue wait time, hardware utilization, and
memory-pressure-driven eviction only exist once requests contend for shared
resources over time, the same reason the MoE project reached for ASTRA-sim once
contention became the question.

- **Entities**: prefill machines, decode machines, one finite intermediate KV pool.
- **Handoff**: Mooncake-style — prefill machine frees *immediately* on ship, KV
  cache lands in a separate pool (not DistServe's pull-from-prefill-GPU-memory
  model, which couples prefill throughput to decode's pace).
- **Decode concurrency**: N *concurrently active* requests per machine,
  memory-bounded (weights + Σ active KV caches ≤ HBM) — not full Orca/vLLM-style
  iteration-level scheduling (real complexity serving a different question) and
  not one-request-per-machine (would gut the entire memory-pressure question this
  project exists to study).
- **Pool-full policy (placeholder)**: block until space frees — simplest
  structurally-correct option, deliberately not solving real eviction yet
  (Phase 3's job).
- **Arrival/request-shape model**: Poisson arrivals (rate λ as the sweep
  parameter, matching DistServe); lognormal request shape anchored to
  **DistServe's own reported benchmark values (input≈512, output≈64 tokens)**,
  not raw ShareGPT corpus stats — chosen for Phase 4 comparability against
  DistServe's own reported numbers over strict realism.

**Reference reading** (Phase 0 🔧, banked as Phase 4's sanity-check targets):
**PagedAttention** (2–4× throughput, ~4× effective batch, fragmentation cut
60–80%→10–15%); **DistServe** (colocation interference: one prefill request in
a decode batch inflates batch time 60ms→200ms; KV transfer <0.1% of latency;
2.0–3.41× throughput gains); **Mooncake** (separate KVCache pool decouples
retention from compute-memory pressure; up to 525% throughput at long context).

---

## 1. Memory Hierarchy Characterization (Llama-3-70B, dense)

**Weights**: 35 GB at FP4 (70B × 0.5B/param), comfortably inside TPU 8i's 288GB
HBM — no sharding needed, unlike the rejected Groq design, confirming the
homogeneous-chip call was the right practical tradeoff.

**KV-cache growth curve**: `bytes/token = 2(K+V) × precision × d_head ×
n_kv_heads × n_layers = 2×0.5×128×8×80 = 81,920 bytes/token`, linear in context
length, GQA already folded in via `n_kv_heads=8`.

**Aggregate capacity → N**: after weights and a 10–15% PagedAttention-style
fragmentation reserve (activations ruled out as an HBM cost — they live in
on-chip SRAM under the fused execution model both sibling projects already
established), usable KV budget ≈215–228 GB. Against a hard per-request cap of
**8,192 tokens** (Llama-3's own native context, reused a third time across this
repo for comparability): **N = 320–339 concurrent max-length requests/chip**,
safe by construction. At the *average* request length (~576 tokens,
DistServe-anchored) instead of the worst-case cap, the same budget supports
**~4,558 concurrent requests — ~14× more** — quantifying, not yet resolving,
what's at stake in the hard-cap-vs-dynamic-admission fork carried to Phase 3.

**The core tension**: weights are fixed-size, known the instant a model is
chosen, never revisited — never a lever in the placement story. **KV cache is
the only elastic resource in the entire system**; every placement/eviction
question from here on is about KV cache, never weights.

---

## 2. Disaggregation Chip Ratio — Dense (Llama-3-70B)

**Reframe**: the compute/memory-bound *label* doesn't determine a chip ratio by
itself — DistServe profiles each phase's own goodput and solves an allocation,
no closed form. What roofline supplies is the ingredient for **service time**
(`max(FLOPs/peak_compute, bytes/peak_bandwidth)`), which converts to throughput,
which converts to a ratio.

**Reused-workload corrections, not just reused formulas**: `prefill_notes.md`/
`decode_notes.md`'s FLOPs/bytes were derived at this project's fixed
`seq_len=8192` — not automatically valid for this project's own DistServe-shaped
request model. Recomputed at **prefill seq_len=512** (prefill flips from
confidently compute-bound, ~17–25×, to only **marginally** compute-bound,
~1.55×) and **decode avg context=544** (time-averaged over a request's growth
from 512→576 tokens).

**The FFN gap — a first-order omission, not rounding error**: both sibling
projects are explicitly SDPA-only. Neither ever derived Llama-3-70B's FFN
compute. Adding it (`d_ff=28,672`, SwiGLU, sourced via HF config) revealed FFN
dominates **both** phases (~99% prefill, ~95% decode at batch=32) — attention
alone was a rounding error.

**A ratio that looked backwards, investigated and resolved**: at batch=32
throughout, the ratio came out **0.84** (needing *more* decode chips than
prefill — backwards from DistServe's own reported direction). Root cause,
confirmed mechanistically: **FFN weight bytes are batch-invariant** (loaded
once, amortized across the batch) while attention's are not — batch=32 badly
under-amortized decode's 352MB FFN weight load, when Phase 1 had already
derived a realistic concurrency ceiling an order of magnitude higher. Recomputed
at **N=320**: FFN's decode regime flips from memory-bound (9.2× at batch=32) to
marginally compute-bound (~1.08×, crossover at N≈296) — a **6.6× throughput
jump**. Ratio corrects to **~5.50:1**, matching DistServe's direction.

**QKVO correction — the SDPA-only gap fixed, supersedes the above**: SDPA-only
was the right scope for the *microarchitecture* question the sibling projects
asked, silently incomplete once reused here for *throughput*. QKVO projection
FLOPs/bytes are exactly **21.4% of FFN's magnitude**, weight-stationary,
same amortization treatment. Adding them: prefill throughput 172.9→**142.70
req/s/chip** (−17.5%), decode 951.6→**830.59 req/s/chip** (−12.7%). **Corrected,
authoritative ratio: ~5.82:1.** Moved only +5.8% despite a real 21.4% local
correction — an Amdahl's-Law mechanism (`prefill_notes.md`'s own named lesson,
recurring for a different lever): QKVO's local magnitude is fixed, but FFN's
*share* of each phase's total differs (98.8% prefill vs. 68% decode), so the
same local addition lands harder on prefill.

**Key mechanism, stated generally**: *the more severely memory-bound a workload
is, the more upside batching creates*, because there's more idle compute to
convert into throughput via amortization. "Decode is hard per-request" and
"decode is cheap to scale via batching" are the same underlying property, not
opposing ones — decode's memory-bound-ness *is* the idle-compute headroom that
N=320 later cashes in.

---

## 2b. MoE Chip Ratio (DeepSeek-V2, 8-device EP group)

**Deployment model — tried full replication, reversed to sharding.** First
choice: every chip holds the full 162-expert table (117.2GB, fits TPU 8i's
288GB HBM — no capacity-forcing function the way Groq had). This made MoE's
FFN bytes/layer come out **5.42× larger than dense's** at realistic batch
sizes — a device's local pool saturates toward the *global* 162-expert
population almost immediately (past dense's flat number by N≈9). **Reversed
after two real findings**: (1) this deployment choice is not cosmetic — it
determines whether a compute-bound crossover exists at all, and (2) full
replication structurally eliminates Phase 3's hot-expert-residency question
(if every chip holds every expert, there's no placement decision left) —
the same "eliminates the question this project exists to study" failure mode
already caught once for the intermediate KV pool. Reversed to **expert-parallel
sharding, 8-device EP group** (reusing `moe_routing_notes.md`'s own topology —
"one decode machine" redefined as one 8-chip group for the MoE leg only).

**Per-device bytes moved, decomposed**: at N=640/EP-group (see grounding
below), per-device FFN bytes/layer = **258.05 MB (0.73× dense)** — the
expected sparse-routing direction, restored by sharding. But decomposing it
revealed **routing sparsity itself contributes almost nothing** (a further
0.41 percentage points beyond dense's baseline): **26.3pp of the "smaller than
dense" gap is just the sharding-topology floor** (20 narrow experts summing to
less than dense's one wide FFN block — a fact about relative dimensions,
unrelated to sparsity), because at this batch size the local 22-expert shard
is already 99.4% saturated (coupon-collector dynamics: 640×6=3,840 dispatch
events against a 20-slot local pool). The genuinely sparsity-driven regime
only exists at small N (0.09×/0.34× dense at N=1/32).

**N=640 itself required a real correction**: the inherited T=8,192 (project
#3's own batch, silently borrowed) was never grounded for DeepSeek-V2's real
capacity. Checked via HF config: real shipped context is **163,840** (YaRN
RoPE-scaled), not 8,192. Regrounded via Phase 1's own methodology → **N=640/EP-group
(80/device)** at a 15% reserve.

**MLA decode attention — derived fresh, not reusable from either sibling
project**: pulled real equations from the DeepSeek-V2 paper (arXiv 2405.04434,
absorbed-inference formulation, the real deployed path — "we do not need to
compute keys and values out for attention"). MLA carries a large **fixed**
per-token FLOPs cost (432.67M, dominated by the absorbed-query and
output-projection terms) that vanilla GQA never had — making **attention
genuinely non-negligible for MoE** (~13% of decode service time, vs. <1–5%
for dense): MLA trades bytes for FLOPs, and that trade is what makes decode
attention compute-bound (3.17×) rather than a rounding error.

**Regime crossover — the third of three flagged possible outcomes**: FFN's
bytes saturate at a hard ceiling (~259.5MB, N-independent past N≈232) while
FLOPs grow linearly forever — a compute-bound crossover exists *mathematically*
at N≈6,459 system-wide. But DeepSeek-V2's real HBM-capacity ceiling is
N≈640–680 — an order of magnitude short. **FFN never reaches compute-bound at
any batch size this model could actually run at real memory capacity** —
memory-bound *in practice* (blocked by a capacity ceiling), a meaningfully
different flavor than dense's imbalance-*proof* structural floor from the MoE
routing project. Mechanistically, MoE's crossover sits ~2.7× higher than
dense's (N≈807/device vs. 296) because sparsity cuts compute (0.27× dense's
FLOPs/token) far harder than it cuts fixed bytes (0.74× dense's) —
`bytes_ratio/flops_ratio ≈ 2.75×` predicts the observed `807/296 ≈ 2.73×`
almost exactly.

**Final ratio, real DeepSeek-V2 deployment**: decode 601.55 req/s/chip,
prefill 457.98 req/s/chip → **~1.31:1** — dramatically more balanced than
dense's 5.82:1.

**Sensitivity check — ~1.31:1 is mostly a context-length artifact, not an
architecture effect.** Llama-3's real cap is 8,192; DeepSeek-V2's real cap is
163,840 — an uncontrolled confound between the two models' architecture *and*
their deployment context length. Recomputed MoE at the **matched** 8,192 cap
(N/device=1,608, far above dense's 320 since MLA's cache is 4.74× more
compact/token): decode 2,735.03 req/s/chip, prefill unchanged at 457.98 →
**~5.97:1 — essentially identical to dense's 5.82:1.** Sparsity + MLA barely
move the prefill:decode *balance* once context length is held fixed; what
they change dramatically is **absolute throughput on both ends** (roughly
proportionally), which is exactly why the ratio itself barely moves.

**Isolating the two architecture changes** (a GQA/MoE-FFN hybrid, since
DeepSeek-V2's own MLA head dims don't transplant a native GQA config
directly): dense-GQA-dense-FFN (5.82) → hybrid-GQA-MoE-FFN (**4.60**, FFN
sparsity alone, −21%) → MoE-MLA-MoE-FFN (**5.97**, MLA vs. GQA alone, the
other direction). **Two real, opposing effects that mostly offset — not one
neutral effect.** The closeness of 5.82 and 5.97 (within 2.6%) is not itself
fundamental: it's a coincidence of where the matched 8,192 context happens to
sit. MLA is a **context-length-conditional trade**, not a universal
optimization — it loses to GQA by ~9% at 8,192 (compute term now binds:
0.0578µs vs. GQA's 0.0128µs) but wins by **2.25×** at DeepSeek-V2's real
163,840 (bytes term dominates at long context). The real-cap 1.31:1 result is
the proof the near-cancellation isn't structural — stretch context length
toward where MLA was actually designed for, and the two effects stop
canceling entirely.

**Key Findings, Phase 2b**:
1. Deployment model (replication vs. sharding) determines whether MoE's
   bandwidth-sparsity story is real (5.42× worse than dense under replication;
   0.70–0.73× better under sharding) — same underlying mechanism, opposite
   headline conclusion, purely from an architectural choice with no
   capacity-forcing function either way.
2. At realistic batch sizes, per-device expert coverage saturates almost
   completely — "sparse routing saves bandwidth" only holds at small N; at
   real N, the "MoE beats dense" result is almost entirely the sharding
   topology floor, not sparsity doing work.
3. Reused numbers need re-verification for the *new* context every time, not
   just formula reuse — caught three real scope/context mismatches this phase
   alone (ungrounded T=8,192, the SDPA-only-to-throughput reuse gap, the
   borrowed-vs-real DeepSeek-V2 context length).
4. MLA is a genuine FLOPs-for-bytes trade, not a strict win — which side
   binds depends entirely on context length, the single most important
   caveat carried into Phase 5's synthesis.

---

## 2c. KV Handoff Mechanism

**Transfer-cost formula, corrected before computing anything**: at the instant
handoff fires, decode hasn't run a step yet — only *input*-token KV cache
exists to ship (output tokens' cache is generated in place, post-handoff, and
never transferred). `bytes = avg_input_tokens × bytes/token`.

**Dense**: 512 × 81,920 = **40 MiB/request**.
**MoE (MLA)**: KV-cache formula pulled fresh from the DeepSeek-V2 paper's own
Table 1 (`(d_c+d_h^R)×n_layers = 34,560 elements/token`, one compressed latent,
not a `2×(K+V)` shape) — **8.4375 MiB/request, 4.74× smaller than dense.**
Deliberately uses `avg_input=512` (dense's own average) for both, to keep the
architecture comparison apples-to-apples rather than DeepSeek-V2's real
(likely longer) deployed prompt length — flagged explicitly, not silently
assumed representative.

**Fabric: Boardfly**, reused from the MoE project — this project has never
claimed Google validated disaggregation specifically on TPU 8i, only that the
fabric itself is real (a distinction that matters for how much the reuse is
actually asserting).

**A real regime difference from MoE dispatch on the same fabric**: dispatch
traffic (2,560-byte payloads) is **latency-dominated** — transfer time ~1ns,
dwarfed by hundreds-of-ns hop latency. KV handoff is **bandwidth-dominated** —
17.48µs (dense) / 3.69µs (MoE) at Boardfly's 2.4TB/s, both far exceeding hop
latency, because the payload is ~16,000× bigger. Dense clears this with huge
margin; MoE's smaller payload sits close enough (~4 hops' worth) that physical
placement of paired chips is a real, non-ignorable lever for MoE specifically.

**Pool placement: locality-aware clustering**, not a pod-wide half/half split
— a naive split risks the worst-case far-apart pair, directly bad given the
hop-count sensitivity above. Small, repeated prefill:decode clusters (e.g. 2
boards/8 chips ≈ 6:1) instead, deliberately sized to the **matched-cap**
architecture ratio (5.97:1 MoE, 5.82:1 dense) rather than the real-cap
1.31:1 — consistent with that number already being flagged as a
context-length artifact, not the architecture comparison this project treats
as primary. No published TPU 8i ICI latency figure exists (Google states only
relative diameter reductions) — swept a plausible range (300–936.25ns/hop, the
same move the MoE project made for its own unpublished latency figure) rather
than inventing one number.

**DistServe sanity check**: handoff cost is **0.29–0.34% (dense) / 0.040–0.071%
(MoE)** of a single decode step. MoE clears DistServe's own `<0.1%-of-latency`
claim outright; dense sits modestly above it, but against a deliberately
*stricter* single-step denominator than DistServe's own full-multi-step-request
one — not read as a real discrepancy. **Mechanism worth keeping precise**:
transfer cost being cheap isn't *why* disaggregation helps — DistServe's own
finding is that colocation hurts via **interference** (one prefill request in
a decode batch: 60ms→200ms). Cheap transfer is what keeps physical separation
from *giving back* that interference-avoidance win, not the reason to
disaggregate in the first place. Two distinct claims, easy to conflate.

---

## 3. KV/Weight Placement Policy

**Cap behavior: hard stop, not compaction/sliding-window.** Not really a new
decision — already the assumption baked into N=320 since Phase 1. Matches this
project's own modeled workload (chatbot-shaped, ~576-token average, ~14× below
the 8,192 cap) rather than the unbounded-session shape (coding agents) that
compaction/attention-sink (StreamingLLM, real mechanism: softmax normalization
needs a place to route "excess" mass once early tokens are evicted — a real,
well-published technique, just for a different workload this project didn't
model) actually solves.

**Admission: hard-cap (N=320) kept over dynamic (N≈4,558), sensitivity
computed anyway.** Result: throughput identical to 3 significant figures
(830.5 vs. 830.59 req/s/chip) — N=320 already sits past FFN/QKVO's
compute-bound crossover (~N=296), so `throughput = N/(layers×combined/layer)`
cancels N entirely, a flat asymptote. Dynamic admission's real benefit is
lower *admission queueing time*, invisible to throughput — quantifying that
needed Phase 4's real simulator.

**Intermediate-pool eviction: block-until-space-frees kept, real cost
flagged.** This re-couples prefill's pace to decode's under sustained
backpressure — partially undoing Mooncake's own decoupling rationale.
Kept anyway (how often it actually binds is unknown without Phase 4;
building real tiered CPU/SSD offload is a genuine new-memory-tier scope
expansion), but stated honestly rather than silently accepted.

**KV-cache quantization (KIVI, 2-bit) and CPU-offload: declined.** Both real
and well-published, but declined for stated reasons: quantization would
retroactively ripple through *every* number derived since Phase 1 (a far
bigger blast radius than the admission-policy fork), and reintroduces the
same requantization-boundary complexity already declined once in the Groq/FP4-FP8
reversal. Even if adopted, the same compute-bound-asymptote argument as
dynamic admission means it likely wouldn't move throughput or the chip ratio
either — noted, not recomputed.

**Hot-expert SRAM residency — the open thread `spec_v2.md`'s Phase 2b section
flagged, closed here.** Under EP-sharding, each device's local shard is just
**22 experts (259.5MB)** — comfortably fits TPU 8i's 384MB SRAM (124.5MB
headroom), *not* a hot/cold ranking problem after all. The naive capacity
check almost wasn't sufficient on its own: a naive materialize-the-whole-batch
estimate of transient attention compute state (S/P intermediates) came out to
56–112MB, leaving the fit **genuinely tight, not obviously clean** (12.5MB
margin in the conservative case). **Resolved by checking real FlashAttention
kernel behavior instead of guessing**: real fused kernels tile K/V and reuse
one small, fixed SRAM buffer sequentially — footprint bounded by *tile size*,
not by `N×seq_k`. The naive tight-fit finding was an artifact of an incorrect
execution model, not a real constraint — confirmed qualitatively from the
FlashAttention paper directly, though an exact TPU-8i-specific tile-size
number was never extracted (flagged as a real, if minor, sourcing gap).

**Quantifying the payoff — depends entirely on which N.** At the *matched*
8,192 cap, residency changes nothing (FFN was already compute-bound there —
large N had already amortized the fixed memory cost past its crossover). At
the **real** 163,840 cap, residency is decisive: FFN's service time collapses
~10× (was memory-bound 10.04×), producing a **~4.5× throughput jump**
(601.55→2,735 req/s/chip). The "2,735 ≈ 2,735.03" match against the
matched-cap+streaming number is not a coincidence — both reach the identical
compute-bound throughput ceiling via different routes (removing a fixed cost
vs. brute-force batching past the crossover), the same asymptote mechanism
the admission-policy sensitivity check hit independently. **Chip-ratio
consequence: real-cap ratio recovers from 1.31:1 to ~5.97:1** — residency
closes almost the entire gap to the matched-cap architecture comparison.

---

## 4. Validation — Discrete-Event Simulator

Dense leg only (Llama-3-70B, hard-cap admission matching §3's actual chosen
policy) — MoE and a from-scratch dynamic-admission/eviction model are
legitimate follow-on passes, not built here. From-scratch SimPy build
(`disagg-and-placement-sim/`, no prior Python convention in this repo) —
**29 prefill : 5 decode machines** (matching the 5.82:1 ratio), one shared
`simpy.Container`-backed intermediate pool.

**Prefill and decode needed genuinely different batching mechanisms, a direct
consequence of §2's compute/memory-bound distinction, not an arbitrary
implementation choice.** Prefill: grabs whatever's queued up to a 32-batch
cap and processes it immediately (no batch-invariant fixed cost, so smaller
batches just take proportionally less time). Decode: each admitted request
runs as its own independent process, reading the machine's *live* occupancy
and recomputing service time fresh per token step — a state-dependent-rate
queue, because decode's FFN/QKVO terms have a real batch-invariant weight-load
floor below the ~296-token crossover.

**Three real bugs caught before any result was trusted** (kept on record):
(1) the verbal design brief describing prefill attention as "memory-bound
throughout" was wrong — a planning sub-agent caught it's actually marginally
*compute*-bound (~1.55×) at the real seq_len=512 operating point, independently
re-verified bit-exact against this project's own numbers; (2) a `simpy.Resource`
race condition where two requests dispatched within one pass could both see
stale occupancy and over-admit past the 320 cap — fixed with a synchronous
integer counter; (3) a missing pool-release call that would have made the pool
monotonically fill and never drain. All caught by validation discipline (bit-exact
regression tests against Phase 2's closed-form numbers, a deliberate
degenerate-case control test, unit tests against the real dispatcher), not
inspection alone.

**Finding 1 — the intermediate pool is drastically over-provisioned at its
original sizing.** The "one round" anchor (worst-case simultaneous full-batch
completion across all 29 prefill machines, ≈36.25 GiB) showed **exactly zero
contention across 2M+ requests**, confirmed real — not a wiring bug — via a
deliberate tiny-pool (84MB) control test that *did* nearly deadlock the
system. The real threshold, refined across two passes at increasing
statistical power, sits far below the original sizing logic: 200MB is total
collapse; 500MB is unstable/bimodal (not a graceful degradation — either
avoids the constraint entirely or snowballs once hit); 1–2GB carries a
small-but-real risk (0.77–1.34% of requests see pool wait) only once arrival
rate exceeds the system's real ceiling; **the realistic 36.25GiB default has
essentially unlimited headroom.**

**Finding 2 — dynamic admission provides zero measurable benefit over
hard-cap**, not just on throughput (§3's finding) but on queueing latency
either (this phase's unique contribution) — matching to 5–6 decimal places at
every arrival rate tested, because neither cap is ever actually the thing a
request waits on.

**Finding 3 — the real, previously-unidentified bottleneck: prefill's fixed
compute-bound throughput ceiling (~4,138 req/s across the 29-machine pool),
not decode capacity and not the pool.** Direct occupancy instrumentation:
decode occupancy plateaus under sustained overload (climbs, then stays flat
from λ=4,500 all the way to λ=10,000 — a 2.2× further increase in demand
producing almost no change) — self-limiting, bounded by how fast *prefill*
can feed it, not by decode's own capacity. Meanwhile `prefill_wait_mean` grows
unboundedly (0.166s→1.616s across the same range) — the real congestion
signature. Found by building the simulator and instrumenting it directly, not
assumed going in.

**External validation, stated honestly rather than glossed over**: the
KV-handoff-latency claim was already checked analytically (§2c) and stands.
DistServe's throughput multipliers (2.0–3.41×) and Mooncake's 525%/75%
figures were **not reproduced** — both need a colocated/monolithic baseline
simulator this pass explicitly didn't build, a genuine scoped-out extension,
not an oversight. The one thing checkable without a baseline — order-of-magnitude
latency plausibility — lands in a believable regime (0.27s sub-ceiling → 2.0–2.2s
at 2.4× overload), despite entirely different underlying hardware
(TPU 8i FP4 Llama-3-70B vs. DistServe's A100 FP16 OPT-175B).

---

## 5. Full Synthesis — Rack-Scale Budget Across All Four Legs

Given a fixed chip/power/area/bandwidth budget for a rack of TPU 8i, how
should it split across: **(a)** on-chip SRAM for attention scratchpad, **(b)**
on-chip SRAM for hot MoE expert weights, **(c)** interconnect bandwidth for
MoE dispatch, **(d)** interconnect bandwidth for prefill→decode KV handoff?
Not a request to design new silicon — TPU 8i has been the fixed given since
Phase 0 of the attention project; the question is where its already-fixed
resource envelope actually goes once four independently-derived pieces of
evidence are made to share it.

### 5.1 Compute stays outside this budget — it's already priced into the chip ratio

SRAM and bandwidth are genuinely fungible resources within their own pools —
bytes of SRAM can go to (a) *or* (b); link capacity can go to (c) *or* (d).
Compute isn't fungible with either: you can't trade FLOPs for SRAM within a
fixed chip design. Compute provisioning happens at a different level entirely
— *how many chips get built as prefill vs. decode* (§2's chip ratio), not how
one chip's internal budget gets sliced. Phase 4's finding that dense's real
bottleneck is prefill's fixed compute ceiling, not SRAM or bandwidth, doesn't
add a fifth allocatable axis here — it's supporting context for *how
generously* to provision (a)–(d): once a system is compute-bound (the same
asymptote §3 and §4 each independently hit — dynamic admission, KV
quantization headroom, and hot-expert residency at matched cap all converge on
the identical ceiling), more SRAM/bandwidth headroom stops buying throughput.
The four-way split can target "sufficient," not "maximal."

### 5.2 The dense-vs-MoE chip-ratio contrast — one paragraph spec_v2 owes on its own

Dense (5.82:1) and MoE at the matched 8,192-token cap (5.97:1) land within
2.6% of each other — but not because the architecture change was neutral.
Isolating the two levers (§2b): sparse FFN alone drops the ratio 5.82→4.60
(prefill's compute savings amortize more cleanly than decode's, whose FFN was
already near its own crossover); MLA vs. GQA alone pushes it back 4.60→5.97,
the opposite direction. **Two real, opposing effects of comparable order of
magnitude that happen to nearly cancel at this specific context length** —
not a law that MoE preserves dense's disaggregation ratio in general. The
proof it isn't structural: at DeepSeek-V2's real 163,840-token deployment
cap, the same two levers land at **1.31:1**, nowhere near dense. MLA's bytes
saving only pays off once context is long enough to matter (§2b.20's decode
attention breakdown: MLA loses to GQA on cost/token at 8,192, wins 2.25× at
163,840) — the cancellation only holds in the context-length regime where
MLA hasn't kicked in yet. **Direct answer to "does disaggregation work the
same way once you're serving MoE": architecture alone, held at matched
context, barely rebalances prefill:decode — but real MoE deployments don't
hold context fixed, and once context length is allowed to be what the model
was actually built for, the ratio diverges sharply.** Context length, not
architecture family, is the dominant lever on the ratio's magnitude.

### 5.3 (a) vs. (b) — same chip, same SRAM pool, turns out not to compete

(a) is real but structurally small and *fixed* — FlashAttention-style tiling
means the scratchpad footprint is bounded by tile size, never by batch or N
(§3.7); no exact TPU-8i number was derived (flagged, not chased — the same
"ground it in a real source or don't claim precision" call made everywhere
else in this project), but the qualitative shape is solid. (b) is a hard,
architecture-fixed **259.5MB** (22 local experts, EP-sharded) that must stay
resident to realize the ~4.5× real-cap throughput payoff from §3. **KV cache
belongs in neither** — Phase 1's core tension (weights fixed, KV cache
elastic and HBM-bound) already settled that; KV cache's only SRAM contact is
the transient per-tile buffer inside (a) during compute, never separate
residency, which makes spec_v2's original "(b) SRAM for hot experts *and* KV
cache" phrasing an imprecise carryover from before Phase 1 was derived. Since
(a) is small/fixed and (b) is a bounded, comfortably-fitting 259.5MB of TPU
8i's 384MB, the two don't meaningfully compete for the same pool — and
because TPU 8i is homogeneous across the rack, this single-chip fit-check
(Phase 3) generalizes directly to every decode chip in the fleet, not just
the one it was computed for.

### 5.4 (c) vs. (d) — same physical fabric, genuinely different regimes

MoE dispatch (2,560-byte payloads, ~1ns transfer) is **latency-dominated**;
KV handoff (17.48µs dense / 3.69µs MoE) is **bandwidth-dominated**, ~16,000×
bigger a payload. Both ride Boardfly. The real risk this raises: every
regime number in §2c was computed assuming each traffic type has the fabric
to itself — a 17.48µs handoff occupying a link is a long time relative to a
~1ns dispatch packet, and if they share a queue, dispatch traffic could sit
behind a handoff transfer, undermining the low-hop-latency result MoE
dispatch was counting on. This is resolved in real interconnects by
**virtual channels** — logically separate queues multiplexed onto the same
physical link, specifically to prevent exactly this head-of-line blocking —
standard practice in NVLink/InfiniBand-class fabrics, plausible given TPU
8i's sophistication but unconfirmed for Boardfly specifically. Same
"flag it, don't fake precision" treatment as the unpublished ICI hop-latency
figure already swept in §2c: the *general* mechanism is well-established;
the TPU-8i-specific confirmation isn't sourced.

### 5.5 Is there one budget, or a family of them across context length?

Checked directly rather than assumed: (a) is tile-bounded (context-invariant),
(b) is a pure architecture property (context-invariant), (c) is a per-token
dispatch payload (context-invariant). Only **(d)** moves with context — and
even there, its real driver is *average prompt length at handoff time*, not
the decode-concurrency cap (§2c.1 — decode hasn't run a step yet when handoff
fires, so cap-driven context growth is irrelevant to transfer size). §2c
already flagged that `avg_input=512` was reused from dense's own workload for
MoE too, deliberately, for comparability — not a claim about DeepSeek-V2's
real deployed average prompt. If a real MoE deployment serves longer average
prompts (plausible, given DeepSeek-V2 targets long-context use), (d)'s real
number would be larger — but only in a direction that *strengthens* the
bandwidth-dominated finding, never threatens it, since bigger payloads only
widen the existing margin over hop latency. **Three of four legs need one
budget statement, not a family of them; the fourth carries an honest,
directionally-safe caveat rather than an unresolved gap.**

---

## Cross-Phase Synthesis

**A single discipline mattered more than any other, recurring in every phase:
a reused number or formula needs re-verification for the *new* context, not
just confirmation the formula/number is right in its original source.**
`seq_len=8192` (right for the attention project, wrong for this project's own
DistServe-shaped request model, §2.3); the SDPA-only attention scope (right
for a microarchitecture question, silently incomplete for a throughput
question, missing 21.4% of FFN's magnitude, §2.9/§2b.11); `T=8,192` (a real
number from a real sibling project, silently ungrounded for *this* project's
own capacity math, §2b.7); DeepSeek-V2's real context length (neither the
borrowed 8,192 nor an invented number, §2b.7). None were formula errors — all
were scope/context mismatches, the harder kind to catch, and the same
discipline that resolved Phase 5's own SRAM/bandwidth budget question.

**Reversals were kept on record, not scrubbed, every time**: the Groq/SRAM-only
decode chip (Phase 0), FP4/FP8 mixed precision (Phase 0, tied to the same
reversal), full-replication MoE deployment (Phase 2b), and §2.6's ~5.50:1
ratio superseded but not deleted by §2.9's ~5.82:1. Every reversal was driven
by a real mechanistic finding surfacing *after* the original choice looked
reasonable — never by a preference change.

**The same compute-bound asymptote surfaced independently at least four
times, from four different starting questions**: dynamic admission vs.
hard-cap (§3.3), KV-cache quantization's would-be capacity gain (§3.5),
hot-expert residency's matched-cap payoff (§3.8–3.9), and Phase 4's own
occupancy plateau (§4.7). Once a system is safely past its compute-bound
crossover, more concurrency/memory headroom stops buying throughput —
batch size and *how* you got to compute-bound (brute-force N vs. removing a
fixed memory cost) stop mattering, only per-token cost and peak FLOPs/s do.
A single generalizable result, discovered independently enough times to trust
it as structural rather than coincidental.

**Building the simulator was the highest-value single activity in the whole
project**, exactly matching the sibling projects' own "gap-hunting is the
real work" theme: it caught a wrong attention-regime formula in this
project's own verbal design brief, a race condition that could have silently
broken the decode admission cap, a missing pool-release call that would have
broken every contention experiment, and — the actual headline finding no
closed-form derivation had surfaced — that the real bottleneck across the
whole system is prefill's fixed compute ceiling, not decode or the pool.

---

## Open Threads / Known Gaps

- **Expert-parallel MoE deployment was never run through Phase 4's real
  simulator** — Phase 4 validated the dense leg only. MoE's queueing/pool
  contention behavior under real bursty arrival is a legitimate, complete
  follow-on pass.
- **A real dynamic-admission-with-eviction model** (as opposed to the
  hard-cap-vs-dynamic-N *comparison* already run) was never built — §3's
  admission-policy finding is about throughput/latency parity, not a full
  eviction implementation.
- **Tiered CPU/SSD offload for the intermediate pool** (Mooncake's own real
  design) — declined as a scope expansion; how often the block-until-free
  placeholder actually binds under realistic load beyond what Phase 4 already
  measured (essentially never, at realistic pool sizes) is answered, but the
  offload alternative itself was never built.
- **KV-cache quantization below FP4** — declined for blast-radius reasons
  (§3.5); real, well-published, would ripple through every number since Phase 1.
- **A colocated/monolithic baseline simulator** — needed to actually reproduce
  DistServe's/Mooncake's throughput-multiplier and SLO-attainment figures,
  rather than the order-of-magnitude latency plausibility check this project
  settled for (§4.9).
- **TPU 8i's real virtual-channel/QoS architecture for Boardfly** — Phase 5's
  §5.4 argument depends on a real mechanism (traffic-class separation) that's
  standard in comparable fabrics but unconfirmed for Boardfly specifically.
- **FlashAttention's exact tile-size formula, instantiated for TPU 8i** — the
  qualitative "small and fixed" finding (§3.7) was enough to close Phase 3 and
  Phase 5's SRAM argument; the precise number was never extracted.
- **A real per-device population/N model for MoE routing under non-uniform
  device assignment** — this project's entire MoE-routing-sparsity analysis
  assumes device-to-expert assignment is uncorrelated with popularity
  (`moe_routing_notes.md`'s own sourced deployment fact); real load-balanced
  placement would break the clean 1/8th-scaling symmetry §2b.5 relies on.

---

Full code and sweep results for the Phase 4 simulator:
[`disagg-and-placement-sim/`](disagg-and-placement-sim/).
