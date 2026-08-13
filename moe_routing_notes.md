# MoE Routing System Codesign — DeepSeek-V2 on TPU 8i

A workload→system codesign study, extending the attention project's single-chip
microarchitecture loop ([`attention/notes.md`](attention/notes.md)) up one level:
from PE array/dataflow/scratchpad sizing on one accelerator, to interconnect
topology/bandwidth/buffering across a multi-chip system. The workload is MoE
expert routing specifically because it's data-dependent — which expert a token
goes to depends on the token itself, so load imbalance (not just average-case
shape) has to be modeled, unlike attention's fixed batch/seq_len/heads.

Every phase follows the same discipline as the attention project: hand-derive a
prediction, ground every number in a real source (paper, spec sheet, or
literature search) rather than memory or invented heuristics, then check it
against a tool and explain any gap mechanistically.

---

## 1. Workload: DeepSeek-V2 on TPU 8i

**Source:** DeepSeek-V2 paper (arXiv 2405.04434), §1/§3.1 (hyperparameters),
§2.1 (MLA architecture, eq. 1–19), §2.2 (routing and load balancing).

### Shape

| Parameter | Value | Source |
|---|---|---|
| Routed experts | 160 | paper §3.1 |
| Shared experts | 2 (replicated on every device, always active) | paper §3.1 |
| top_k (routed) | 6 | paper §3.1 |
| Device-limited routing cap (M) | ≤3 of 8 devices per token | paper, device-limited routing section |
| d_model | 5,120 | paper §3.1 |
| Expert FFN intermediate dim (d_ff) | 1,536 | paper §3.1 |
| Transformer layers | 60 (layer 1 is a dense FFN, not MoE — not modeled below; known small gap) | paper §3.1 |
| Total / activated params | 236B / 21B | paper §3.1 |
| EP group size (D) | 8 devices — "routed experts uniformly deployed on 8 devices" | paper |
| n_heads / d_head | 128 / 128 | paper §3.1 |
| KV compression dim (d_c) / query compression dim (d_c') | 512 / 1,536 | paper §3.1 |
| Decoupled RoPE dim (d_R_h) | 64 | paper §3.1 |
| seq_len | 8,192 (reused from the attention project for cross-project comparability; not a first-order lever for MoE comms — dispatch is per-token, not sequence-quadratic) | — |
| Chip | **TPU 8i** (GA April 2026) | [Google Cloud technical deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) |
| Precision | **FP4, uniform** (compute + comms — simplifying assumption, revisited in §2.2) | matches TPU 8i's native format |

**TPU 8i specs used throughout:** 288 GB HBM/chip @ 8.6 TB/s; 384 MB on-chip
SRAM; 10.1 PFLOPS FP4; inter-chip interconnect 19.2 Tb/s (= 2.4 TB/s, converting
bits→bytes) on a new **"Boardfly" topology** explicitly built to cut network
diameter ~56% for MoE/reasoning workloads; a Collective Acceleration Engine
(CAE) offloading reductions.

**Why TPU 8i, not TPU v5e** (the attention project's chip): v5e was chosen there
to mimic Gemmini's small RTL-simulatable scale; that constraint doesn't apply
here, since ASTRA-sim is topology-agnostic. TPU 8i wins on its own merits: it's
purpose-built for MoE serving (Boardfly directly informs the topology question
in §3), and it gives one consistent source for both per-chip memory and
inter-chip interconnect numbers.

### Deployment model

- **Attention: data-parallel** across the 8 EP-group devices — each device
  holds a full, unsharded MLA weight stack and runs attention locally on its
  own batch slice. Zero cross-device communication for attention itself.
- **MoE FFN: expert-parallel.** Each token's post-attention hidden-state vector
  (d_model=5,120) gets **dispatched** to whichever device(s) hold its top-6
  routed experts (capped at M≤3 devices/token by construction), computed there,
  and the result **combined** back to the token's home device.
- **Shared experts generate no dispatch traffic** — replicated on every device,
  always computed locally, avoiding a guaranteed hotspot.
- Per-device local expert count: 20 routed (160/8) + 2 shared = **22**.

**Why not tensor-parallel attention** (a real detour, not assumed): TP-by-heads
only works when a weight matrix's *output* has a per-head axis — a device can
own a disjoint set of heads and combine once at the end. Three of MLA's 8
weight matrices (`W^DKV`, `W^DQ`, `W^KR`) have **no** head axis at all — each
produces one shared latent that every head needs in full, so no device could
compute even one complete head's output without fetching the missing pieces
first (communication before every layer, defeating TP's point). Resolution:
don't TP attention at all — data-parallel instead, each device holding the
full 8-matrix stack.

### MLA weight matrices and KV cache

| Matrix | Role | Shape | Per-head? |
|---|---|---|---|
| W^DKV | KV down-projection | d_c × d | No |
| W^UK | Key up-projection | (n_h·d_h) × d_c | Yes |
| W^UV | Value up-projection | (n_h·d_h) × d_c | Yes |
| W^DQ | Query down-projection | d_c' × d | No |
| W^UQ | Query up-projection | (n_h·d_h) × d_c' | Yes |
| W^QR | Decoupled query (RoPE) | (n_h·d_R_h) × d_c' | Yes |
| W^KR | Decoupled key (RoPE), shared across heads | d_R_h × d | No |
| W^O | Output projection | d × (n_h·d_h) | Yes |

`W^UK`/`W^UV` can be algebraically absorbed at inference time (a
compute/activation-memory trick, not fewer *stored* params — all 8 matrices
above are what's checkpointed).

**KV cache** (paper's own result): `(d_c + d_R_h) · l` = **576
elements/token/layer** — no factor of 2 (only the compressed latent is cached;
V is reconstructed via up-projection), no per-head axis. That compression is
MLA's whole point.

### Weight footprint and batch size

- Attention (8 MLA matrices, unsharded): 149,225,472 params/layer
- FFN (22 local experts × 3×d_model×d_ff, SwiGLU gate/up/down): 519,045,120 params/layer
- Total × 60 layers = **40.1B params ≈ 20.05 GB at FP4** (of the 288 GB HBM budget)
- *Consistency check:* summing FFN params across all 162 experts × 59 MoE
  layers + attention × 60 layers gives ≈234.5B vs. the paper's stated 236B —
  close enough to trust the formulas (gap is the unmodeled dense first layer,
  embeddings, output head).
- Remaining HBM for KV cache: 267.95 GB; bytes/token (all 60 layers) = 0.5 ×
  576 × 60 = 17,280; max cached tokens ≈ 15,506,474; max local batch @
  seq_len=8,192 ≈ 1,893.
- **Chosen: local batch = 1,024/device** (~46% safety margin for activations/
  router/framework overhead) → **system-wide batch = 8,192 tokens/decode step**
  (1,024 × 8 devices, since each device's local batch independently feeds the
  shared/routed expert pool via dispatch).
- Sanity check: ~59× the attention project's batch=32 (vanilla MHA/GQA,
  similar TPU-generation memory budget) — expected direction, since MLA's
  entire point is enabling larger serving batches via the compressed cache.

---

## 2. Communication volume

### 2.1 Uniform routing (the ideal case)

Scope decisions: **decode-step, not prefill** (one new token per active
sequence per forward pass — seq_len governs KV-cache sizing, not per-step
token count); dispatch/combine kept at **FP4** for consistency with compute
(caveat: real EP systems sometimes prefer higher precision at network
boundaries specifically, since each quantize↔dequantize hop introduces its own
rounding error — flagged, not yet load-bearing); FLOPs scope is **all 8
experts/token** (2 shared + 6 routed), since the compute-to-comms ratio is
about which physical engine (compute vs. network port) gates execution time —
shared-expert compute occupies the same compute engine even though it
generates no dispatch traffic.

**Per-token dispatch fan-out**, verified against the primary source (§2.2.2,
not assumed): *"for each token, we first select M devices that have experts
with the highest affinity scores... then perform top-K selection among experts
on these M devices."* This is structural — a token's 6 routed experts are
guaranteed to live on ≤3 devices by construction (device selection happens
*before* expert selection). Using expected value for the uniform/control case
(worst case belongs in §2.2):

- P(home device ∈ top-3) = 3/8 → 2 remote hops; P(home ∉ top-3) = 5/8 → 3 remote hops
- **E[remote devices/token] = (3/8)(2) + (5/8)(3) = 2.625**

Combine mirrors dispatch exactly: multiple experts landing on the same remote
device get summed locally there and returned as **one** message, and the
down-projection (d_ff→d_model) happens before crossing the network, so combine
payload size equals dispatch payload size.

**Bytes moved:** payload = 5,120 × 0.5B (FP4) = 2,560 bytes/token. Per-token-
per-layer = 2,560 × 2 (dispatch+combine) × 2.625 = 13,440 bytes. × 8,192
system-wide tokens = **110,100,480 bytes = exactly 105 MiB/layer** (whole
system, one decode step).

**FLOPs moved:** per-token-per-expert (SwiGLU: gate+up+down, standard 2×M×N×K)
= 3 × 2 × 5,120 × 1,536 = 47,185,920. × 8 experts/token × 8,192 tokens =
**≈3.09 TFLOPs/layer**.

**Ridge point** (TPU 8i ICI, 19.2 Tb/s ÷ 8 = 2.4 TB/s, verified independently
since Google's own blog doesn't state it explicitly): 10.1×10¹⁵ ÷ 2.4×10¹² ≈
**4,208 FLOPs/byte**. Workload AI = 3.09×10¹² ÷ 1.101×10⁸ ≈ **28,087
FLOPs/byte** → **≈6.7× above ridge — decisively compute-bound**, in the ideal
case. Caveat carried forward: this is a system-wide *average*; a locally
comms-bound moment on one overloaded device could exist even while the global
average looks fine — exactly why imbalance needs its own analysis.

### 2.2 Load imbalance (the real case)

**A correction, caught by re-reading the primary source rather than trusting
an earlier note:** DeepSeek-V2 does **not** restrict token-dropping to
training. §2.2.4: *"we can flexibly decide whether to drop tokens during
inference according to the efficiency requirements, and always ensure
consistency between training and inference."* Locked down from the same
passage: **CF = 1.0 exactly** (strictest possible, no slack above average),
**device-level** (matching this project's 8-device EP model), drop rule is
**lowest-affinity-score tokens first**, ~10% of sequences protected from
dropping (a fairness guarantee, irrelevant to the volume math here).

**Imbalance magnitude**, from real literature rather than an invented
heuristic (an initial instinct to reach for a Pareto-principle-style analogy
was explicitly rejected — no MoE-specific evidence for that shape):

- **Gini coefficient ≈ 0.70** (baseline expert load, before mitigation),
  averaged across measured open-source MoE models including DeepSeek-V3 —
  "Latent Prototype Routing" (arXiv 2506.21328).
- **GPU stall time up to 70%** under skewed routing, measured on real
  Mixtral-8×7B serving (databricks-dolly-15k, DGX A100, SGLang
  expert-parallel) — "Toward Cost-Efficient Serving of Mixture-of-Experts with
  Asynchrony" (arXiv 2505.08944). Device-level and directly convertible to a
  load ratio: idle 70% of a step waiting on the hot device → only doing useful
  work 30% of the time → hot device's load ≈ 1/0.30 ≈ **3.3× average**. Used
  as the working multiplier; cross-validated against the Gini finding
  (independently implies severe multi-x imbalance). Neither number is
  DeepSeek-V2-specific and Mixtral is architecturally quite different (8
  experts/top-2 vs. 160/top-6/device-limited) — carried forward as a magnitude
  anchor, not a precise prediction.

**Per-device decomposition** (the key structural move — naively scaling *all*
of §2.1's numbers by the imbalance multiplier would leave the FLOPs/byte ratio
unchanged, making the whole exercise pointless):

- `F_shared,device` = (T/8) × 2 × F_expert ≈ 96.6 GFLOPs — driven by how many
  tokens call *this device home*, independent of routing popularity, does
  **not** scale with the multiplier
- `F_routed,device` = (T/8) × 6 × F_expert ≈ 289.9 GFLOPs — driven by
  inbound-dispatched tokens, **scales** with the multiplier
- `B_device` = (T/8) × 2 × P × R ≈ 13.1 MiB — also driven entirely by
  inbound-dispatched tokens, **scales** with the multiplier

(T=8,192 tokens, P=2,560 bytes, R=2.625.)

**Dropping-off (uncapped) sensitivity sweep:**

| Stall % | Multiplier | Device FLOPs | Device bytes | AI (FLOPs/byte) |
|---|---|---|---|---|
| 70% (working number) | 3.33× | 1.05 TFLOPs | 43.3 MiB | 23,193 |
| 80% | 5.0× | 1.55 TFLOPs | 65.5 MiB | 22,469 |
| 90% | 10.0× | 3.00 TFLOPs | 131.1 MiB | 21,767 |
| 95% | 20.0× | 5.90 TFLOPs | 262.2 MiB | 21,416 |
| 97% | 33.3× | 9.75 TFLOPs | 437.0 MiB | 21,276 |
| 99% | 100.0× | 29.09 TFLOPs | 1.31 GiB | 21,135 |

**Structural finding:** as the multiplier → ∞, `F_shared,device` becomes
negligible and the ratio converges to a **hard, imbalance-proof floor**: `3 ×
F_expert / (P × R) ≈ 21,065 FLOPs/byte` — a pure function of DeepSeek's expert
shape and routing fan-out, with **zero dependence on imbalance severity or
hardware**. This floor sits entirely above TPU 8i's ridge point (4,208) — no
amount of imbalance, however extreme, flips this workload comms-bound at FP4.
The 70%→99% sweep only erodes headroom from 6.7× to ~5.0× — imbalance barely
moves the needle.

**What actually can flip the verdict — precision, not imbalance:**

| Precision | Payload bytes | Floor (FLOPs/byte) | Margin vs. ridge (4,208) |
|---|---|---|---|
| FP4 (chosen) | 0.5 | 21,065 | ~5.0× |
| FP8 | 1.0 | 10,533 | ~2.5× |
| FP16/BF16 | 2.0 | 5,266 | ~1.25× (thin) |
| FP32 | 4.0 | 2,633 | **below ridge — flips comms-bound** |

**Exact crossover ≈ 2.50 bytes/element** — FP16-or-finer keeps this workload
compute-bound regardless of hardware generation or imbalance; coarser flips
it. This is a far more sensitive lever than imbalance severity.

**Dropping-on (CF=1.0)**, the third leg: capping realized load at exactly 1.0×
average makes FLOPs/bytes/AI **identical to the uniform §2.1 baseline**
(28,087 FLOPs/byte) — the interesting output isn't a ratio, it's the
**dropped-token fraction**: desired 3.3× average, allowed 1.0×, excess 2.3× →
dropped fraction of that device's excess demand = 2.3/3.3 ≈ **69.7%**. Initial
read ("70% dropped = model quality collapse") was overgeneralized; correctly
scoped, this is (1) one device's demand, not system-wide — the other 7 are
correspondingly *under* average by conservation, (2) a drop costs a token one
expert out of 8, not the whole computation, (3) the rule specifically targets
each token's *lowest-affinity* pick. A real but narrow, deliberately-cushioned
cost — though the exact system-wide magnitude (what fraction of all tokens
even want the hot device) is left unresolved.

**Summary — the three-point range:**

| Case | Bottleneck-device AI | Verdict | Cost |
|---|---|---|---|
| Uniform | 28,087 FLOPs/byte | Compute-bound, 6.7× margin | None (idealized) |
| Dropping-on (CF=1.0) | 28,087 (identical) | Compute-bound, 6.7× margin | ~69.7% of one device's *excess* demand dropped |
| Dropping-off (uncapped) | 23,193 → floor 21,065 | Compute-bound, ~5.0–5.5× margin | No drops, but real stall/buffer pressure |

**Headline:** this workload is robust to realistic load imbalance on TPU 8i at
FP4 — every case stays decisively compute-bound. If anything tips it
comms-bound, it's a numerics decision (dispatch precision coarser than ~2.5
bytes/element), not routing skew.

---

## 3. System architecture hypothesis (hand-derived, pre-simulation)

### 3.1 Topology — direct full mesh across the 8-device EP group

Starting instinct: MoE dispatch/combine is "all-to-all-ish," the way NVIDIA
NVSwitch racks handle it. Worth testing what property of NVSwitch actually
earns that reputation before assuming it transfers.

**Traffic-pattern precision:** device-limited routing caps a single *token's*
fan-out at M≤3 of 8, but which 3 varies token-by-token by content-driven
affinity, unrelated to which device physically hosts the token. Aggregated
over a device's ~1,024-token batch, the device's *aggregate* traffic is
effectively any-to-any across all 7 peers — the per-token cap shrinks per-token
hop count, not the topology requirement.

**NVSwitch mechanism, verified via search:** an internal crossbar — every
input port cross-connected to every output port — giving any pair a single
hop at full bandwidth, non-blocking (no pair's transfer steals bandwidth from
another, as long as the switch's own total capacity holds). ([NVSwitch
overview](https://www.nextpcb.com/blog/what-is-nvswitch-nvidia-gpu-cluster-scale-out),
[NVIDIA technical
overview](https://images.nvidia.com/content/pdf/nvswitch-technical-overview.pdf))

**Mesh vs. switch, worked through directly:**
- Direct mesh eliminates the switch "middleman" → lower latency, but costs
  O(N²) links (28 at N=8) — pin/board-space/heat-constrained, infeasible fast
  as N grows.
- A switch's bandwidth budget isn't pre-partitioned into fixed per-peer slices
  the way mesh's dedicated links are — it statistically multiplexes, following
  wherever demand actually is. Matters specifically under **imbalanced**
  traffic. Caveat: doesn't eliminate contention, just relocates it — from the
  sender's dedicated links to the *destination's* incoming link, if multiple
  senders pile onto the same hot device simultaneously.

**The deciding factor — this workload is latency-dominated, not
bandwidth-dominated.** Dispatch payload is 2,560 bytes/hop (FP4). Transfer
time at TPU 8i's 2.4 TB/s = 2,560/2.4×10¹² ≈ **1.07 ns** — utterly dwarfed by
typical fixed per-hop latency (hundreds of ns). That undercuts the switch's
main advantage (multiplexing barely matters when bandwidth is already this
abundant relative to payload) and reinforces mesh's main advantage (no
middleman hop).

**Real-world resolution:** TPU 8i doesn't use a centralized switch chip at
all — it uses **hierarchical direct mesh** ("Boardfly"): full mesh at 4
chips/board, full mesh at 8 boards/group, full mesh at 36 groups/pod (1,152
chips total). Each tier stays small enough that O(N²) wiring is feasible,
avoiding a dedicated switch ASIC while still reaching pod scale via nesting.

**Conclusion: direct full mesh across the 8-device EP group** (28 dedicated
links) — matches the tier size TPU 8i itself already treats as mesh-feasible,
and fits this workload's latency-dominated regime.

### 3.2 Bandwidth allocation — uniform across all 28 links

**Argument 1 (sufficient alone):** §2.2's compute-bound headroom survives even
the most extreme modeled imbalance (99% stall → floor ≈21,065, still ~5× above
ridge). No modeled scenario makes comms the bottleneck, so extra link
bandwidth has no exploitable payoff.

**Argument 2** (raised, then largely undercut, but didn't change the
conclusion): non-uniform provisioning is a wiring-time decision trying to
anticipate a routing pattern that's data-dependent and shifts per decode step
— you can't know which links to favor in advance. Tested by asking whether
expert popularity is actually *persistent* (e.g. semantic specialization)
rather than purely transient — if persistent, the design-time objection
weakens. Searched rather than assumed (recognizing this as the same move —
Pareto-by-analogy — already rejected once for lack of evidence, this time
backed with real citations):

- "A Closer Look into Mixture-of-Experts in Large Language Models" (arXiv
  2406.18219): *"in some layers of DeepSeek, there is an expert selected by
  most tokens"* — DeepSeek-specific.
- "Do Domain-specific Experts exist in MoE-based LLMs?" (arXiv 2604.05267,
  preprint): evaluated 10 MoE LLMs (3.8B–120B), found empirical evidence for
  genuine domain-specific expert specialization, not random routing.

Real evidence for structural popularity exists — Argument 2's objection is
weaker than it looked. **Doesn't change the answer**: Argument 1 alone
suffices, and there's no payoff to the added complexity. Uniform wins on
headroom + Occam's razor.

### 3.3 Buffering/SRAM — CF=1.0 is what makes on-chip buffering feasible

Conservative framing, chosen deliberately: assume **no pipelining** (a full
layer's dispatch+combine traffic must be resident before compute proceeds),
partly because SRAM is shared with on-chip needs never sized here (matmul
operand tiles, activation staging — this project only ever budgeted the *HBM*
side, weights + KV cache).

| Case | Device bytes (one layer) | Fraction of 384MB SRAM |
|---|---|---|
| Uniform | ≈13.1 MiB (`B_device`) | ~3.4% |
| Worst modeled imbalance (99% stall, 100× multiplier) | ≈1.31 GiB | **>340% — infeasible, alone** |

The worst case is infeasible outright — over 3× the *entire* SRAM budget,
before a single byte is reserved for anything else on-chip.

**Resolution:** CF=1.0 (device-level capacity capping from §2.2) caps a
device's *realized* inbound traffic at exactly the uniform-case volume,
regardless of underlying demand skew. This reframes CF=1.0 from "one point in
a three-way range" into a **practical necessity once buffering is
considered** — dropping-off isn't just costlier, it's physically infeasible
on-chip past fairly mild imbalance (even 80%/5× already needs 65.5 MiB).
**Buffering requirement with CF=1.0: ≈13.1 MiB/device (~3.4% of 384MB)**,
leaving ~96.6% for everything else on-chip.

**A correction to "imbalance barely matters":** that conclusion holds for the
compute-vs-comms ratio specifically (wide headroom, 6.7×→~5×). It does **not**
hold for buffering — SRAM is a hard physical wall with no equivalent headroom,
and uncapped imbalance blows straight past it. Buffering is the one place
imbalance genuinely bites, and CF=1.0 capacity capping — not pipelining, not
overprovisioned SRAM — is the mechanism that resolves it.

### 3.4 Summary

| Question | Answer | Key reason |
|---|---|---|
| Topology | Direct full mesh, 8 devices | Latency-dominated (transfer time ~1ns ≪ hop latency); matches TPU 8i's own mesh-tier precedent |
| Bandwidth allocation | Uniform across all 28 links | Compute-bound headroom survives worst-case imbalance; no exploitable payoff to non-uniform |
| Buffering/SRAM | ≈13.1 MiB/device, requires CF=1.0 | Uncapped imbalance is infeasible (>3× the whole SRAM budget); capacity-factor capping is load-bearing |

---

## 4. Testing the hypothesis in ASTRA-sim

**Setup:** [ASTRA-sim](https://astra-sim.github.io/) cloned to `~/dev/astra-sim`
(sibling to this repo, matching where the attention project's Timeloop
tooling lives, rather than nested inside this repo). Built via
`docker build -t astra-sim:dev .` from the repo's own Dockerfile — confirmed
**native arm64**, no emulation, since the Dockerfile is unpinned
`FROM ubuntu:22.04`. Compiled both `AstraSim_Analytical_Congestion_Unaware` and
`_Congestion_Aware` targets. Checkpoint: the stock 4-NPU ring all-reduce
microbenchmark ran cleanly (43,000 cycles, fully exposed as communication —
expected, since it has no compute component). One process hiccup along the
way: Docker Desktop had silently quit between an earlier health check and the
actual build; caught via the daemon-socket error, relaunched, re-ran clean.

Custom configs for this project's own numbers used the analytical backend,
congestion-unaware, 8-device EP group throughout — inlined in the
[Appendix](#appendix-astra-sim-configs-4) rather than left in the astra-sim
clone.

### 4.1 Topology comparison — the core test

Ran the **real per-hop dispatch payload** (2,560 bytes, FP4 — not a stock
microbenchmark size) across three topologies, same bandwidth (2,400 GB/s, TPU
8i ICI) and latency (500 ns — see §4.2 for why no exact TPU-specific figure
exists), each paired with its natural collective algorithm:

| Topology | Algorithm | Wall time (cycles) |
|---|---|---|
| Full mesh (`FullyConnected`) | `direct` | **520** |
| Switch | `direct` | 1,020 |
| Ring | `ring` | 14,290 |

**Matches §3.1's prediction**: full mesh wins clearly. Two things worth noting
mechanistically:
- Switch costs almost exactly **2×** the full mesh — lines up with the
  "eliminates the middleman" reasoning: a switch hop models as two legs
  (sender→switch, switch→receiver), a direct mesh link is one.
- Ring is catastrophic by comparison (~27× the mesh) — at this tiny payload
  size, every hop pays the full fixed latency with almost nothing to amortize
  it against, and reaching non-adjacent ring nodes takes multiple hops.
  Confirms the latency-dominated framing sharply: topology matters enormously
  *because* the payload is this small.
- A supporting mechanistic check from an earlier pass at the stock 1MB
  microbenchmark (before switching to the real 2,560-byte payload above): the
  bandwidth-driven component of wall time came out to ~70 cycles, not the
  ~417 a naive "whole 1MB over one link" calculation would predict. That's
  consistent with `direct` on a full mesh sending all 7 peer-messages
  **simultaneously**, each carrying only ~1MB/7 ≈ 150KB (150KB/2,400GB/s ≈
  62ns, close to observed) — a direct mechanical confirmation of the "no
  pre-partitioning penalty" argument from §3.1.

### 4.2 Latency sensitivity sweep

No published TPU 8i ICI latency figure exists (searched — Google states
relative diameter/latency reductions, ~56%/~50%, but no absolute ns number,
for any TPU generation). Swept a plausible range instead of inventing one
number: 500 ns (ASTRA-sim's own generic stock default), 936.25 ns (ASTRA-sim's
*validated* HGX-H100 figure — a real measured NVSwitch fabric latency,
empirically curve-fit as a single effective parameter per the tool's own
validation docs, not decomposable into separate hop legs and therefore not
principled to halve), plus 300 and 750 ns to fill out the range.

**A methodology catch worth keeping, not just the numbers:** the first sweep
attempt showed wall time scaling **~4× the latency delta** instead of the
expected 1×. Traced to `preferred-dataset-splits: 4`, inherited unmodified
from the stock config used as a template — it silently splits every
collective into 4 chunks, each apparently paying its own latency. Not a
property of the topology; an artifact of a copied config value never
scrutinized. Fixed by setting splits=1 (also more representative — a single
2,560-byte token vector isn't something you'd chunk further in practice).
After the fix:

| Latency | Wall time (cycles) |
|---|---|
| 300 ns | 370 |
| 500 ns | 570 |
| 750 ns | 820 |
| 936.25 ns | 1,007 |

Scales exactly **1:1** with latency across all four points (Δwall time =
Δlatency, e.g. +200ns → +200 cycles) — confirms the fix and isolates the true
per-hop cost. The lesson generalizes: an unscrutinized default inherited from
a template config can silently corrupt a sensitivity result, and the fix is
to notice the multiplier doesn't match physical intuition and trace it to a
mechanism, not just accept the first number the tool returns.

### 4.3 Bandwidth allocation and buffering/SRAM — not testable in this backend

Checked both *before* attempting to build tests, not after:

- **Bandwidth allocation**: the analytical network config sets one `bandwidth`
  value per topology *dimension*, applied uniformly to every link in that
  dimension (confirmed across all stock configs, including multi-dimensional
  composites). No way to give one specific link more bandwidth than its
  neighbors within the same topology — uniform by construction.
- **Buffering/SRAM**: the remote-memory backend has exactly one mode,
  `NO_MEMORY_EXPANSION` — the source explicitly states *"remote memory access
  is not supported"* in that mode. This backend models HBM/weight-offload
  expansion, not on-chip network-buffer occupancy or SRAM backpressure — a
  different concept entirely.

**Not a disagreement between tool and hypothesis** — the analytical backend's
abstraction level doesn't reach either question. Testing them for real would
mean hacking multi-dimensional topology tricks to fake asymmetric bandwidth,
or switching to a packet-level backend (ns-3/htsim) that might model buffers
explicitly. Assessed as not worth the scope increase given what §3 already
established by hand — a deliberate stopping point, not an abandoned thread.

**Deferred, not chased:** whether per-hop latency can be *hidden* behind
concurrent compute in steady state (rather than showing up as forced-serial
"exposed" time, the way the isolated single-hop tests above necessarily do).
ASTRA-sim does track an "exposed communication" metric separate from total
time, so this is answerable in principle — it would test whether Phase 1's
implicit `time = max(compute, comm)` framing actually holds once real
overlap/scheduling is modeled. A real, legitimate question, just not on this
project's critical path; worth returning to if a future gap needs this kind
of mechanistic explanation.

### 4.4 Summary

| Question | Prediction (§3) | ASTRA-sim result |
|---|---|---|
| Topology | Full mesh beats switch/ring | **Confirmed** — 520 vs. 1,020 vs. 14,290 cycles |
| Bandwidth allocation | Uniform, no exploitable payoff to non-uniform | **Not testable** — backend has no per-link bandwidth |
| Buffering/SRAM | CF=1.0 required, ~13.1 MiB/device | **Not testable** — backend has no buffer-capacity model |

The one testable leg confirmed the hand-derivation cleanly. Worth being honest
about the shape of that result: it wasn't surprising — full mesh beating a
27×-worse ring was already the expected direction. That's a *good* outcome,
not a wasted phase: a hypothesis surviving contact with a simulator validates
the reasoning chain that produced it (§2's numbers → §3's topology/bandwidth/
buffering argument), rather than indicating nothing was learned. If Phase 3
had overturned §3, that would mean the hand-derivation missed something real.

---

## 5. Cross-project synthesis: attention's SRAM vs. MoE's interconnect

In a real deployment, attention (the sibling project, [`attention/`](attention/))
happens *on* each chip, and MoE routing (this project) happens *across* them,
in the same forward pass, competing for the same chip's power/area budget. Does
provisioning more interconnect for MoE come at the expense of on-chip SRAM for
attention's scratchpad?

**The core reframe:** SRAM and interconnect (SerDes/PHY circuitry for ICI
links) look like they do completely different jobs — but both draw on the same
finite, upstream resources when a chip actually gets designed: **die area and
power**. SRAM cells cost area roughly proportional to capacity; so does the
analog PHY/SerDes circuitry needed for every interconnect lane, and that
circuitry is notoriously area-expensive per unit of bandwidth compared to
digital logic. Power is the more load-bearing half of the argument: moving a
bit across an on-chip SRAM access costs energy; moving that same bit off-die
across an interconnect link costs dramatically more — one of the best-
established facts in accelerator architecture (data movement gets more
expensive the farther it travels: register → SRAM → HBM → chip-to-chip
interconnect, each hop costing meaningfully more energy per bit than the last).
Under a fixed power envelope, pushing more bits through ICI at higher
bandwidth draws from the same budget that SRAM capacity/activity draws from.

**It's not literal competition for the same pool.** MoE's own directly
measured SRAM footprint (§3.3: ~13.1 MiB of TPU 8i's 384 MB, ~3.4%) is
negligible — MoE isn't fighting attention for SRAM capacity in any meaningful
sense. The competition is one level up: SRAM cells and SerDes/PHY circuits are
two different consumers of the same upstream area/power budget, not two
consumers of the same memory pool.

**Grounded in a real architectural decision, not a vague dial:** "attention-
focus → more SRAM, MoE-focus → more interconnect" traces directly back to §1's
deployment model — attention is data-parallel (fully on-chip, zero cross-
device traffic), MoE is expert-parallel (cross-chip by construction). The
resource split follows from that, not from an abstract analogy.

**The sharpened version — it's connectivity, not raw bandwidth.** §2.2 found
huge compute-bound headroom (5–6.7× above ridge) — bandwidth has real slack.
§4.1's actual simulation found *topology* swinging wall time by 27× (mesh vs.
ring). So a MoE-heavy chip's interconnect investment should go toward more
**direct links/fan-out** (more SerDes ports enabling low-latency, high-
connectivity topology — area cost scaling as O(N²) for full mesh), not fatter
individual links. This is the single most specific, load-bearing conclusion
of the whole synthesis: "more interconnect for MoE" is really "more ports for
connectivity," a different silicon investment than "wider/faster SerDes per
lane."

**A humbling finding on cross-project number-combining:** the two projects'
numbers can't be naively added. Attention's real scratchpad/accumulator sizing
(from `attention/notes.md`, Phase 1b) — **~1 MiB scratchpad + 256 KiB
accumulator** (Gemmini's real hardware defaults, ~1.25 MiB total) — was
deliberately derived for **Gemmini**, a small RISC-V accelerator generator
chosen specifically because it's RTL-simulatable. That project's own notes
record explicitly rejecting TPU-8i-scale SRAM as a sizing target: *"Verified
TPU 8t/8i on-chip SRAM claim via web search (128 MiB / 384 MiB) — rejected as
sizing target (breaks v5e ridge-point consistency; unrealistic for Gemmini
RTL simulation)."* So attention's 1.25 MiB figure isn't "what attention needs
on TPU 8i" — it's what attention needs on a completely different, much
smaller reference chip. Adding it to this project's 13.1 MiB MoE figure
against TPU 8i's 384 MB budget would compare numbers derived for two different
physical chips as if they were the same one. Neither project, individually,
actually answers "how much SRAM does *one* chip running both workloads need" —
each picked the right-scale reference chip for its own tooling constraints,
correctly, for reasons specific to that project. A rigorous answer would
require re-deriving attention's scratchpad sizing at TPU-8i scale, which
neither project did.

**Real-world consolation, even without a clean same-chip number:** the
attention project's actual roofline chip, TPU v5e, has 128 MiB of on-chip
VMEM (also verified there) — same vendor, one generation earlier than this
project's TPU 8i (384 MB). A real ~3× SRAM increase across TPU generations,
and this project's own §1 sourcing already explains *why*: Jeff Dean's own
description of TPU 8i cites more on-chip SRAM and the new Boardfly topology as
both explicitly built for MoE/reasoning workloads. Real hardware is already
making exactly the area/power trade-off this synthesis reasoned through by
hand.

---

## 6. Key takeaways

1. **The compute/comms-bound regime, like the compute/memory-bound regime in
   the attention project, turned out to be far more robust than expected** —
   not to a design decision this time (that was attention's finding), but to
   *load imbalance*: a structural floor (≈21,065 FLOPs/byte) exists that no
   routing skew, however extreme, can break. The lever that actually matters
   is a numerics decision (dispatch precision, crossover at ≈2.5
   bytes/element), echoing how attention's fusion decision — not the workload
   shape — was what actually set its regime.
2. **Device-limited routing's per-token cap (M≤3) doesn't shrink the topology
   requirement** — only the per-token hop count. Aggregated over a device's
   batch, the real traffic pattern is any-to-any regardless of the cap. A
   genuinely non-obvious correction made mid-derivation, not assumed from the
   start.
3. **A full mesh wins not because raw bandwidth matters, but because this
   workload is latency-dominated** — transfer time per hop (~1ns) is
   negligible next to fixed hop latency (hundreds of ns), so minimizing hop
   count (mesh) beats statistically multiplexing bandwidth (switch), which is
   the opposite of what you'd conclude from bandwidth considerations alone.
   Confirmed in ASTRA-sim with a clean, mechanistically-explained 2× (switch)
   and 27× (ring) penalty relative to mesh.
4. **CF=1.0 token-dropping isn't just one point in a three-way imbalance
   range — it's a practical necessity.** Uncapped imbalance blows past the
   entire 384MB on-chip SRAM budget by >3×; capacity-factor capping is the
   only one of the three legs of §3 (topology, bandwidth, buffering) where
   imbalance actually bites hard, even though it barely moves the
   compute-vs-comms verdict.
5. **A "boring" simulation result is a good result, not a wasted phase.**
   Phase 3 mostly confirmed Phase 2's hand-derivation rather than overturning
   it — the correct read is that this validates the reasoning chain, not that
   the tool added nothing. The place real friction *did* show up (the
   dataset-splits chunking bug, a 4× artifact from an unscrutinized inherited
   config value) was more valuable precisely because it was a methodology
   catch, not a physics surprise — the actual skill Phase 3 is meant to
   exercise.
6. **Two of Phase 2's three legs (bandwidth allocation, buffering/SRAM) turned
   out to be untestable in ASTRA-sim's analytical backend at all** — not a
   disagreement between tool and hypothesis, but a real ceiling on what a
   formula-based analytical simulator (as opposed to a packet-level one) can
   represent. Worth knowing before reaching for this tool on a similar
   question again.
7. **The cross-project synthesis only holds up as a *qualitative* argument,
   not a quantitative one** — the two projects deliberately modeled different
   reference chips (Gemmini for RTL-validatable attention work, TPU 8i for
   network-level MoE work), each for good, project-specific reasons, so their
   absolute numbers can't be combined directly. Recognizing that boundary
   is itself one of the more useful findings of Phase 4, not a failure to
   reach a cleaner number.

### Known gaps / possible extensions

- Layer 1's dense FFN (not MoE) was never incorporated into the per-layer
  arithmetic — a small, acknowledged gap (the ~1.5B-param consistency-check
  gap in §1 is presumably this).
- A prefill-regime version of the comms derivation (tokens = sequences ×
  seq_len, rather than one token/sequence/step) was flagged early and never
  built — this project stayed decode-only throughout.
- The three paper-stated balance-loss coefficients (α1/α2/α3, expert- vs.
  device- vs. communication-level) were flagged as an open research question
  in §1 but effectively superseded — the Gini/stall-time literature approach
  in §2.2 answered the imbalance-magnitude question without needing to
  resolve which coefficient grounds it.
- Rescaling attention's scratchpad/accumulator sizing to TPU-8i scale (rather
  than Gemmini scale) would turn §5's qualitative synthesis into an actual
  same-chip quantitative comparison — flagged as the natural next step there,
  not attempted.
- A packet-level ASTRA-sim backend (ns-3/htsim) could actually test §3.2 and
  §3.3 (bandwidth allocation, buffering/SRAM) instead of leaving them as
  hand-derivation only.
- Whether per-hop latency can be hidden behind concurrent compute in steady
  state (§4.3's deferred item) — would test whether the `time = max(compute,
  comm)` framing implicit throughout this project actually holds once real
  scheduling/overlap is modeled, rather than assumed.

---

## Appendix: ASTRA-sim configs (§4)

Inlined so this project doesn't depend on files left in the `~/dev/astra-sim`
clone, which is upstream `astra-sim/astra-sim` (no fork) and not backed up.
The `.et` Chakra trace files themselves aren't inlined — they're binary and
fully regenerated by the script below.

**Network configs** (`examples/network/analytical/`, analytical backend,
8 NPUs, bandwidth 2,400 GB/s = TPU 8i ICI's 19.2 Tb/s converted):

```yaml
# FullyConnected — full mesh, §4.1/§4.2. One file per latency point.
topology: [ FullyConnected ]
npus_count: [ 8 ]
bandwidth: [ 2400.0 ]  # GB/s
latency: [ 300.0 ]  # ns — also ran at 500.0, 750.0, 936.25

# Ring — same bandwidth/latency, §4.1 comparison point
topology: [ Ring ]
npus_count: [ 8 ]
bandwidth: [ 2400.0 ]  # GB/s
latency: [ 500.0 ]  # ns

# Switch — same bandwidth/latency, §4.1 comparison point
topology: [ Switch ]
npus_count: [ 8 ]
bandwidth: [ 2400.0 ]  # GB/s
latency: [ 500.0 ]  # ns
```

**System configs** (`examples/system/native_collectives/`) — identical except
`all-to-all-implementation`, paired one-to-one with the matching topology
(`direct` for FullyConnected/Switch, `ring` for Ring). `preferred-dataset-
splits: 1` is the fix from §4.2's methodology catch — the stock template this
was copied from had it at 4, which silently quadrupled every latency delta:

```json
{
    "scheduling-policy": "LIFO",
    "endpoint-delay": 10,
    "active-chunks-per-dimension": 1,
    "preferred-dataset-splits": 1,
    "all-reduce-implementation": ["ring"],
    "all-gather-implementation": ["ring"],
    "reduce-scatter-implementation": ["ring"],
    "all-to-all-implementation": ["direct"],
    "collective-optimization": "localBWAware",
    "local-mem-bw": 1600,
    "boost-mode": 0,
    "roofline-enabled": 0,
    "peak-perf": 900
}
```

**Workload generator** (`examples/workload/microbenchmarks/generator_scripts/`)
— produces the 8 per-NPU Chakra `.et` trace files for the real 2,560-byte
dispatch/combine hop (Phase 1b, `d_model=5120` @ FP4), run from the astra-sim
repo root (needs its `extern/graph_frontend/chakra` submodule):

```python
import os
from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, COMM_COLL_NODE, ALL_TO_ALL
from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import AttributeProto as ChakraAttr
from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode
from extern.graph_frontend.chakra.src.third_party.utils.protolib import encodeMessage as encode_message

NPUS = 8
COMM_SIZE_BYTES = 2560  # Phase 1b: one dispatch/combine hop, d_model=5120 @ FP4
OUT_DIR = "examples/workload/microbenchmarks/moe_project/all_to_all_8npus_2560B"

os.makedirs(OUT_DIR, exist_ok=True)
for npu in range(NPUS):
    with open(os.path.join(OUT_DIR, f"all_to_all.{npu}.et"), "wb") as et:
        encode_message(et, GlobalMetadata(version="0.0.4"))
        node = ChakraNode()
        node.id = npu
        node.name = f"all_to_all_{NPUS}npus_2560B"
        node.type = COMM_COLL_NODE
        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        node.attr.append(ChakraAttr(name="comm_type", int64_val=ALL_TO_ALL))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=COMM_SIZE_BYTES))
        encode_message(et, node)
print(f"wrote {NPUS} ET files to {OUT_DIR}")
```

Run: `AstraSim_Analytical_Congestion_Unaware --workload-configuration=.../all_to_all_8npus_2560B/all_to_all --system-configuration=direct_alltoall.json --network-configuration=FullyConnected_8npus_500ns.yml --remote-memory-configuration=.../no_memory_expansion.json`
