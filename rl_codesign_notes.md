# RL Post-Training Codesign — Rollout vs. Training on DeepSeek-V2

Hand-derived roofline codesign for an RL post-training loop (GRPO, DeepSeek-R1-style) on a 236B MoE+MLA model — the headline result is that **colocated beats disaggregated at every chip budget tested, the opposite of the field's own stated tradeoff**, because this workload's rollout:train FLOPs ratio is so lopsided that disaggregation's fixed training-pool tax never pays for itself in the tested range.

**Stack:** hand-derived roofline (FLOPs/bytes/wall-clock, no simulator) · TPU 8t (training-optimized) / TPU 8i (inference-optimized), FP4-native · DeepSeek-V2 MoE+MLA (236B total / 21B activated) · GRPO (DeepSeek-R1)

- **Colocated wins at every tested chip budget (168→320 chips) by 4.5×→1.12×, converging but never crossing over** — directly contradicts RLinf's claimed pattern (disaggregation wastes compute small-scale, colocation stalls large-scale), a real result specific to this model's own very lopsided rollout:train ratio, not a universal law. Explicitly conditioned on fully on-device optimizer state — real systems (including this project's own anchor workload, R1) offload it to CPU, which would shrink the mechanism driving this result.
- **Adam optimizer state alone costs 2.83 TB (≥13 chips)** before a single activation is stored, and **activation recomputation isn't an optimization here — it's numerically forced**: one MLA layer's attention-score activations alone are **12.7× an entire TPU 8t's 216 GB HBM** at R=65,536, batch=1.
- **Rollout wall-clock dominates training by 2×–35× depending on chip layout**, driven almost entirely by KV-cache read bytes, not FFN weights — decode's cost is O(R²), not O(R), from the growing KV cache, and prefill is 4–5 orders of magnitude smaller than decode at any tested response length.
- **Cross-pool weight sync costs ≈2.75s and exceeds an entire training step's wall-clock by 3.4× at short response lengths (R=8,192)** — a real, load-bearing disaggregated-architecture cost, not a footnote, though one full step of staleness (the real-world default) fully absorbs it with ~43% margin.

---

## 1. Workload & setup

**Model**: DeepSeek-V2, 236B total / 21B activated params, 60 layers, MoE (160 routed + 2 shared experts, top-6 routing) + MLA attention (576-element/token/layer compressed KV cache) — same architecture and formulas as [`moe_routing_notes.md`](moe_routing_notes.md) and `disagg_and_placement_notes.md`.

**RL algorithm**: GRPO (DeepSeek-R1's own formulation, confirmed from the primary source, arXiv 2501.12948) — group-relative advantage, no critic, PPO-style clipped objective with an explicit KL term (β=0.001). **K=16** completions/prompt (R1-confirmed group size, identical across R1-Zero/R1/32B distillation), batch = 32 prompts × 16 samples = **512 completions/step**. Two response-length anchors used throughout: **R=8,192** (this repo's existing convention) and **R=65,536** (R1's own late-training generation cap).

**Hardware — the first-generation TPU split into two physically distinct chips**, real evidence against a single "colocated pool" being hardware-native at all for this generation:

| | TPU 8t (training) | TPU 8i (rollout) |
|---|---|---|
| Peak FP4 | 12.6 PFLOPS/chip (121 EFLOPS/pod) | 10.1 PFLOPS/chip |
| HBM capacity | 216 GB | 288 GB |
| HBM bandwidth | 6,528 GB/s | 8,601 GB/s |
| On-chip SRAM | 128 MB | 384 MB |
| Ridge point | ≈1,930 FLOPs/byte | ≈1,174 FLOPs/byte |
| ICI bandwidth | 19.2 Tb/s/chip (pod-internal only) | 19.2 Tb/s/chip (pod-internal only) |

No public BF16 peak exists for either chip (confirmed absent via primary source + secondary corroboration, not just unsourced) — both are marketed purely around FP4. **Decision: model training on 8t at its own native FP4 peak**, not a borrowed/derived BF16 number on the inference chip. This reframes the weight-sync question from a precision conversion into a same-precision, cross-chip transfer — and the chip split itself is real evidence *for* the disaggregated end of the design spectrum this project compares (§4), since Google ships training/inference as separate silicon by design in this generation.

**Reference model kept, not dropped**: R1's GRPO objective explicitly includes the KL term (not all GRPO variants keep it — confirmed from source, not assumed). Shares rollout's chip pool by default (pure inference, prefill-shaped log-prob pass, no training state); refreshed every 400 steps (a periodic weight-copy event, same shape as one training→rollout sync, far rarer).

---

## 2. Rollout-side roofline

Rollout generation reuses `decode_notes.md`'s FLOPs/bytes formulas and MLA's real cache formula directly — rollout *is* decode-heavy inference, the same workload shape already derived twice in this repo. The genuinely new piece: GRPO's K-way prefix sharing.

**K-way prefix sharing**: K=16 completions per prompt share one prefill pass instead of K independent ones — savings = (K−1)/K = **93.75%** of prefill's own cost avoided, R-independent, closed-form.

**FLOPs**: decode dominates prefill by **377:1 at R=8,192, 13,779:1 at R=65,536** — N (batch size) cancels out of every ratio. Decode's own accumulated cost is **not O(R)** — a `R(R−1)/2` term from the growing KV cache makes it **O(R²)** past a crossover at R\*≈7,865, reaching 89% of decode's total cost at R=65,536. This is the first project in the repo where a *single request's own response length* crosses into that regime without needing a large batch or long prompt to get there.

**Wall-clock** (prefill compute-bound confirmed at B≥3 concurrent prompts, decode memory-bound — both confirmed, not assumed, against TPU 8i's own ridge point):

| | R=8,192 | R=65,536 |
|---|---|---|
| Prefill (32 prompts) | 147.9 ms | 147.9 ms |
| Decode (512 completions, N=640 concurrent-batch capacity) | 27.67 s | 523.5 s |
| **Rollout wall-clock/step** | **≈27.82 s** | **≈523.65 s** |
| Prefill's share | 0.53% | 0.028% |

**Key finding**: rollout wall-clock is completely dominated by decode, driven specifically by **KV-cache read bytes**, not FFN weights — the memory-bound conversion makes the dominance even more extreme than raw FLOPs suggested. N=640 concurrent-batch sizing reused from `disagg_and_placement_notes.md`'s HBM-capacity grounding (R1's own 512-completion minibatch is close enough in scale for the same near-saturated regime to apply — a flagged assumption, not free reuse).

---

## 3. Training-side roofline

No formula to reuse here — this repo had only ever derived inference before this project.

**Backward FLOPs multiplier: 4×, not the naive 3×.** Pure backprop algebra gives 3× forward (fwd + grad-activation + grad-weight) regardless of MoE sparsity, as long as only actually-activated compute is counted on both sides — confirmed, not assumed. What pushes it to 4×: **activation recomputation is numerically forced at this project's own response lengths**, not a discretionary optimization. A single MLA layer's attention-score activation term alone (`5·b·n_h·s²` bytes) is **≈20% of TPU 8t's entire 216 GB HBM at R=8,192, and ≈12.7× the entire chip's capacity at R=65,536** — batch=1, one layer. Summed across all 61 layers with no recomputation: 2.7 TB to 168 TB, both far beyond any real chip. Working number: 4× forward, flagged as an upper-bound (real practice uses *selective* recomputation, landing somewhere between 3× and 4×, not resolved further).

**Adam optimizer state: 2.83 TB, a hard floor independent of MoE sparsity.** 236B *total* params (not the 21B activated — Adam's momentum/variance must persist per-parameter regardless of per-token routing) × 12 bytes/param (ZeRO's canonical mixed-precision Adam: 4 FP32 master weights + 4 FP32 momentum + 4 FP32 variance) = **2.83 TB ⇒ ≥13 TPU 8t chips just to hold optimizer state**, before weights, gradients, or activations. Real-world corroboration: R1's own reported training uses CPU optimizer offloading — independent confirmation this doesn't fit on-device, not an artifact of TPU 8t's specific capacity. **Distinct from the recomputation finding, not reinforcing it**: recomputation only touches activations (recreatable, *reduces* chip pressure); optimizer state is persistent accumulated state with no recompute equivalent, so it sets its own independent floor.

**GRPO's critic-free saving, quantified against a stated PPO baseline** (GRPO's own origin paper states the critic is "comparable to the policy model in size" — modeled as a second 236B-param copy): a flat **~2× reduction across every major training resource** — optimizer state 2.83→5.66 TB (≈13→26 chips), weights doubled, activation-recomputation burden doubled, training FLOPs/step ~4×→~8× forward-equivalent.

**Per-device parallelism config**: EP=8 alone (inherited from rollout) needs 661.5 GB/device — 3.06× over budget. Landed on **TP=4 (attention) × EP=40 (experts) = 160 devices**, clearing weights+gradients+optimizer with ~41 GB headroom and dividing 160 routed experts evenly. Validated, not just accepted: EP alone maxed to EP=160 still fails (216.6 GB, zero headroom) — attention-sharding wasn't a convenient add-on, it was structurally required. **PP turned out unnecessary** — a non-obvious result; 60 layers looked like it should force pipeline-splitting, but TP+wider-EP closed the gap without it.

**Meta-pattern carried into the architecture comparison**: Phase 1's constraint was *bandwidth* (decode is memory-bandwidth-bound); Phase 2's is *capacity* (does the whole model fit at all) — same roofline instinct, structurally different axis.

---

## 4. Colocated vs. disaggregated — the architecture comparison

**Rollout:train wall-clock split, per full RL step** — training's forward pass is prefill-shaped over the full generated sequence (seq = P+R), ×4 for forced recomputation:

| | R=8,192 | R=65,536 |
|---|---|---|
| Rollout | 27.82 s | 523.65 s |
| Training (160 devices, native 8t) | 0.80 s | 24.86 s |
| **Ratio (rollout:train)** | **≈34.6:1** | **≈21.1:1** |

Rollout dominates despite training carrying the 4× recomputation penalty, because rollout is R *sequential* memory-bound steps while training is one parallel compute-bound pass across 160 chips at once — the parallelism gap outweighs the FLOPs multiplier. The ratio never crosses over as R grows; both wall-clocks pick up a quadratic-in-R term from the same mechanism (growing-context attention/cache reads), so the ratio asymptotes to the ratio of those two quadratic coefficients: **≈16.1×**, flat by R≈10⁷.

**This 16.1× number answers a disaggregated-flavored question** (each phase independently capacity-sized — EP=8 rollout vs. 160-device training), not a colocated one. Colocated reshards the *same* pool, so training's ≥160-device floor forces rollout onto 160 chips too:

| Config | R=8,192 | R=65,536 |
|---|---|---|
| Mismatched pools (EP=8 vs. 160) | 34.6:1 | 21.1:1 |
| Matched chips, EP=160 only | 9.2:1 | 2.95:1 |
| Matched chips, sensitivity range across parallelism layouts | 2.2×–13.8× | 2.2×–13.8× |

Rollout dominates in *every* tested configuration (never below ~2:1) but the magnitude is genuinely config-sensitive, not one clean number — optimal rollout layout turns out to be a real closed-form surface, `TP* = √(A·N_r/C(R))` (A = attention's replicated-weight term, C(R) = FFN+cache's combined coefficient), not two separate "R-dependent" and "N_r-dependent" observations as it first looked.

**Colocated forces a choice of physical chip, not just matched parallelism degrees** — TPU 8t/8i are different silicon, not modes of one chip (unlike HybridFlow's homogeneous-GPU assumption). Whichever role doesn't get its native chip pays a real, recurring tax every step. **Chosen: 8i** (rollout stays native, training eats a 2.50× tax) since rollout dominates wall-clock in every tested config — tax the minority term. Tightest colocated config found: **TP=2×EP=40=80 devices on 8i** (76.4 GB headroom, more margin than 8t's own 41 GB), same layout for both modes (zero reshard — HybridFlow's "zero memory redundancy" claim holds in the *time* sense too for this specific config, because there's nothing to reshard):

| | R=8,192 | R=65,536 |
|---|---|---|
| Rollout | 7.537 s | 119.671 s |
| Training (2.50× cross-chip tax) | 2.006 s | 62.035 s |
| **Ratio** | **3.76:1** | **1.93:1** |

**Disaggregated's real chip-ratio methodology** (service time → throughput → chip count, reused directly from `disagg_and_placement_notes.md`): balancing `rollout_time(N_r) = train_time(N_t)` shows the ratio **climbs with N_r** (1.73→9.21 at R=8,192 as N_r goes 8→160) and **shrinks as R grows** at any fixed N_r — both robust findings. Converges to AReaL's real reported ~3:1 inference:training split around N_r≈32–40 at R=8,192; needs far more rollout chips to hit the same ratio at R=65,536, a genuine new finding (longer responses need proportionally more rollout chips to sustain a given balance point). **Correction caught mid-derivation**: the naive balance-point solve gives infeasible N_t values (as low as ≈5) that don't have enough HBM to hold the model — training's real capacity floor (≥160 on native 8t) is fixed, not a free variable, and sits below rollout's own achievable range at every tested N_r, so training can never be the bottleneck in disaggregated once the real capacity constraint is respected.

**Weight-broadcast topology and sync cost**: TPU 8t and 8i share **no fabric** (3D torus/Virgo vs. Boardfly hierarchical mesh — confirmed via primary source, not assumed) — real cross-pool path is **Jupiter** (Google's general DCN), 400 Gb/s/host. Single-link baseline transfer (fan-in on 8t + cross-pool over Jupiter + fan-out on Boardfly) for the full 118 GB FP4 weight payload:

| Leg | Bandwidth | Cost |
|---|---|---|
| Fan-in (8t-internal, 112.7 GB) | 19.2 Tb/s/chip | 0.047 s |
| Cross-pool (118 GB) | 400 Gb/s (Jupiter) | 2.36 s |
| Fan-out (Boardfly, 7-hop bound) | 19.2 Tb/s/chip | 0.344 s |
| **Total** | | **≈2.75 s** |

**This exceeds training's own per-step wall-clock by ≈344% at R=8,192** (≈11% at R=65,536) — a real, load-bearing disaggregated cost, not a footnote. Faster architectures exist (parallel per-device links, ≈0.106 s) but require giving up disaggregated's whole point (independent pool sizing — matched-layout parallel transfer needs N_r=N_t=160) or eating an unmodeled reshard cost (HybridFlow's real 70B-dense reshard numbers — up to 36.4% of iteration time naively, 11.7s average optimized, 78.2s worst case — are the anchor for how bad that gets at this project's 236B MoE scale).

### Item 4: the final chip-budget-matched comparison

With rollout's optimal TP applied per N_r (correcting the earlier "structural floor" — rollout time actually falls as O(1/√N_r), no hard floor once TP scales too) and N_t=160 confirmed optimal (not just minimal — training's own floor already beats rollout's achievable range at every tested N_r, so more chips would only sit idle):

| Total chips | Disaggregated `max(rollout,train)` | Colocated (serial) | Colocated wins by |
|---|---|---|---|
| 168 | 27.82 s / 523.65 s | 6.24 s / 99.57 s | **4.5× / 5.3×** |
| 200 | 10.63 s / 144.42 s | 5.76 s / 87.15 s | **1.85× / 1.66×** |
| 320 | 5.39 s / 72.27 s | 4.82 s / 64.08 s | **1.12× / 1.13×** |

**Colocated wins at every tested chip budget — the opposite of RLinf's claimed pattern.** Mechanism: disaggregated pays a fixed 160-device training tax that sits almost entirely idle (training finishes in under a second while rollout is still running), a real structural waste that only shrinks as a fraction of the total as the budget scales up; colocated never pays it, since the same chips serve both roles. Margin shrinks sharply with scale (4.5×→1.85×→1.12×) but never crosses over in the tested range.

**Decision 3 — the condition this finding rests on, stated up front rather than discovered as a caveat.** The 160-device training floor driving this entire mechanism is **~70%-composed of optimizer state bytes**. Offloading it to host memory (as R1, this project's own anchor workload, actually does) would shrink that floor toward the throughput-balance point — directly undercutting the idle-capacity mechanism the colocated-wins result rests on. **Explicitly out of scope for quantitative derivation** — modeling it needs host-device interconnect bandwidth and CPU-side Adam throughput, a fourth genuinely new research axis this project never built infrastructure for — but not silently ignored: **read the headline finding as "colocated wins, given neither architecture offloads optimizer state," not as an unconditional recommendation.** An "optimistic bookend" (assume offloading is free) was considered and rejected — colocated's own capacity floor would plausibly shrink under the same assumption too, so two stacked idealizations don't cleanly bound the truth.

**Rollout Routing Replay** (from Miles/SGLang — a real mechanism not covered by any of the six original reading-list papers): MoE routing can drift between rollout's engine and training's recomputed forward pass; replay fixes it by shipping rollout's actual routing choices to training. Bounded systems cost: routing metadata ≈1.48 GB (R=8,192) to ≈11.88 GB (R=65,536), flowing rollout→training — opposite direction from the weight sync, so it doesn't share the weight-sync's staleness slack. Checked against the real margin, not assumed away: `T_meta + train_time` still clears `rollout_time` by a wide margin even at the tightest tested point (25.10 s vs. 72.27 s at N_r=160, R=65,536) — real latency, small enough not to change steady-state throughput.

**Flagged, not chased**: whether colocated's own TP should also float with N (pinned at TP=2 here); a back-of-envelope N_r≈10,000 crossover where training *would* become the bottleneck (~60× the largest N_r tested); RLinf's claim may hold at different model scales or rollout:train ratios than this project's own 236B MoE config.

---

## 5. Staleness

**Mechanism**: disaggregated's steady-state dependency chain differs sharply by staleness level. At **N=0** (fully synchronous), rollout(i) cannot start until train(i−1) finishes *and* syncs — a strict serial chain, `train_time + sync_time + rollout_time`, nothing overlaps. At **N=1** (the real-world default — RhymeRL, StreamRL, AReaL's η≥0 formal bound all use this), rollout(i) uses weights already synced during rollout(i−1)'s own execution window, so both `train_time` and `sync_time` sit entirely in rollout's shadow and cycle_time = rollout_time alone, as long as `train_time + sync_time ≤ rollout_time`.

**N=0 is a strictly dominated architecture point, not a real design choice** — worse than *both* alternatives, not just "as bad as colocated with idle chips." Colocated's mode-switch is free in this project (same-layout-on-8i needs zero reshard — no cross-pool transfer exists to pay for at all). Disaggregated N=0 pays `train_time + sync_time` serially *on top of* idling two separate fixed pools it already had to buy. Consistent with every real system in this space defaulting to staleness≥1.

**N=1's real margin, corrected** — the naive check (sync cost alone vs. rollout's floor) understates the real requirement, which is `train_time + sync_time` together against that floor:

| | train_time(160) | sync_time | combined | rollout floor | margin |
|---|---|---|---|---|---|
| R=8,192 | 0.80 s | 2.75 s | 3.55 s | 6.18 s | **42.6%** |
| R=65,536 | 24.86 s | 2.75 s | 27.61 s | 49.46 s | **44.2%** |

Still comfortably hideable at both R anchors — but the real margin is ~43–44%, not the ~45–55% a sync-only framing would suggest. The composition shifts with R: at R=8,192 sync_time is the majority of the exposed cost (2.75 of 3.55 s); at R=65,536 train_time dominates 9:1 — staleness=1 is hiding "mostly the weight transfer" at short R and "mostly the training step itself" at long R.

**N>1 buys nothing further here — a scoped finding, not a general claim.** Since N=1 already fully hides both terms with ~43–44% margin, going further (AReaL's real η=4/η=8) provides zero additional mechanical throughput benefit *for this project's own numbers*. AReaL and StreamRL's real motivation for larger η is plausibly a different axis entirely — tail-latency/straggler smoothing across a batch of wildly-varying response lengths (flagged qualitatively in both sources, no measured utilization table in either) — that this project's mean-value roofline approach doesn't model at all. Whether staleness hurts training *quality* remains explicitly out of scope, per this project's own methodology boundary — AReaL's η=4/η=8 stays the empirical anchor, not re-derived.

---

## 6. Cross-project synthesis

- **Rollout:train resource split is the direct structural analog of `disagg_and_placement_notes.md`'s prefill:decode chip ratio** — same methodology (service time → throughput → chip count), different roles; §4's disaggregated chip-ratio derivation reused that chain directly.
- **Weight-broadcast topology reuses `moe_routing_notes.md`'s mesh-vs-switch reasoning for a genuinely different traffic pattern** — §4's Jupiter/Boardfly fan-in/cross-pool/fan-out work is the same topology instinct applied to broadcast-to-all-rollout-workers traffic, not MoE's token dispatch.
- **The precision-conversion-at-sync question dissolved rather than reusing a framework.** The original plan was to reuse the sibling `numerics-and-sparse-attn` project's coupled/decoupled precision framework for the training→rollout weight-sync boundary. Once neither TPU 8t nor 8i turned out to have a public BF16 peak (§1), the sync boundary became an FP4(8t)→FP4(8i) same-precision transfer, not a BF16→FP4 conversion — the framework was never actually invoked. Worth stating plainly rather than forcing a connection that stopped applying once the hardware picture changed.
- **The optional "real measured kernel throughput" substitution (swap `numerics-and-sparse-attn`'s real A100 decode-kernel numbers into rollout's idealized roofline, check whether §4's colocated-vs-disaggregated conclusion changes) was scoped out, not performed.** Two independent reasons: (1) the precedent this move claims to repeat — "Phase 4 Q1" in that sibling project — was never actually executed there; it was posed as an open question in that project's own handoff and the project moved straight to its README capstone without answering it. (2) Even the raw ingredient that exists (the dense kernel's real measured achieved-bandwidth, ~15–20% of A100 peak) is explicitly flagged in that project's own notes as coming from a correctness-only, deliberately unautotuned reference kernel, with its L-dependent trend left unresolved between a kernel-quality artifact and a real cache-reuse effect. Porting that number onto TPU 8i's idealized FP4 roofline (different hardware, different precision regime, different compiler) would be new modeling work, not data already in hand — exactly the condition under which this project's own scope discipline says to cut a step rather than force it.

---

## 7. Key takeaways

1. **Rollout dominates training's wall-clock in every tested configuration** (2×–35× depending on chip layout and R) — driven by KV-cache read bytes once converted to real wall-clock, not by raw FLOPs. Direction is robust; magnitude is genuinely config-sensitive, not one clean number.
2. **Colocated beats disaggregated at every chip budget tested (168→320 chips), margin shrinking 4.5×→1.12× but never crossing over** — the opposite of RLinf's claimed tradeoff, because disaggregated pays a fixed, mostly-idle 160-device training tax that colocated never pays. Explicitly conditional on fully on-device optimizer state (Decision 3) — this project's own anchor workload (R1) is known to violate that assumption via CPU offloading, which is plausibly the single biggest lever that could narrow or reverse the result.
3. **Two independent, non-reinforcing memory constraints set training's chip floor**: activation recomputation (forced by MLA's quadratic attention-score term, ~12.7× one chip's HBM at R=65,536) and Adam optimizer state (2.83 TB, ~70% of the byte footprint behind the 160-device floor) — recomputation *reduces* chip pressure, optimizer state sets an independent floor with no recompute equivalent.
4. **Weight-sync cost (2.75 s) is real and load-bearing at short response lengths** (344% of a training step at R=8,192) but fully absorbed by one step of staleness with ~43–44% margin — the real-world default (staleness=1) isn't just a convention, it's mechanically necessary and sufficient here; going further (AReaL's η=4/8) buys nothing more for this project's own numbers, though it likely solves a different problem (straggler variance) this methodology can't reach.
5. **GRPO's critic-free design is a flat ~2× resource saving** across optimizer state, weights, activation burden, and training FLOPs — not a qualitative simplicity claim, a directly quantified one against a stated PPO baseline.
6. **Chip heterogeneity (TPU 8t vs. 8i) is itself evidence for disaggregation, not just a modeling inconvenience** — Google ships training and inference as physically separate silicon in this generation, and colocation forces a real, recurring cross-chip tax on whichever role doesn't get its native chip, unlike GPU-homogeneous colocated systems (HybridFlow).

### Known gaps / open threads

- **Decision 3 (optimizer offloading)** is the single most load-bearing open question — explicitly out of scope for quantitative derivation (would need host-device bandwidth + CPU Adam throughput, a fourth new research axis), but directionally, it would shrink training's chip floor and narrow or reverse the colocated-wins finding.
- **P=1,024 prompt length** is this project's own unsourced assumption (no R1 anchor exists).
- **Selective vs. full activation recomputation**: the 4× multiplier assumes full recomputation; DeepSeek-V3's own real practice uses selective, landing somewhere between 3× and 4×, not resolved further.
- **V2 vs. V3 architecture dims**: this project uses DeepSeek-V2's numbers throughout, but the only real activation-memory source found (arXiv 2502.07846) analyzes V3's dims — same family, not identical, not reconciled.
- **Colocated's own TP-floating-with-N** and the **N_r≈10,000 training-becomes-bottleneck crossover** — both real mathematical consequences of the derivation, ~60× beyond any config this project actually tests, flagged rather than chased.
- **TP/EP communication overhead** (all-reduce/all-to-all cost) is never modeled anywhere in this project — exactly where a "does scaling TP proportionally remove rollout's floor too" question would need it, deliberately left open rather than answered with an untrustworthy idealized number.
