# RL Post-Training Codesign — Notes

Working notes for project #6 (`rl-codesign`) — Rollout vs. Training. See
`spec.md` for the full phase plan; see `handoff.md` for current status and
where to pick up next. This file is a running log, appended to phase by
phase — not yet a polished writeup (that consolidation happens at project
completion, matching how `disagg_and_placement_notes.md` and the
sibling `numerics-and-sparse-attn/README.md` were finished).

---

## Phase 0 — Setup

### Reading list, synthesized

**Colocated/resharding:**

- **HybridFlow** (arXiv 2409.19256): 3D-HybridEngine resharding —
  separate weight buffers for train vs. generation mode; transition =
  reload offloaded generation weights to GPU + concurrent intra-group
  all-gathers to re-partition into the target 3D layout. **"Zero memory
  redundancy" is a memory-only claim, not a time claim** — real, measured
  transition cost: naive resharding eats up to **36.4%** of iteration time
  on a 70B model when stages sit on different devices; HybridFlow's engine
  cuts that by 55.2% average (~11.7s), up to 89.1% (~78.2s) — reduced,
  never eliminated.
- **RLHFuse** (arXiv 2409.13221 — found fresh via search, not in the
  original spec's reading list): two distinct colocated bottlenecks —
  generation stragglers (long-tailed response lengths) and training
  pipeline bubbles — fixed via dynamically migrating straggler samples to
  dedicated instances mid-iteration, repurposing freed GPUs. A third
  architecture point, more granular than pure colocated/disaggregated.

**Disaggregated/asynchronous:**

- **RhymeRL** (arXiv 2508.18588): 3 pools (rollout/reward/train), strict
  **one-step staleness** enforced (not more), weight sync via host-memory
  buffer async propagation, speculative decoding using historical rollout
  tokens as drafts (65–79% acceptance).
- **AReaL** — the real staleness-quantification anchor for Phase 4. Formal
  bound `⌊(N_r−1)/B⌋ ≤ i + η`, **η=4 recommended for coding, η=8 for math
  reasoning** (η=0 = fully synchronous). Correction mechanism: "decoupled
  PPO" — importance ratio taken against a *proximal* policy (params just
  before the current update), not the original behavior policy. Reported
  speedup vs. synchronous: 1.49×–2.27× (1.5B→32B models), up to 2.77×
  headline. ~75%/25% inference/training GPU split on a 512-GPU H800 setup.
- **StreamRL**: one-step async (staleness=1), dynamic DP-unit-granularity
  resource rebalancing, skew-aware dispatch.
- **BiDiRL**: the real hybrid, hot-switch end of the spectrum — GPUs
  default to a primary role but can temporarily execute the other's work,
  gated by predicted-time-saved vs. measured switch cost (3.6–7.7s per
  switch, 2B–4B models). **Caveat**: the paper's "idleness is a problem on
  both sides" claim is qualitative (timeline diagrams), not a measured
  utilization table — don't cite it as a quantified number later.

**Workload grounding — DeepSeek-R1 (arXiv 2501.12948):**

- GRPO confirmed: no critic, group-relative advantage = mean/std
  normalization within group (Eq. 3), PPO-style clipped objective +
  explicit KL term (Eq. 1, β=0.001), Schulman k3 KL estimator (Eq. 2).
- **K = 16** (group size), confirmed identical across R1-Zero, R1's first
  RL stage, and the 32B distillation ablation. Batch = 32 prompts × 16
  samples = 512 completions/step.
- **KL penalty is kept, not dropped** — but the reference model is **not
  literally frozen**: swapped for the latest policy every 400 steps.
- Response length: max generation length 32,768 tokens before step 8.2k,
  then **65,536** afterward — grows over training, not a fixed anchor.
- Infra: 512×H800 GPUs, R1's RL stage ≈80 hours. **Training precision not
  stated** anywhere in the paper.

**Miles** (SGLang/RadixArk — found mid-Phase-1 via a direct question, not
in the original reading list, worth folding in retroactively): a real,
current production framework implementing *both* colocated and
disaggregated layouts as a launch-time config choice (Ray placement
groups), not a research pick of one side. Unified BF16/FP8/MXFP8/INT4-QAT
precision spanning rollout+training — real confirmation the
precision-conversion-at-boundary problem is a genuine production concern,
not a hypothetical the spec invented. Weight sync over dedicated NCCL/RDMA
channels, Ray handling only the control path. Fully async mode
(queue-based, no per-iteration blocking) — no explicit staleness bound
given (unlike AReaL's η). **New scope flag, not in the original spec**:
"Rollout Routing Replay" — MoE expert-routing decisions can drift between
rollout (SGLang) and training (Megatron) recomputation, destabilizing the
policy update if uncorrected. Directly relevant since this project's
workload reuses `moe-routing-notes.md`'s formulas (DeepSeek-style MoE +
MLA) — not addressed by any of the six original reading-list papers.
**Not yet scoped in or out — flagged for Phase 2/3.**
