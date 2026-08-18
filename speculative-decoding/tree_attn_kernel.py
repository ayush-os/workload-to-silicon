# Tree-attention verification kernel -- Phase 3 (spec.md).
#
# Boilerplate only: block-pointer tiling, online-softmax bookkeeping, and
# GQA head indexing, forked from TritonFlashAttention2.flash_fwd_kernel
# (see #5's numerics-and-sparse-attn) -- forward-only, since verification is
# inference-only and nothing ever backprops through accept/reject. GQA head
# indexing (kv_head = head // GQA_GROUP) grafted in from the #5 decode
# kernels' pattern, since the original flash_fwd_kernel folded heads into
# the batch dim and had no GQA awareness at all.
#
# TODO (the real Phase 3 work, not filled in here):
#   - Represent the tree's ancestor-path structure (Medusa's per-node
#     ancestor bitmask, notes.md Phase 0) and pass it into the kernel.
#   - Replace the placeholder mask below with the real hybrid mask: full
#     causal over the real accepted prefix, ancestor-only within the
#     flattened tree portion.
#   - Decide whether tree-internal K/V (candidate tokens attending to their
#     own ancestors) get concatenated into K/V here, or handled as a
#     separate block -- not resolved by this scaffolding.
#   - Non-sequential position ids (same tree-depth = same position id,
#     Medusa's mechanism) aren't wired in at all yet -- RoPE application
#     isn't in this kernel currently.

import triton
import triton.language as tl
import torch


@triton.jit
def tree_verify_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qh, stride_qq, stride_qd,
    stride_kb, stride_kh, stride_kk, stride_kd,
    stride_vb, stride_vh, stride_vk, stride_vd,
    stride_ob, stride_oh, stride_oq, stride_od,
    stride_lb, stride_lh, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    head_index = tl.program_id(1)
    batch_index = tl.program_id(2)

    kv_head_index = head_index // GQA_GROUP

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb + head_index * stride_qh,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb + kv_head_index * stride_kh,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb + kv_head_index * stride_vh,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob + head_index * stride_oh,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb + head_index * stride_lh,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), value=float("-inf"), dtype=tl.float32)

    Qi = tl.load(Q_block_ptr)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        Kj = tl.load(K_block_ptr)
        Vj = tl.load(V_block_ptr)

        Si_j = tl.dot(Qi, tl.trans(Kj)) * scale

        # PLACEHOLDER -- plain triangular causal, standing in for the real
        # hybrid mask (causal prefix + ancestor-only tree portion). Not
        # valid tree-attention logic; replace before trusting any output.
        q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        placeholder_mask = q_idx[:, None] >= k_idx[None, :]
        Si_j = tl.where(placeholder_mask, Si_j, Si_j - 1e6)

        mi_prev = mi
        mi = tl.maximum(mi_prev, tl.max(Si_j, axis=1))

        Pi_j = tl.exp(Si_j - mi[:, None])

        correction = tl.exp(mi_prev - mi)
        li = (correction * li) + tl.sum(Pi_j, axis=1)

        Oi = tl.dot(Pi_j.to(Vj.dtype), Vj) + (correction[:, None] * Oi)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    Oi = Oi / li[:, None]
    Li = mi + tl.log(li)

    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, Li)


def tree_verify_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, gqa_group: int):
    """
    Q: [batch, n_q_heads, n_queries, d_head]
    K, V: [batch, n_kv_heads, n_keys, d_head]

    n_queries/n_keys are left generic here (not pinned to tree_len/
    seq_len_kv) -- how the flattened tree's query/key ranges get built and
    passed in is real Phase 3 design work, not resolved by this wrapper.
    """
    batch, n_q_heads, n_queries, d = Q.shape
    n_keys = K.shape[-2]
    scale = 1.0 / (d ** 0.5)

    tile_size = 64 if d * Q.element_size() <= 256 else 32
    Q_TILE_SIZE = tile_size
    K_TILE_SIZE = tile_size

    O = torch.empty_like(Q)
    L = torch.empty((batch, n_q_heads, n_queries), device=Q.device, dtype=torch.float32)

    grid = (triton.cdiv(n_queries, Q_TILE_SIZE), n_q_heads, batch)

    tree_verify_fwd_kernel[grid](
        Q, K, V,
        O, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        n_queries, n_keys,
        scale,
        D=d,
        GQA_GROUP=gqa_group,
        Q_TILE_SIZE=Q_TILE_SIZE,
        K_TILE_SIZE=K_TILE_SIZE,
    )

    return O, L
