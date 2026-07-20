import math
import torch
from dlinfer.utils.type_annotation import Tensor, Optional
from dlinfer.framework.lmdeploy_ext.cudagraph.ascend_cudagraph import (
    AscendGraphRunner,
    get_graph_params,
    aclgraph_use_torch_npu_update,
)


def _unpack_int4(packed_states: Tensor) -> Tensor:
    """Unpack biased signed INT4 nibbles stored by dynamic_quant_int4."""
    # torch_npu remainder/div support INT32 but not INT16.
    packed = packed_states.to(torch.int32) + 128
    low = torch.remainder(packed, 16)
    high = torch.div(packed, 16, rounding_mode="floor")
    return torch.stack((low, high), dim=-1).flatten(-2).sub(8).float()


def _validate_dynamic_kv_cache(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    key_scales: Tensor,
    value_scales: Tensor,
    quant_bits: int,
):
    if quant_bits not in (4, 8):
        raise ValueError(f"unsupported dynamic KV quant_bits={quant_bits}")
    if query.dtype != torch.bfloat16:
        raise ValueError("dynamic INT4/INT8 KV cache requires bfloat16 query")
    expected_dtype = torch.int8
    if key_cache.dtype != expected_dtype or value_cache.dtype != expected_dtype:
        raise ValueError(
            f"dynamic int{quant_bits} KV cache requires {expected_dtype} K/V cache tensors")
    if key_scales is None or value_scales is None:
        raise ValueError("dynamic INT4/INT8 KV cache requires K/V scale caches")
    expected_scale_dim = 2 if quant_bits == 4 else 1
    if key_scales.shape[-1] != expected_scale_dim or value_scales.shape[-1] != expected_scale_dim:
        raise ValueError(
            f"dynamic int{quant_bits} KV scale caches must end in size {expected_scale_dim}")
    if key_cache.shape[-1] != value_cache.shape[-1]:
        raise ValueError("dynamic INT4/INT8 KV cache does not support MLA cache layout")
    expected_head_dim = key_cache.shape[-1] * (2 if quant_bits == 4 else 1)
    if query.shape[-1] != expected_head_dim:
        raise ValueError(
            f"query head dim {query.shape[-1]} does not match int{quant_bits} cache head dim "
            f"{expected_head_dim}")


def _gather_dynamic_kv(
    cache: Tensor,
    scales: Tensor,
    block_ids: Tensor,
    kv_len: int,
    num_kv_heads: int,
    quant_bits: int,
) -> Tensor:
    packed_head_dim = cache.shape[-1]
    states = cache.index_select(0, block_ids).reshape(-1, num_kv_heads, packed_head_dim)[:kv_len]
    if quant_bits == 4:
        states = _unpack_int4(states)
    else:
        states = states.float()
    scale_dim = scales.shape[-1]
    scales = scales.index_select(0, block_ids).reshape(-1, num_kv_heads, scale_dim)[:kv_len]
    if quant_bits == 4:
        group_dim = states.shape[-1] // scale_dim
        scales = scales.unsqueeze(-1).expand(*scales.shape, group_dim).reshape(states.shape)
    return states * scales


def _dynamic_decode_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    block_table: Tensor,
    block_size: int,
    kv_seq_len: Tensor,
    softmax_scale: Optional[float],
    key_scales: Tensor,
    value_scales: Tensor,
    quant_bits: int,
) -> Tensor:
    """Eager decode over active pages for dynamically quantized KV caches."""
    _validate_dynamic_kv_cache(
        query, key_cache, value_cache, key_scales, value_scales, quant_bits)
    if num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    kv_lens = [int(length) for length in kv_seq_len.tolist()]
    if query.shape[0] != len(kv_lens):
        raise ValueError("decode query batch size does not match kv_seq_len")

    head_dim = query.shape[-1]
    scale = softmax_scale if softmax_scale else 1.0 / math.sqrt(head_dim)
    repeats = num_q_heads // num_kv_heads
    out = torch.empty_like(query)
    for batch_idx, kv_len in enumerate(kv_lens):
        block_count = math.ceil(kv_len / block_size)
        block_ids = block_table[batch_idx, :block_count].to(torch.long)
        if block_ids.numel() != block_count or torch.any(block_ids < 0):
            raise ValueError("invalid block table for dynamic KV cache")
        keys = _gather_dynamic_kv(
            key_cache, key_scales, block_ids, kv_len, num_kv_heads, quant_bits)
        values = _gather_dynamic_kv(
            value_cache, value_scales, block_ids, kv_len, num_kv_heads, quant_bits)
        keys = keys.transpose(0, 1).repeat_interleave(repeats, dim=0)
        values = values.transpose(0, 1).repeat_interleave(repeats, dim=0)
        scores = torch.einsum("hd,hkd->hk", query[batch_idx].float(), keys) * scale
        out[batch_idx] = torch.einsum("hk,hkd->hd", scores.softmax(dim=-1), values).to(query.dtype)
    return out


def decode_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    scale_value: float,
    block_table: Tensor,
    block_size: int,
    q_seq_len: Tensor,
    kv_seq_len: Tensor,
    softmax_scale: float,
    attn_output: Tensor,
    key_scales_zeros: Optional[Tensor] = None,
    value_scales_zeros: Optional[Tensor] = None,
    quant_bits: int = 0,
):
    if quant_bits == 4:
        # No fused Ascend INT4 page-attention API on the installed torch_npu.
        # Eagerly unpacking live pages preserves the packed cache footprint.
        return _dynamic_decode_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_table=block_table,
            block_size=block_size,
            kv_seq_len=kv_seq_len,
            softmax_scale=softmax_scale,
            key_scales=key_scales_zeros,
            value_scales=value_scales_zeros,
            quant_bits=quant_bits,
        )
    if quant_bits == 8:
        if key_scales_zeros is None or value_scales_zeros is None:
            raise ValueError("dynamic int8 KV cache requires K/V scale caches")
        if query.dtype != torch.bfloat16:
            raise ValueError("dynamic int8 KV cache requires bfloat16 query")
        if key_cache.dtype != torch.int8 or value_cache.dtype != torch.int8:
            raise ValueError("dynamic int8 KV cache requires int8 K/V cache tensors")
        if key_scales_zeros.shape[-1] != 1 or value_scales_zeros.shape[-1] != 1:
            raise ValueError("dynamic int8 KV scale caches must end in size 1")
        if block_size % 32:
            raise ValueError("dynamic int8 KV cache requires a block size divisible by 32")

        # npu_dynamic_quant returns one float32 scale for every token and KV
        # head.  IncreFlashAttention's Python API only supports per-channel
        # scales, so use FusedInferAttentionScore's page-aware per-token,
        # per-head mode (5) for eager decode.
        batch_size, _, head_dim = query.shape
        num_blocks = key_cache.size(0)
        key = key_cache.view(num_blocks, block_size, num_kv_heads, head_dim)
        value = value_cache.view(num_blocks, block_size, num_kv_heads, head_dim)
        key = key.permute(0, 2, 1, 3).contiguous()
        value = value.permute(0, 2, 1, 3).contiguous()
        key_scales = key_scales_zeros.squeeze(-1).permute(0, 2, 1).contiguous()
        value_scales = value_scales_zeros.squeeze(-1).permute(0, 2, 1).contiguous()
        key_zeros = torch.zeros_like(key_scales)
        value_zeros = torch.zeros_like(value_scales)
        query = query.unsqueeze(2).contiguous()

        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            block_table=block_table.to(torch.int32),
            input_layout="BNSD",
            actual_seq_lengths=[1] * batch_size,
            actual_seq_lengths_kv=kv_seq_len.tolist(),
            key_antiquant_scale=key_scales,
            key_antiquant_offset=key_zeros,
            value_antiquant_scale=value_scales,
            value_antiquant_offset=value_zeros,
            key_antiquant_mode=5,
            value_antiquant_mode=5,
            block_size=block_size,
            num_key_value_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale=softmax_scale if softmax_scale else 1.0 / math.sqrt(head_dim),
            sparse_mode=0,
        )
        return output.squeeze(2).contiguous()

    if AscendGraphRunner.capturing and not aclgraph_use_torch_npu_update():
        graph_params = get_graph_params()
        num_tokens = query.shape[0]
        stream = torch.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attn_params[num_tokens].append(
            (
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                num_q_heads,
                scale_value,
                block_table,
                kv_seq_len,
                attn_output,
            )
        )
        graph_params.is_mla = False
        torch.npu.graph_task_group_begin(stream)
        torch.ops.atb._npu_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale_value=scale_value,
            block_table=block_table,
            context_lens=kv_seq_len,
            out=attn_output,
        )
        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
    else:
        bs, _, dim = query.shape
        block_num = key_cache.size(0)
        query = query.contiguous()
        attn_output = attn_output.contiguous()
        key_cache = key_cache.view(block_num, block_size, -1)
        value_cache = value_cache.view(block_num, block_size, -1)
        scale_value = softmax_scale if softmax_scale else 1.0 / math.sqrt(dim)

        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query=query,
            key=key_cache,
            value=value_cache,
            atten_mask=None,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=q_seq_len,
            actual_seq_lengths_kv=kv_seq_len,
            num_key_value_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale=scale_value,
            sparse_mode=0,
        )
    return attn_output


def prefill_attention_dynamic(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    block_size: int,
    q_seq_len: Tensor,
    kv_seq_len: Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    softmax_scale: Optional[float],
    key_scales_zeros: Tensor,
    value_scales_zeros: Tensor,
    quant_bits: int,
) -> Tensor:
    """Eager prefill over only the live dynamic INT4/INT8 KV cache tokens.

    PromptFlashAttention cannot combine page attention with per-token K/V
    antiquantization on the installed torch_npu version.  Gathering just the
    active pages keeps the fallback bounded by request length rather than the
    complete cache capacity, while preserving dynamic scales exactly.
    """
    _validate_dynamic_kv_cache(
        query,
        key_cache,
        value_cache,
        key_scales_zeros,
        value_scales_zeros,
        quant_bits,
    )
    if num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    q_lens = [int(length) for length in q_seq_len.tolist()]
    kv_lens = [int(length) for length in kv_seq_len.tolist()]
    if len(q_lens) != len(kv_lens):
        raise ValueError("q_seq_len and kv_seq_len must have the same batch size")
    if query.shape[0] != sum(q_lens):
        raise ValueError("query tokens do not match q_seq_len")

    head_dim = query.shape[-1]
    scale = softmax_scale if softmax_scale else 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(query)
    q_offset = 0
    for batch_idx, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        if q_len > kv_len:
            raise ValueError("q_seq_len cannot exceed kv_seq_len")
        block_count = math.ceil(kv_len / block_size)
        block_ids = block_table[batch_idx, :block_count].to(torch.long)
        if block_ids.numel() != block_count or torch.any(block_ids < 0):
            raise ValueError("invalid block table for dynamic KV cache")

        keys = _gather_dynamic_kv(
            key_cache, key_scales_zeros, block_ids, kv_len, num_kv_heads, quant_bits)
        values = _gather_dynamic_kv(
            value_cache, value_scales_zeros, block_ids, kv_len, num_kv_heads, quant_bits)
        keys = keys.transpose(0, 1).repeat_interleave(num_q_heads // num_kv_heads, dim=0)
        values = values.transpose(0, 1).repeat_interleave(num_q_heads // num_kv_heads, dim=0)
        queries = query[q_offset:q_offset + q_len].transpose(0, 1).float()

        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
        q_positions = torch.arange(q_len, device=query.device) + kv_len - q_len
        kv_positions = torch.arange(kv_len, device=query.device)
        scores.masked_fill_(kv_positions.unsqueeze(0) > q_positions.unsqueeze(1), float("-inf"))
        out[q_offset:q_offset + q_len] = torch.matmul(scores.softmax(-1), values).transpose(0, 1).to(query.dtype)
        q_offset += q_len
    return out


def decode_attention_mla(
    query: Tensor,
    key_cache: Tensor,
    num_kv_heads: int,
    num_q_heads: int,
    scale_value: float,
    block_table: Tensor,
    kv_seq_len: Tensor,
    mla_vheadsize: int,
    attn_output: Tensor,
):
    if AscendGraphRunner.capturing:
        graph_params = get_graph_params()
        num_tokens = query.shape[0]
        stream = torch.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attn_params[num_tokens].append(
            (
                query,
                key_cache,
                num_kv_heads,
                num_q_heads,
                scale_value,
                block_table,
                kv_seq_len,
                mla_vheadsize,
                attn_output,
            )
        )
        graph_params.is_mla = True
        torch.npu.graph_task_group_begin(stream)
        torch.ops.atb._npu_paged_attention_mla(
            query=query,
            key_cache=key_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale_value=scale_value,
            block_table=block_table,
            context_lens=kv_seq_len,
            mla_vheadsize=mla_vheadsize,
            out=attn_output,
        )
        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
    else:
        torch.ops.atb._npu_paged_attention_mla(
            query=query,
            key_cache=key_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale_value=scale_value,
            block_table=block_table,
            context_lens=kv_seq_len,
            mla_vheadsize=mla_vheadsize,
            out=attn_output,
        )
    return attn_output
