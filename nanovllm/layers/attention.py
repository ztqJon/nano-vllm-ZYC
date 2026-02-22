import torch
from torch import nn
import triton
import triton.language as tl
# 两个不同的kernel，一个不需要拿kv cache，一个需要拿kv cache
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """每个线程处理一个token的key和value，将它们存储到kv cache中"""
    idx = tl.program_id(0) # 获取线程idx，也就是第idx个token
    slot = tl.load(slot_mapping_ptr + idx) # 从 slot_mapping 读取第idx个token的cache位置
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D) # 计算这个token的KV在输入key/value张量中的一段连续地址
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets) #  从输入（还没存到KV cache里的KV）里把该token的KV向量读出来
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D) # 计算它在KV cache中的目标地址
    tl.store(k_cache_ptr + cache_offsets, key) # 把KV存到KV cache
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor, # 所有待处理token的key和value
    value: torch.Tensor, # (N, num_heads, head_dim)
    k_cache: torch.Tensor, # kv cache的内存地址
    v_cache: torch.Tensor, 
    slot_mapping: torch.Tensor # 当前token在KV cache中的物理位置索引，形状 (N,)
):
    """
    .stride() 是 PyTorch 张量的属性，返回每个维度上的步长（stride），
    表示在该维度移动一个单位需要跳过的内存元素数。
    """
    N, num_heads, head_dim = key.shape # N is total_tokens
    D = num_heads * head_dim # D is hidden_size of k / v
    assert key.stride(-1) == 1 and value.stride(-1) == 1 # 最后一维是head_dim，内存连续
    assert key.stride(1) == head_dim and value.stride(1) == head_dim # 从一个head移动到下一个head需要跳过head_dim个元素
    assert k_cache.stride(1) == D and v_cache.stride(1) == D # 从一token的k(v) cache移动到下一个token的k(v) cache需要跳过D个元素
    assert slot_mapping.numel() == N # slot_mapping的长度等于N
    # [(N,)]：启动N个线程，每个线程负责一个token
    # key.stride(0): key跨一个token的步长，等于num_heads * head_dim
    # value.stride(0): value跨一个token的步长，等于num_heads * head_dim
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads # TP切分后的num_heads (query)
        self.head_dim = head_dim
        self.scale = scale # sqrt(d_k)
        self.num_kv_heads = num_kv_heads # TP切分后的num_kv_heads (key / value)
        self.k_cache = self.v_cache = torch.tensor([]) # 初始化空的 kv cache
        # model_runner初始化时会调用allocate_kv_cache()，为每层Attention类分配kv cache的实际内存

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context() # 获取全局变量，如is_prefill，slot_mapping，block_tables等
        k_cache, v_cache = self.k_cache, self.v_cache

        # 在做attn计算前，先写kv cache!!

        if k_cache.numel() and v_cache.numel(): # kv cache有内存才写入
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping) # 将当前 k / v 写入 kv cache

        # attn计算
        if context.is_prefill: # prefill
            if context.block_tables is not None: # block_tables不为空，说明存在prefix cache（见model_runner.py - prepare_prefill）
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else: # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o # attn_output
