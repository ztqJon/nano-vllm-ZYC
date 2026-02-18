import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

"""基于layers中的各种组件搭建Qwen3模型"""

class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int, # GQA，num_kv_heads != num_heads
        max_position: int = 4096 * 32, # 128K
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size() # TP数
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_scaling = None

        self.rotary_emb = get_rope(
            self.head_dim, # 每个头的维度
            rotary_dim=self.head_dim, # 每个头的维度
            max_position=max_position, # 4K * 32
            base=rope_theta, # 基频
            rope_scaling=rope_scaling, # 缩放因子, 无
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states) # qkv: (total_tokens/seq_num, 3 * hidden_size)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1) # q, k, v: (total_tokens/seq_num, hidden_size)

        # GQA (Group Query Attention)
        q = q.view(-1, self.num_heads, self.head_dim) # q: (total_tokens/seq_num, num_heads, head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim) # k: (total_tokens/seq_num, num_kv_heads, head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim) # v: (total_tokens/seq_num, num_kv_heads, head_dim)
        if not self.qkv_bias: # 无qk_bias，做qk_norm
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k) # 调用的是RotaryEmbedding.forward函数
        o = self.attn(q, k, v) # attn: (total_tokens/seq_num, num_heads, head_dim)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """
    先列后行
    gate_up -> Column
    down -> Row
    """
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str) -> None:
        super().__init__()
        # 把gate_proj和up_proj合并成一个线性层进行计算
        # 根据Qwen3-0.6B的config.json：
        # intermediate_size = 3072, 
        # hidden_size = 1024,
        # hidden_act: "silu"
        self.gate_up_proj = MergedColumnParallelLinear( # 竖着切/按列切分
            hidden_size,
            [intermediate_size] * 2, # 2 = gate + up, [3072, 3072]
            bias=False,
        )
        self.down_proj = RowParallelLinear( # 横着切/按行切分
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None: # 第一层residual为None，直接把hidden_states作为residual
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """
    Qwen3Model = VocabParallelEmbedding + Qwen3DecoderLayer * num_hidden_layers + RMSNorm
    """
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None # 初始residual为None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual) # 最后的输出不需要residual
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3ForCausalLM = Qwen3Model + ParallelLMHead
    """
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.model = Qwen3Model(config) # 默认Qwen3Config但没用到，用的是hf_config
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings: # False 头尾不共享权重
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        prefill:
            input_ids: [total_tokens]
            positions: [total_tokens]
        decode:
            input_ids: [seq_num] the last token id of each seq
            positions: [seq_num] the position of the last token of each seq
        """
        return self.model(input_ids, positions) # 计算Qwen3Model的hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states) # 计算logits，在ModelRunner.run_model中调用
