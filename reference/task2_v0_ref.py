# -*- coding: utf-8 -*-
# 官方 Task02 参考实现（v0）。
# 注意：官方原文有两处会被 auto_bench.py 的 _filter_module_ast 丢弃的模块级赋值：
#   args = ModelArgs(...)              (Call, 非字面量 -> 被丢弃 -> get_inputs NameError)
#   default_dtype = torch.bfloat16     (Attribute, 非字面量 -> 被丢弃 -> Linear NameError)
# 本文件把二者移入函数体（_make_args / 内联 torch.bfloat16），数值语义与原文完全一致。
# 建议同时向主办方反馈该问题（官方 v0 原文无法通过其自身评测器加载）。
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from contextlib import contextmanager
import math

world_size = 1
rank = 0
block_size = 128
fp4_block_size = 32

@dataclass
class ModelArgs:
    max_batch_size: int = 4
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"
    expert_dtype: Literal[None, "fp4"] = None
    scale_dtype: Literal["fp32", "fp8"] = "fp8"
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 4096
    n_layers: int = 7
    n_hash_layers: int = 0
    n_mtp_layers: int = 1
    n_heads: int = 64
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.
    swiglu_limit: float = 0.
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or torch.bfloat16          # 原文: default_dtype（会被过滤器丢弃）
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)
    def forward(self, x):
        return linear(x, self.weight, self.bias)

def linear(x, weight, bias=None):
    assert bias is None
    return F.linear(x, weight)

class ColumnParallelLinear(Linear):
    def __init__(self, in_features, out_features, bias=False, dtype=None):
        assert out_features % world_size == 0
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)
    def forward(self, x):
        return linear(x, self.weight, self.bias)

def apply_rotary_emb(x, freqs_cis, inverse=False):
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y

class Model(torch.nn.Module):
    def __init__(self, args, freqs_cis, kv_cache, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio
        self.kv_cache = kv_cache
        self.freqs_cis = freqs_cis

    def forward(self, x, qr, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            mask = torch.arange(seqlen // ratio).cuda().repeat(seqlen, 1) >= torch.arange(1, seqlen + 1).cuda().unsqueeze(1) // ratio
            index_score += torch.where(mask, float("-inf"), 0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1).cuda().unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs += offset
        return topk_idxs


def _make_args():
    # 原文: 模块级 args = ModelArgs(...)（会被过滤器丢弃，故移入函数）
    return ModelArgs(max_batch_size=8, max_seq_len=2600, dim=1024, index_n_heads=16,
                     index_head_dim=64, index_topk=128, q_lora_rank=256, rope_head_dim=32)

def get_inputs():
    a = _make_args()
    batch_size = 8
    seq_len = 2600
    x = torch.randn(batch_size, seq_len, a.dim, dtype=torch.bfloat16).cuda()
    qr = torch.randn(batch_size, seq_len, a.q_lora_rank, dtype=torch.bfloat16).cuda()
    start_pos = 0
    offset = 0
    return [x, qr, start_pos, offset]

def get_init_inputs():
    a = _make_args()
    compress_ratio = 4
    max_seq_len = a.max_seq_len
    rope_theta = 10000.0
    freqs = 1.0 / (rope_theta ** (torch.arange(0, a.rope_head_dim, 2)[:a.rope_head_dim//2].float() / a.rope_head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float().cuda()
    freqs_cis = torch.polar(torch.ones_like(freqs).cuda(), freqs).view(max_seq_len, -1)
    kv_cache = torch.randn(a.max_batch_size, a.max_seq_len // compress_ratio, a.index_head_dim, dtype=torch.bfloat16).cuda()
    return [a, freqs_cis, kv_cache, compress_ratio]
