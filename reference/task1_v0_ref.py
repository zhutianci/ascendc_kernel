import torch
import torch.nn as nn


def sparse_attn_ref(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    valid_mask = topk_idxs >= 0
    safe_idxs  = topk_idxs.clamp(min=0).long()
    b_idx = torch.arange(b, device=q.device)[:, None, None].expand(b, m, topk)
    gathered_kv = kv[b_idx, safe_idxs]
    gathered_kv = gathered_kv.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), gathered_kv.float()) * softmax_scale
    scores = scores.masked_fill(~valid_mask.unsqueeze(2), float("-inf"))
    sink = attn_sink.float().view(1, 1, h, 1)
    max_scores = torch.amax(scores, dim=-1, keepdim=True)
    max_scores = torch.maximum(max_scores, sink)
    exp_scores = torch.exp(scores - max_scores)
    exp_scores = exp_scores.masked_fill(~valid_mask.unsqueeze(2), 0.0)
    exp_sink  = torch.exp(sink - max_scores)
    sum_exp   = exp_scores.sum(dim=-1, keepdim=True) + exp_sink
    attn_weights = exp_scores / sum_exp
    output = torch.einsum("bmht,bmtd->bmhd", attn_weights, gathered_kv.float())
    return output.to(q.dtype)


class Model(nn.Module):
    def __init__(self, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.softmax_scale = head_dim ** -0.5
        self.attn_sink = nn.Parameter(torch.zeros(n_heads, dtype=torch.float32))

    def forward(self, q, kv, topk_idxs):
        return sparse_attn_ref(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)


batch_size = 8
seq_len    = 2600
n_kv       = 32
n_heads    = 64
head_dim   = 128
topk       = 16


def get_inputs():
    q         = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=torch.bfloat16)
    kv        = torch.randn(batch_size, n_kv,   head_dim,           dtype=torch.bfloat16)
    topk_idxs = torch.randint(0, n_kv, (batch_size, seq_len, topk), dtype=torch.int32)
    return [q, kv, topk_idxs]


def get_init_inputs():
    return [n_heads, head_dim]
