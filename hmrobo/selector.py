"""Learned budgeted frame selector (memory manager) for a frozen FrameSamp VLA.

Input: per-frame features of every frame in the policy's memory buffer (the VLA's own pooled
SigLIP tokens, 2048-d) plus position features and the current observation; output: a
Plackett-Luce distribution over subsets of MEM frames. Trained with GRPO on the frozen VLA's
flow-matching loss given the selected memory (decision distortion). Baselines share the budget:
uniform (FrameSamp's even sampling, the released policy's default), recency, retrieval, and a
lesson oracle that knows which session the query depends on.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np

MEM = 32        # FrameSamp budget: 512 tokens / 16 tokens per frame
POS_DIM = 4


def pos_features(n: int, H: int) -> np.ndarray:
    """Position features for frames 0..n-1; history (previous sessions) = [0,H), current episode = [H,n)."""
    i = np.arange(n, dtype=np.float32)
    is_exec = (i >= H).astype(np.float32)
    within = np.where(i >= H, (i - H) / max(n - H, 1), i / max(H, 1))
    rec = (n - 1 - i) / max(n - 1, 1)
    return np.stack([i / max(n - 1, 1), is_exec, within, rec], 1).astype(np.float32)


class FrameSelector(nn.Module):
    def __init__(self, emb_dim: int = 2048, d: int = 256, n_layers: int = 2, n_heads: int = 4, pos_dim: int = POS_DIM):
        super().__init__()
        self.cfg = dict(emb_dim=emb_dim, d=d, n_layers=n_layers, n_heads=n_heads, pos_dim=pos_dim)
        self.norm_in = nn.LayerNorm(emb_dim)
        self.inp = nn.Linear(emb_dim, d)
        self.pos = nn.Linear(pos_dim, d)
        self.cur = nn.Linear(emb_dim, d)
        self.role = nn.Embedding(2, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.score = nn.Linear(d, 1)

    def forward(self, emb, pos, cur, mask):
        """emb (B,N,E), pos (B,N,P), cur (B,E), mask (B,N) valid -> scores (B,N), -inf where invalid."""
        x = self.inp(self.norm_in(emb)) + self.pos(pos) + self.role.weight[1]
        c = (self.cur(self.norm_in(cur)) + self.role.weight[0]).unsqueeze(1)
        pad = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], 1)
        h = self.tf(torch.cat([c, x], 1), src_key_padding_mask=pad)
        s = self.score(h[:, 1:]).squeeze(-1)
        return s.masked_fill(~mask, float("-inf"))


def pl_sample(scores, m, greedy=False):
    """Sample m indices without replacement (Plackett-Luce). Returns (list of sorted LongTensors, logp (B,))."""
    B, N = scores.shape
    s = scores.clone(); logp = torch.zeros(B, device=scores.device)
    idx = torch.full((B, m), -1, dtype=torch.long, device=scores.device)
    for j in range(m):
        valid = torch.isfinite(s).any(1)
        if not valid.any():
            break
        lp = F.log_softmax(s, 1)
        pick = lp.argmax(1) if greedy else torch.multinomial(lp.exp().nan_to_num(0.0), 1).squeeze(1)
        logp = logp + torch.where(valid, lp.gather(1, pick.unsqueeze(1)).squeeze(1), torch.zeros_like(logp))
        idx[:, j] = torch.where(valid, pick, torch.full_like(pick, -1))
        s = s.scatter(1, pick.unsqueeze(1), float("-inf"))
    return [torch.sort(idx[b][idx[b] >= 0]).values for b in range(B)], logp


def even_over(cand, m: int = MEM):
    """FrameSamp's even sampling restricted to an ordered candidate list (truncating linspace, as theirs)."""
    cand = list(cand)
    if len(cand) <= m:
        return cand
    return [cand[j] for j in np.linspace(0, len(cand) - 1, m, dtype=int)]
