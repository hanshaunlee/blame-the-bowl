"""Learned memory-management policies operating on the frozen student's unit embeddings.

OnlineWriter      : at each event, choose (slot j, modality m) to write the event into, or DROP.
Consolidator      : at a session end, choose B items out of {old slots} U {session events x 2
                    modalities} by Plackett-Luce sampling without replacement (hindsight: it
                    sees the whole session; query-agnostic: it does not see the next task).
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class _TF(nn.Module):
    def __init__(self, d, n_layers=2, n_heads=4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

    def forward(self, x, pad_mask=None):
        return self.tf(x, src_key_padding_mask=pad_mask)


class OnlineWriter(nn.Module):
    def __init__(self, d=128, ctx_dim=1, feat_dim=11):
        super().__init__()
        self.d = d
        self.inp = nn.Linear(d, d)
        self.feat = nn.Linear(feat_dim, d)     # age/place feats of each token
        self.role = nn.Embedding(4, d)         # 0 ctx, 1 cand sym, 2 cand frame, 3 slot
        self.ctx = nn.Linear(ctx_dim, d)
        self.empty = nn.Parameter(torch.zeros(d))
        self.tf = _TF(d)
        self.slot_head = nn.Linear(d, 2)       # write cand (sym / frame) into this slot
        self.drop_head = nn.Linear(d, 1)

    def forward(self, ctx, cand, cand_feat, slots, slot_feat, slot_present):
        """ctx (N,ctx_dim); cand (N,2,d) base embeddings; cand_feat (N,feat); slots (N,B,d) finished
        embeddings; slot_feat (N,B,feat); slot_present (N,B). Returns logits (N, 2B+1): [slot0 sym,
        slot0 frm, slot1 sym, ..., DROP]."""
        N, B, d = slots.shape
        c = self.ctx(ctx).unsqueeze(1) + self.role.weight[0]
        cs = self.inp(cand) + self.feat(cand_feat).unsqueeze(1) + self.role.weight[1:3]
        s = torch.where(slot_present.unsqueeze(-1), self.inp(slots), self.empty.expand(N, B, d)) + self.feat(slot_feat) + self.role.weight[3]
        h = self.tf(torch.cat([c, cs, s], 1))
        slot_logits = self.slot_head(h[:, 3:]).reshape(N, 2 * B)
        drop = self.drop_head(h[:, 0])
        return torch.cat([slot_logits, drop], 1)


class Consolidator(nn.Module):
    def __init__(self, d=128, ctx_dim=1, feat_dim=12):
        super().__init__()
        self.inp = nn.Linear(d, d)
        self.feat = nn.Linear(feat_dim, d)
        self.role = nn.Embedding(2, d)  # 0 ctx, 1 item
        self.ctx = nn.Linear(ctx_dim, d)
        self.tf = _TF(d)
        self.score = nn.Linear(d, 1)

    def forward(self, ctx, items, item_feat, item_mask):
        """ctx (N,ctx_dim); items (N,M,d); item_feat (N,M,feat); item_mask (N,M) valid. Returns scores (N,M)
        with -inf on invalid."""
        c = self.ctx(ctx).unsqueeze(1) + self.role.weight[0]
        x = self.inp(items) + self.feat(item_feat) + self.role.weight[1]
        pad = torch.cat([torch.zeros_like(item_mask[:, :1]), ~item_mask], 1)
        h = self.tf(torch.cat([c, x], 1), pad_mask=pad)
        s = self.score(h[:, 1:]).squeeze(-1)
        return s.masked_fill(~item_mask, float("-inf"))


def plackett_luce_sample(scores, k, greedy=False):
    """Sample k items without replacement from softmax(scores) row-wise. scores (N,M) with -inf for
    invalid. Returns idx (N,k) (-1 where fewer than k valid) and total log-prob (N,)."""
    N, M = scores.shape
    s = scores.clone()
    idx = torch.full((N, k), -1, dtype=torch.long, device=scores.device)
    logp = torch.zeros(N, device=scores.device)
    for j in range(k):
        valid = torch.isfinite(s).any(1)
        if not valid.any():
            break
        lp = F.log_softmax(s, 1)
        if greedy:
            pick = lp.argmax(1)
        else:
            probs = lp.exp().nan_to_num(0.0)
            probs = torch.where(valid.unsqueeze(1), probs, torch.ones_like(probs) / M)
            pick = torch.multinomial(probs, 1).squeeze(1)
        pick = torch.where(valid, pick, torch.full_like(pick, -1))
        logp = logp + torch.where(valid, lp.gather(1, pick.clamp(min=0).unsqueeze(1)).squeeze(1), torch.zeros_like(logp))
        idx[:, j] = pick
        s = s.scatter(1, pick.clamp(min=0).unsqueeze(1), float("-inf")) if valid.any() else s
        s = torch.where(valid.unsqueeze(1), s, scores)  # keep rows without valid untouched
    return idx, logp
