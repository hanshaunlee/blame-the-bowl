"""Learned session router: which history session should fill the long-term memory slots?

Input per session: the VLA's own pooled SigLIP features of its frames (2048-d, the same features the
cosine-retrieval baseline uses). Query: the feature of the current observation. The router attends
from the query over each session's frames (learned metric) and scores the attention-pooled summary;
it is length-invariant on purpose (the lesson is longer than a distractor unit, and that cue must
not be used). Trained with session labels on the *train* split (hmrobo/train_router.py), used
label-free at test time as memory system `learned_route` (hmrobo/mm_policy.py).
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn


class SessionRouter(nn.Module):
    def __init__(self, emb_dim: int = 2048, d: int = 256):
        super().__init__()
        self.d = d
        self.norm = nn.LayerNorm(emb_dim)
        self.kenc = nn.Sequential(nn.Linear(emb_dim, d), nn.GELU(), nn.Linear(d, d))
        self.venc = nn.Sequential(nn.Linear(emb_dim, d), nn.GELU(), nn.Linear(d, d))
        self.qenc = nn.Sequential(nn.Linear(emb_dim, d), nn.GELU(), nn.Linear(d, d))
        self.head = nn.Sequential(nn.Linear(3 * d + 2, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, q, sessions):
        return self.score_sessions(q, sessions)

    def score_sessions(self, q: torch.Tensor, sessions: list[torch.Tensor]) -> torch.Tensor:
        """q: (emb,), sessions: list of (n_i, emb) -> (S,) logits."""
        qe = self.qenc(self.norm(q))
        qn = q / (q.norm() + 1e-8)
        out = []
        for s in sessions:
            sn = self.norm(s)
            k, v = self.kenc(sn), self.venc(sn)
            sim = (k @ qe) / math.sqrt(self.d)                     # (n_i,)
            att = torch.softmax(sim, 0)
            pooled = att @ v                                       # (d,)
            cos = ((s / (s.norm(dim=1, keepdim=True) + 1e-8)) @ qn)  # raw cosine, the retrieval baseline's signal
            feats = torch.cat([qe, pooled, qe * pooled, sim.max().reshape(1), cos.max().reshape(1)])
            out.append(self.head(feats).squeeze(0))
        return torch.stack(out)


def route(router: SessionRouter, F: np.ndarray, session_bounds, T: int) -> int:
    """Index of the chosen session (argmax logit). F: (>=T+1, emb) features of all frames in the buffer."""
    with torch.no_grad():
        q = torch.from_numpy(np.ascontiguousarray(F[T])).float()
        sess = [torch.from_numpy(np.ascontiguousarray(F[int(a):int(b)])).float() for a, b in session_bounds]
        return int(torch.argmax(router.score_sessions(q, sess)).item())


def cosine_route(F: np.ndarray, session_bounds, T: int) -> int:
    f = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    c = f[T]
    return int(np.argmax([float((f[int(a):int(b)] @ c).max()) if b > a else -2.0 for a, b in session_bounds]))
