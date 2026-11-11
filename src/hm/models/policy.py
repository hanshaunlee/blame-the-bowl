"""Student policy: a small transformer over [current + W past frames, context, B memory slots].

Memory slots are typed units: either a symbolic event vector (what a language memory would
hold) or a frame snapshot (what a video memory would hold), each with age features.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..env.world import EVENT_DIM, PROPRIO_DIM, INSTR_DIM, N_ACTIONS, IMG

AGE_DIM = 3  # same-session flag, session delta, step delta


class CNN(nn.Module):
    def __init__(self, d: int, coord: bool = False):
        super().__init__()
        self.coord = coord
        if coord:  # CoordConv: append normalised x/y channels so positions are easy to read
            ys, xs = torch.meshgrid(torch.linspace(-1, 1, IMG), torch.linspace(-1, 1, IMG), indexing="ij")
            self.register_buffer("grid", torch.stack([xs, ys])[None])
        self.net = nn.Sequential(
            nn.Conv2d(3 + (2 if coord else 0), 32, 5, 2, 2), nn.GELU(),      # 32
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),     # 16
            nn.Conv2d(64, 64, 3, 2, 1), nn.GELU(),     # 8
            nn.Conv2d(64, 64, 3, 2, 1), nn.GELU(),     # 4
            nn.Flatten(), nn.Linear(64 * 4 * 4, d), nn.LayerNorm(d))

    def forward(self, x):  # x: (N,3,64,64) float in [0,1]
        if self.coord:
            x = torch.cat([x, self.grid.expand(x.shape[0], -1, -1, -1)], 1)
        return self.net(x)


class UnitEncoder(nn.Module):
    """Embeds memory units of either modality into d. Shares the CNN with the policy."""

    def __init__(self, d: int, cnn: CNN, age_dim: int = AGE_DIM):
        super().__init__()
        self.cnn = cnn
        self.sym = nn.Sequential(nn.Linear(EVENT_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.age = nn.Linear(age_dim, d)
        self.modality = nn.Embedding(2, d)   # 0 symbolic, 1 frame
        self.empty = nn.Parameter(torch.zeros(d))

    def forward(self, present, modality, sym, frames, age):
        """present (N,B) bool; modality (N,B) long; sym (N,B,EVENT_DIM); frames (N,B,3,H,W);
        age (N,B,AGE_DIM). Frames only need to be valid where modality==1."""
        N, B = present.shape
        d = self.empty.numel()
        out = self.empty.expand(N, B, d).clone()
        e_sym = self.sym(sym)
        fm = (modality == 1) & present
        e = torch.where((modality == 0).unsqueeze(-1), e_sym, torch.zeros_like(e_sym))
        if fm.any():
            f = self.cnn(frames[fm])
            e = e.masked_scatter(fm.unsqueeze(-1).expand_as(e), f)
        e = e + self.age(age) + self.modality(modality)
        out = torch.where(present.unsqueeze(-1), e, out)
        return out


class Policy(nn.Module):
    def __init__(self, d: int = 128, n_layers: int = 3, n_heads: int = 4, n_frames: int = 4,
                 ctx_dim: int = PROPRIO_DIM + INSTR_DIM, age_dim: int = AGE_DIM, coord: bool = False, fix_dim: int = 13):
        super().__init__()
        self.d = d
        self.cnn = CNN(d, coord)
        self.units = UnitEncoder(d, self.cnn, age_dim)
        self.ctx = nn.Sequential(nn.Linear(ctx_dim, d), nn.GELU(), nn.Linear(d, d))
        self.fix = nn.Sequential(nn.Linear(fix_dim, d), nn.GELU(), nn.Linear(d, d))
        self.frame_pos = nn.Embedding(n_frames, d)
        self.type_emb = nn.Embedding(5, d)  # 0 cls, 1 frame, 2 ctx, 3 memory, 4 fixture
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, N_ACTIONS))
        self.n_frames = n_frames

    def forward(self, frames, ctx, mem, mapt):
        """frames (N,F,3,H,W) float; ctx (N,ctx_dim); mem: dict with present, modality, sym, frames, age;
        mapt (N,N_FIX,fix_dim) fixture tokens."""
        N, Fn = frames.shape[:2]
        f = self.cnn(frames.flatten(0, 1)).view(N, Fn, self.d) + self.frame_pos.weight[:Fn] + self.type_emb.weight[1]
        c = self.ctx(ctx).unsqueeze(1) + self.type_emb.weight[2]
        m = self.units(**mem) + self.type_emb.weight[3]
        g = self.map_tokens(mapt)
        x = torch.cat([self.cls.expand(N, 1, self.d) + self.type_emb.weight[0], f, c, g, m], 1)
        h = self.tf(x)
        return self.head(h[:, 0])

    def map_tokens(self, mapt):
        return self.fix(mapt) + self.type_emb.weight[4]

    # ---- factored interface (used by rollouts and writer training with a frozen policy) ----
    def frame_tokens(self, frames):
        """frames (N,F,3,H,W) -> (N,F,d) including positional/type embeddings."""
        N, Fn = frames.shape[:2]
        return self.cnn(frames.flatten(0, 1)).view(N, Fn, self.d) + self.frame_pos.weight[:Fn] + self.type_emb.weight[1]

    def ctx_tokens(self, ctx):
        return self.ctx(ctx) + self.type_emb.weight[2]

    def unit_base_sym(self, sym):      # (M,EVENT_DIM) -> (M,d)
        return self.units.sym(sym) + self.units.modality.weight[0]

    def unit_base_frame(self, frames):  # (M,3,H,W) -> (M,d)
        return self.cnn(frames) + self.units.modality.weight[1]

    def unit_finish(self, base, age):   # add age/place term
        return base + self.units.age(age)

    def forward_tokens(self, frame_tok, ctx_tok, mem_tok, present, map_tok):
        """frame_tok (N,F,d), ctx_tok (N,d), mem_tok (N,B,d) already finished, present (N,B) bool, map_tok (N,N_FIX,d)."""
        N = frame_tok.shape[0]
        m = torch.where(present.unsqueeze(-1), mem_tok, self.units.empty.expand_as(mem_tok)) + self.type_emb.weight[3]
        x = torch.cat([self.cls.expand(N, 1, self.d) + self.type_emb.weight[0], frame_tok, ctx_tok.unsqueeze(1), map_tok, m], 1)
        return self.head(self.tf(x)[:, 0])
