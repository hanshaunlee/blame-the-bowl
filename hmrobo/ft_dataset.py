"""Training dataset for the frozen-policy study: FrameSamp memory built from a composite cross-session buffer.

FT_MODE (env):
  control  the released recipe: even sampling over this episode [lesson][execution], true positions
  shift    same frames; the execution frames' time positions are moved later by 32*k steps (k sampled per sample):
           the positional layout of k distractor sessions with no distractor content (isolates position robustness)
  interf   the RoboMME-Interference buffer [lesson][k units of other-task episodes][execution], even sampling over the
           composite buffer with true composite positions (what the released policy faces at evaluation)
FT_KS (env, default "0,1,3,7,15"): values of k sampled uniformly for shift/interf; FT_P_K0 (default .2): extra mass on k=0.
Features come from <dataset>/features/ep<g>.npz written by build_ft_data.py; positions are computed on the fly with a
numpy copy of PosEmb3D (checked against the JAX version in the smoke test).
"""
from __future__ import annotations
import json, os, random
from collections import OrderedDict
from pathlib import Path
import numpy as np
import ml_dtypes

from mme_vla_suite.training.dataset import RoboMMEDataset
from mme_vla_suite.shared.data_utils import even_sampling_indices

UNIT, UNIT_SRC, UNIT_STRIDE = 32, 256, 8


def pos_table(n: int, dim: int = 768, temporal_base: float = 10_000, spatial_base: float = 1_000) -> np.ndarray:
    """PosEmb3D(dim)(arange(n), spatial_size=4) in numpy: (n, 16, dim) float32; [t sin | t cos | y sin | y cos | x sin | x cos]."""
    width = dim // 6
    omega = np.arange(width, dtype=np.float64) / (width - 1)
    t_om, s_om = 1.0 / temporal_base ** omega, 1.0 / spatial_base ** omega
    pos = np.arange(n, dtype=np.float64)
    t = np.einsum("m,d->md", pos, t_om)
    temporal = np.concatenate([np.sin(t), np.cos(t)], -1)                       # (n, 2w)
    gy, gx = np.mgrid[:4, :4]
    y = np.einsum("m,d->md", (4 * gy + 2).flatten().astype(np.float64), s_om)
    x = np.einsum("m,d->md", (4 * gx + 2).flatten().astype(np.float64), s_om)
    spatial = np.concatenate([np.sin(y), np.cos(y), np.sin(x), np.cos(x)], -1)   # (16, 4w)
    out = np.concatenate([np.repeat(temporal[:, None], 16, 1), np.repeat(spatial[None], n, 0)], -1)
    return out.astype(np.float32)


class InterferenceDataset(RoboMMEDataset):
    def __init__(self, dataset_path, *a, **kw):
        super().__init__(dataset_path, *a, **kw)
        self.mode = os.environ.get("FT_MODE", "control")
        self.ks = [int(x) for x in os.environ.get("FT_KS", "0,1,3,7,15").split(",")]
        self.p_k0 = float(os.environ.get("FT_P_K0", "0.2"))
        self.max_pos = int(os.environ.get("FT_MAX_POS", "4096"))
        self.episodes = json.load(open(Path(dataset_path) / "meta" / "episodes.json"))
        self.by_task = {}
        for e in self.episodes:
            self.by_task.setdefault(e["task"], []).append(e["epis_idx"])
        self.pe = pos_table(self.max_pos)
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._cache_n = int(os.environ.get("FT_CACHE", "48"))
        self._rng = random.Random(os.getpid())
        print(f"InterferenceDataset mode={self.mode} ks={self.ks} p_k0={self.p_k0} episodes={len(self.episodes)} tasks={list(self.by_task)}", flush=True)

    # ---- episode features -------------------------------------------------------------------------------------
    def _ep(self, g: int) -> dict:
        if g in self._cache:
            self._cache.move_to_end(g); return self._cache[g]
        z = np.load(self.feature_dir / f"ep{g}.npz")
        ep = {"img": z["img"].view(ml_dtypes.bfloat16), "state": z["state"], "exec_start": int(z["exec_start"]), "task": str(z["task"])}
        self._cache[g] = ep
        if len(self._cache) > self._cache_n:
            self._cache.popitem(last=False)
        return ep

    def _unit(self, exclude_task: str) -> dict:
        tasks = [t for t in self.by_task if t != exclude_task] or list(self.by_task)
        g = self._rng.choice(self.by_task[self._rng.choice(tasks)])
        ep = self._ep(g)
        idx = np.arange(0, min(len(ep["img"]), UNIT_SRC), UNIT_STRIDE)
        idx = np.concatenate([idx, np.repeat(idx[-1:], UNIT - len(idx))]) if len(idx) < UNIT else idx[:UNIT]
        return {"img": ep["img"][idx], "state": ep["state"][idx]}

    def _sample_k(self) -> int:
        if self._rng.random() < self.p_k0:
            return 0
        return self._rng.choice(self.ks)

    def _feat(self, img, state, pos: int) -> dict:
        return {"image_emb_4x4": img[None], "pos_emb_4x4": self.pe[min(pos, self.max_pos - 1)][None], "state_emb": state}

    # ---- the hook used by RoboMMEDataset.__getitem__ ----------------------------------------------------------
    def prepare_frame_sampling(self, epis_idx: int, step_idx: int):
        budget, tpi = self.history_config.budget, self.history_config.token_per_image
        ep = self._ep(int(epis_idx)); D = ep["exec_start"]; step_idx = int(step_idx)
        k = 0 if self.mode == "control" else self._sample_k()
        feats = {}
        if self.mode == "interf" and k > 0:
            units = [self._unit(ep["task"]) for _ in range(k)]
            L = step_idx + 1 + UNIT * k                       # composite buffer length
            idx = even_sampling_indices(L - 1, budget // tpi)
            for j in idx:
                if j < D:
                    feats[j] = self._feat(ep["img"][j], ep["state"][j], j)
                elif j < D + UNIT * k:
                    u, r = divmod(j - D, UNIT)
                    feats[j] = self._feat(units[u]["img"][r], units[u]["state"][r], j)
                else:
                    i = j - UNIT * k
                    feats[j] = self._feat(ep["img"][i], ep["state"][i], j)
        else:
            idx = even_sampling_indices(step_idx, budget // tpi)
            shift = UNIT * k if self.mode == "shift" else 0
            for j in idx:
                feats[j] = self._feat(ep["img"][j], ep["state"][j], j if j < D else j + shift)
        return self.mem_buffer._prepare_frame_sampling(feats, idx, budget, tpi)
