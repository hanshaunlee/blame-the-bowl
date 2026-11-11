"""Trial generation and the behaviour-cloning dataset.

Information-matched labels: the BC target for a step is the expert's action computed from
exactly the information the student is given (the sampled memory units plus events visible
in the working-memory window), not from the full history. So the student learns the best
response to *whatever* its memory contains, and the value of a memory system is measured
cleanly downstream.
"""
from __future__ import annotations
import numpy as np, pickle, torch
from torch.utils.data import Dataset
from ..env.world import *
from ..env.session import *
from ..env.oracle import expert_from_state

PLACE_DIM = 8
MAP_DIM = 2 * (N_DRAWERS + N_BUTTONS + 5)   # fixture coordinates (the robot knows the room map)
CTX_DIM = PROPRIO_DIM + INSTR_DIM + PLACE_DIM + MAP_DIM


N_FIX = N_DRAWERS + N_BUTTONS + 5
FIX_DIM = 7 + 4 + 2   # fixture type one-hot, index one-hot, coordinates


def layout_tokens(L: Layout) -> np.ndarray:
    """One token per fixture: [type(7) | index(4) | row, col]."""
    out = np.zeros((N_FIX, FIX_DIM), dtype=np.float32)
    k = 0
    for ftype, pts in ((0, L.drawers), (1, [L.goal]), (2, [L.bin]), (3, [L.dispenser]), (4, [L.cue]), (5, L.buttons), (6, [L.cabinet])):
        for idx, (r, c) in enumerate(pts):
            out[k, ftype] = 1.0; out[k, 7 + idx] = 1.0; out[k, 11] = r / (G - 1); out[k, 12] = c / (G - 1); k += 1
    return out


def layout_vec(L: Layout) -> np.ndarray:
    pts = list(L.drawers) + [L.goal, L.bin, L.dispenser, L.cue] + list(L.buttons) + [L.cabinet]
    return (np.array(pts, dtype=np.float32) / (G - 1)).reshape(-1)
TRIAL_MIX = (  # (weight, spec)
    (1.0, {"kind": "single", "task": "find"}),
    (1.0, {"kind": "single", "task": "count"}),
    (1.0, {"kind": "single", "task": "sequence"}),
    (1.0, {"kind": "single", "task": "adapt"}),
    (0.6, {"kind": "single", "task": "tidy"}),
    (0.8, {"kind": "cross", "k": 0}),
    (0.8, {"kind": "cross", "k": 1}),
    (0.8, {"kind": "cross", "k": 3}),
    (0.8, {"kind": "cross", "k": 7}),
)


def place_codes(n_layouts: int, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    return (r.random((n_layouts, PLACE_DIM)) > 0.5).astype(np.float32)


def generate(n: int, seed: int) -> list[Trial]:
    rng = np.random.default_rng(seed)
    w = np.array([m[0] for m in TRIAL_MIX]); w = w / w.sum()
    out = []
    for i in range(n):
        spec = TRIAL_MIX[int(rng.choice(len(TRIAL_MIX), p=w))][1]
        tr = make_trial(rng, spec)
        tr.place = place_codes(len(tr.layouts), seed * 100003 + i)
        out.append(tr)
    return out


def save(trials, path):
    with open(path, "wb") as f:
        pickle.dump(trials, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def age_feats(trial: Trial, cur_step: int, ev: EventRec) -> np.ndarray:
    st = trial.steps[cur_step]
    return np.array([float(ev.session == st.session), min(st.session - ev.session, 8) / 8.0,
                     min(cur_step - ev.step, 120) / 120.0], dtype=np.float32)


def window_events(trial: Trial, i: int, n_prev: int = W_FRAMES) -> list[int]:
    """Indices of events whose frames are inside the working-memory window of step i."""
    st = trial.steps[i]
    lo = i - n_prev
    return [j for j in range(st.n_avail) if trial.events[j].step >= lo and trial.events[j].step <= i
            and trial.events[j].session == st.session]


def label_for(trial: Trial, i: int, mem_idx: list[int]) -> int:
    """Expert action from the information in `mem_idx` + working window, this layout only."""
    st = trial.steps[i]
    idx = sorted(set(mem_idx) | set(window_events(trial, i)))
    hist = [(j, trial.events[j].ev) for j in idx if trial.events[j].layout == st.layout]
    L = trial.layouts[st.layout]
    task = trial.sessions[st.session][1]
    a, _ = expert_from_state(L, task, st.state, hist)
    return int(a)


def build_mem(trial: Trial, i: int, mem_idx: list[int], modality: list[int], B: int):
    """Tensors for B slots (zero-padded)."""
    present = np.zeros(B, dtype=bool)
    mod = np.zeros(B, dtype=np.int64)
    sym = np.zeros((B, EVENT_DIM), dtype=np.float32)
    frames = np.zeros((B, 3, IMG, IMG), dtype=np.float32)
    age = np.zeros((B, AGE_DIM_), dtype=np.float32)
    place = np.zeros((B, PLACE_DIM), dtype=np.float32)
    for s, (j, m) in enumerate(zip(mem_idx[:B], modality[:B])):
        ev = trial.events[j]
        present[s] = True
        mod[s] = m
        sym[s] = ev.vec()
        if m == 1:
            frames[s] = frame_for_event(trial, j).transpose(2, 0, 1) / 255.0
        age[s] = age_feats(trial, i, ev)
        place[s] = trial.place[ev.layout]
    return dict(present=torch.from_numpy(present), modality=torch.from_numpy(mod), sym=torch.from_numpy(sym),
                frames=torch.from_numpy(frames), age=torch.from_numpy(np.concatenate([age, place], 1)))


AGE_DIM_ = 3


def ctx_for(trial: Trial, i: int) -> np.ndarray:
    st = trial.steps[i]
    return ctx_vec(st.proprio, st.instr, trial.place[st.layout], trial.layouts[st.layout])


def ctx_vec(proprio, instr, place, L: Layout) -> np.ndarray:
    return np.concatenate([proprio, instr, place, layout_vec(L)]).astype(np.float32)


class BCDataset(Dataset):
    def __init__(self, trials: list[Trial], B: int = 8, p_frame: float = 0.4, seed: int = 0):
        self.trials = trials
        self.B = B
        self.p_frame = p_frame
        self.index = [(ti, i) for ti, tr in enumerate(trials) for i in range(len(tr.steps))]
        self.seed = seed

    def __len__(self):
        return len(self.index)

    def sample_memory(self, trial: Trial, i: int, rng) -> list[int]:
        st = trial.steps[i]
        avail = np.arange(st.n_avail)
        B = self.B
        r = rng.random()
        if r < 0.10 or len(avail) == 0:
            return []
        if r < 0.40:
            m = int(rng.integers(0, B + 1))
            return list(rng.choice(avail, size=min(m, len(avail)), replace=False))
        if r < 0.60:
            # the most common closed-loop configuration: everything recent, FIFO at the budget
            return [int(j) for j in avail[-B:]]
        if r < 0.90:
            rel = list(st.rel)
            if len(rel) > B:
                rel = list(rng.choice(rel, size=B, replace=False))
            extra = [j for j in avail if j not in set(rel)]
            m = int(rng.integers(0, B - len(rel) + 1))
            if extra and m > 0:
                rel += list(rng.choice(extra, size=min(m, len(extra)), replace=False))
            return rel
        prev = [j for j in avail if trial.events[j].session < st.session]
        m = int(rng.integers(0, B + 1))
        return list(rng.choice(prev, size=min(m, len(prev)), replace=False)) if prev else []

    def __getitem__(self, k):
        ti, i = self.index[k]
        tr = self.trials[ti]
        rng = np.random.default_rng((self.seed * 1000003 + k * 7919) % (2 ** 32))
        mem_idx = [int(j) for j in self.sample_memory(tr, i, rng)]
        rng.shuffle(mem_idx)
        modality = [int(rng.random() < self.p_frame) for _ in mem_idx]
        frames = np.stack(frames_for_step(tr, i)).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        y = label_for(tr, i, mem_idx)
        return dict(frames=torch.from_numpy(frames), ctx=torch.from_numpy(ctx_for(tr, i)),
                    mapt=torch.from_numpy(layout_tokens(tr.layouts[tr.steps[i].layout])),
                    mem=build_mem(tr, i, mem_idx, modality, self.B), y=torch.tensor(y),
                    y_full=torch.tensor(tr.steps[i].action))


def collate(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in ("frames", "ctx", "mapt", "y", "y_full")}
    out["mem"] = {k: torch.stack([b["mem"][k] for b in batch]) for k in batch[0]["mem"]}
    return out
