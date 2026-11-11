"""Closed-loop wrappers around the learned OnlineWriter / Consolidator."""
from __future__ import annotations
import numpy as np, torch, torch.nn.functional as F
from ..env.world import render_from
from ..models.writer import OnlineWriter, Consolidator, plackett_luce_sample
from ..train.writer_rl import FEAT, ONLINE_CTX, SLEEP_CTX
from .memory import MemorySystem


def _base(trial, ei, m):
    """Base embedding of event ei in modality m, cached on the LiveTrial."""
    pol, dev = trial.pol, trial.dev
    d = trial.ev_base.setdefault(ei, {})
    if m not in d:
        ev = trial.events[ei]
        if m == 0:
            d[0] = pol.unit_base_sym(torch.from_numpy(ev.vec()).to(dev).unsqueeze(0))[0]
        else:
            st = trial.steps[ev.step].state if ev.step < len(trial.steps) else trial.w.render_state()
            li, task = trial.sessions[ev.session]
            img = torch.from_numpy(render_from(trial.layouts[li], task, st)).to(dev).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            d[1] = pol.unit_base_frame(img)[0]
    return d[m]


def _feats(trial, ei, cur_session, cur_step):
    ev = trial.events[ei]
    return [float(ev.session == cur_session), min(cur_session - ev.session, 8) / 8.0,
            min(cur_step - ev.step, 120) / 120.0] + trial.place[ev.layout].tolist()


class LearnedMemory(MemorySystem):
    learned = True

    def __init__(self, B, ckpt_path, mode=None, dev=None, greedy=True):
        super().__init__(B)
        ck = torch.load(ckpt_path, map_location="cpu")
        self.mode = mode or ck["mode"]
        self.name = f"learned_{self.mode}"
        self.Bo = B // 2 if self.mode == "split" else B
        self.Bc = B - self.Bo if self.mode == "split" else B
        self.lt_slots = []
        self.sym_only = bool(ck.get("sym_only", False))
        if self.sym_only:
            self.name += "_sym"
        self.dev = dev or torch.device("cpu")
        self.greedy = greedy
        self.d = ck.get('d', 128)
        self.writer = OnlineWriter(self.d, ONLINE_CTX, FEAT).to(self.dev); self.writer.load_state_dict(ck["writer"]); self.writer.eval()
        self.cons = Consolidator(self.d, SLEEP_CTX, FEAT + 1).to(self.dev); self.cons.load_state_dict(ck["cons"]); self.cons.eval()

    def reset_trial(self):
        self.slots = []; self.lt_slots = []

    def contents(self):
        return list(self.slots) + list(self.lt_slots)

    @torch.no_grad()
    def on_event(self, trial, ei):
        if self.mode not in ("online", "both", "split"):
            return
        pol, dev, B = trial.pol, trial.dev, self.Bo
        cs, cst = trial.cur, len(trial.steps)
        li, task = trial.sessions[cs]
        ctx = torch.from_numpy(np.concatenate([task.instr_vec(), trial.place[li]])).to(dev).unsqueeze(0).float()
        cand = torch.stack([_base(trial, ei, 0), _base(trial, ei, 1)]).unsqueeze(0)
        cf = torch.tensor(_feats(trial, ei, cs, cst), device=dev).unsqueeze(0)
        slots = torch.zeros(1, B, pol.d, device=dev); sf = torch.zeros(1, B, FEAT, device=dev); pres = torch.zeros(1, B, dtype=torch.bool, device=dev)
        for k, (ej, m) in enumerate(self.slots):
            slots[0, k] = _base(trial, ej, m); sf[0, k] = torch.tensor(_feats(trial, ej, cs, cst), device=dev); pres[0, k] = True
        logits = self.writer(ctx, cand, cf, pol.unit_finish(slots, sf), sf, pres)
        n = len(self.slots)
        if n < B:
            logits[0, 2 * (n + 1): 2 * B] = float("-inf")
        if self.sym_only:
            logits[0, 1:2 * B:2] = float("-inf")
        a = int(logits.argmax(-1)) if self.greedy else int(torch.distributions.Categorical(logits=logits).sample())
        if a < 2 * B:
            k, m = a // 2, a % 2
            if k < n:
                self.slots[k] = (ei, m)
            else:
                self.slots.append((ei, m))

    @torch.no_grad()
    def on_session_end(self, trial, s):
        if self.mode not in ("sleep", "both", "split"):
            return
        if s >= len(trial.ref.sessions) - 1:
            return
        pol, dev, B = trial.pol, trial.dev, self.Bc
        ev_in = [j for j, e in enumerate(trial.events) if e.session == s]
        old = (self.lt_slots + self.slots) if self.mode == "split" else self.slots
        mods = (0,) if self.sym_only else (0, 1)
        items = [(ei, m, 1) for (ei, m) in old] + [(j, m, 0) for j in ev_in for m in mods]
        if not items:
            return
        cst = len(trial.steps)
        li = trial.sessions[s][0]
        ctx = torch.from_numpy(trial.place[li]).to(dev).unsqueeze(0).float()
        it = torch.stack([_base(trial, ei, m) for ei, m, _ in items]).unsqueeze(0)
        itf = torch.tensor([_feats(trial, ei, s, cst) + [float(old)] for ei, m, old in items], device=dev).unsqueeze(0)
        mask = torch.ones(1, len(items), dtype=torch.bool, device=dev)
        scores = self.cons(ctx, pol.unit_finish(it, itf[:, :, :FEAT]), itf, mask)
        idx, _ = plackett_luce_sample(scores, B, greedy=self.greedy)
        chosen = [items[k][:2] for k in idx[0].tolist() if k >= 0]
        if self.mode == "split":
            self.lt_slots = chosen; self.slots = []
        else:
            self.slots = chosen
