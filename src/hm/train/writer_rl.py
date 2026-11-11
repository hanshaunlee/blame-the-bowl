"""GRPO training of memory-management policies against the frozen student.

Decision-centric reward: a memory is good exactly to the extent that the student, given that
memory, assigns probability to the full-information expert's action. The writer never sees
the future, and the consolidator never sees the next task.

Variants (--mode):
  online   : OnlineWriter decides at every event (MEM-style, writes as events arrive)
  sleep    : Consolidator re-selects the B slots at every session end (hindsight)
  both     : online during the session + consolidation at session end
"""
from __future__ import annotations
import argparse, time, json, os, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from ..env.world import *
from ..env.session import frames_for_step, frame_for_event
from ..models.writer import OnlineWriter, Consolidator, plackett_luce_sample
from .data import load, save, generate, PLACE_DIM, INSTR_DIM, ctx_for, layout_tokens
from .bc import make_policy, load_policy, device

FEAT = 3 + PLACE_DIM          # same-session, session delta, step delta, place code
ONLINE_CTX = INSTR_DIM + PLACE_DIM
SLEEP_CTX = PLACE_DIM


class Cache:
    """Per-trial precomputed tensors (frozen student features)."""

    def __init__(self, pol, dev):
        self.pol, self.dev, self.store = pol, dev, {}

    @torch.no_grad()
    def get(self, ti, tr):
        if ti in self.store:
            return self.store[ti]
        T, E = len(tr.steps), len(tr.events)
        frames = np.stack([np.stack(frames_for_step(tr, i)) for i in range(T)])  # (T,4,H,W,3)
        ft = torch.from_numpy(frames).to(self.dev).permute(0, 1, 4, 2, 3).float() / 255.0
        ftok = torch.cat([self.pol.frame_tokens(ft[i:i + 256]) for i in range(0, T, 256)])
        ctx = np.stack([ctx_for(tr, i) for i in range(T)])
        ctok = self.pol.ctx_tokens(torch.from_numpy(ctx).to(self.dev))
        mtok = self.pol.map_tokens(torch.from_numpy(np.stack([layout_tokens(tr.layouts[s.layout]) for s in tr.steps])).to(self.dev))
        sym = torch.from_numpy(np.stack([e.vec() for e in tr.events])).to(self.dev)
        esym = self.pol.unit_base_sym(sym)
        ef = torch.from_numpy(np.stack([frame_for_event(tr, j) for j in range(E)])).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
        efrm = self.pol.unit_base_frame(ef)
        base = torch.stack([esym, efrm], 1)  # (E,2,d)
        d = dict(ftok=ftok.cpu(), ctok=ctok.cpu(), mtok=mtok.cpu(), base=base.cpu(),
                 a=torch.tensor([s.action for s in tr.steps]),
                 relmask=torch.tensor([len(s.rel) > 0 for s in tr.steps]),
                 n_avail=[s.n_avail for s in tr.steps],
                 step_session=[s.session for s in tr.steps],
                 ev_session=[e.session for e in tr.events], ev_step=[e.step for e in tr.events],
                 ev_layout=[e.layout for e in tr.events],
                 instr=torch.from_numpy(np.stack([tr.sessions[e.session][1].instr_vec() for e in tr.events])),
                 place=torch.from_numpy(tr.place),
                 sess_first_step=[b[0] for b in tr.session_bounds], n_sessions=len(tr.sessions))
        self.store[ti] = d
        return d


def feats(c, ev_idx, cur_session, cur_step):
    """age/place features for an event at a given time."""
    es, est, el = c["ev_session"][ev_idx], c["ev_step"][ev_idx], c["ev_layout"][ev_idx]
    return [float(es == cur_session), min(cur_session - es, 8) / 8.0, min(cur_step - est, 120) / 120.0] + c["place"][el].tolist()


class Sim:
    """One sampled memory trajectory for one trial."""

    def __init__(self, ti, c, B):
        self.ti, self.c, self.B = ti, c, B
        self.slots = []            # list of (ei, m)  (online region in split mode)
        self.lt_slots = []         # consolidated region (split mode only)
        self.snap_after_event = []  # slots snapshot after processing event j (for every j)
        self.logps = []            # tensors (scalar) of decisions
        self.sess_snap = {}        # session s -> slots at start of session s


def slots_at_step(sim, t):
    """Memory contents visible at step t: the snapshot after the last event observable at t."""
    n = sim.c["n_avail"][t]
    return sim.snap_after_event[n - 1] if n > 0 else []


SYM_ONLY = False  # set by main(); restricts learned writers to symbolic units


def simulate(mode, writer, cons, sims, dev, pol, greedy=False):
    """Interleave per session: (online decisions over the session's events) then consolidation.
    split: B//2 slots are written online and cleared at session end (short-term), the other B - B//2
    are re-selected by the consolidator at session end (long-term), MEM-style two-scale memory."""
    B = sims[0].B
    Bo = B // 2 if mode == "split" else B
    Bc = B - Bo if mode == "split" else B
    maxS = max(s.c["n_sessions"] for s in sims)
    for sidx in range(maxS):
        act = [s for s in sims if sidx < s.c["n_sessions"]]
        ev_lists = [[j for j, es in enumerate(s.c["ev_session"]) if es == sidx] for s in act]
        if mode in ("online", "both", "split"):
            maxE = max(len(e) for e in ev_lists)
            for jj in range(maxE):
                cur = [(s, ev[jj]) for s, ev in zip(act, ev_lists) if jj < len(ev)]
                _online_step(writer, cur, dev, pol, greedy, Bo)
        else:
            for s, ev in zip(act, ev_lists):
                for _ in ev:
                    s.snap_after_event.append(list(s.slots))
        nxt = [s for s in act if sidx < s.c["n_sessions"] - 1]
        if nxt and mode in ("sleep", "both", "split"):
            _consolidate(cons, nxt, sidx, dev, pol, greedy, Bc, split=(mode == "split"))


def _online_step(writer, cur, dev, pol, greedy, B):
    N = len(cur)
    ctx = torch.zeros(N, ONLINE_CTX); cand = torch.zeros(N, 2, pol.d); cf = torch.zeros(N, FEAT)
    slots = torch.zeros(N, B, pol.d); sf = torch.zeros(N, B, FEAT); pres = torch.zeros(N, B, dtype=torch.bool)
    for i, (s, j) in enumerate(cur):
        c = s.c; cs, cst = c["ev_session"][j], c["ev_step"][j]
        ctx[i] = torch.cat([c["instr"][j], c["place"][c["ev_layout"][j]]])
        cand[i] = c["base"][j]; cf[i] = torch.tensor(feats(c, j, cs, cst))
        for k, (ei, m) in enumerate(s.slots):
            slots[i, k] = c["base"][ei, m]; sf[i, k] = torch.tensor(feats(c, ei, cs, cst)); pres[i, k] = True
    slots_fin = pol.unit_finish(slots.to(dev), sf.to(dev))
    logits = writer(ctx.to(dev), cand.to(dev), cf.to(dev), slots_fin, sf.to(dev), pres.to(dev))
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for i, (s, j) in enumerate(cur):
        n = len(s.slots)
        if n < B:
            mask[i, 2 * (n + 1): 2 * B] = True
    if SYM_ONLY:
        mask[:, 1:2 * B:2] = True
    lp = F.log_softmax(logits.masked_fill(mask, float("-inf")), -1)
    a = lp.argmax(-1) if greedy else torch.distributions.Categorical(logits=lp).sample()
    for i, (s, j) in enumerate(cur):
        ai = int(a[i]); s.logps.append(lp[i, ai])
        if ai < 2 * B:
            k, m = ai // 2, ai % 2
            if k < len(s.slots):
                s.slots[k] = (j, m)
            else:
                s.slots.append((j, m))
        s.snap_after_event.append(list(s.slots) + list(s.lt_slots))


def _consolidate(cons, act, sidx, dev, pol, greedy, B, split=False):
    N = len(act)
    rows = []
    for s_ in act:
        c = s_.c
        ev_in = [j for j, es in enumerate(c["ev_session"]) if es == sidx]
        old = (s_.lt_slots + s_.slots) if split else s_.slots
        mods = (0,) if SYM_ONLY else (0, 1)
        items = [(ei, m, 1) for (ei, m) in old] + [(j, m, 0) for j in ev_in for m in mods]
        rows.append(items)
    M = max(len(r) for r in rows)
    it = torch.zeros(N, M, pol.d); itf = torch.zeros(N, M, FEAT + 1); mask = torch.zeros(N, M, dtype=torch.bool)
    ctx = torch.zeros(N, SLEEP_CTX)
    for i, (s_, items) in enumerate(zip(act, rows)):
        c = s_.c
        end_step = c["sess_first_step"][sidx + 1]
        lay = c["ev_layout"][[j for j, es in enumerate(c["ev_session"]) if es == sidx][0]]
        ctx[i] = c["place"][lay]
        for k, (ei, m, old) in enumerate(items):
            it[i, k] = c["base"][ei, m]; itf[i, k] = torch.tensor(feats(c, ei, sidx, end_step) + [float(old)]); mask[i, k] = True
    it_fin = pol.unit_finish(it.to(dev), itf[:, :, :FEAT].to(dev))
    scores = cons(ctx.to(dev), it_fin, itf.to(dev), mask.to(dev))
    idx, logp = plackett_luce_sample(scores, B, greedy=greedy)
    for i, (s_, items) in enumerate(zip(act, rows)):
        chosen = [items[k][:2] for k in idx[i].tolist() if k >= 0]
        s_.logps.append(logp[i])
        if split:
            s_.lt_slots = list(chosen); s_.slots = []   # short-term region cleared at session end
        else:
            s_.slots = list(chosen)
        s_.sess_snap[sidx + 1] = list(chosen)


@torch.no_grad()
def student_reward(pol, sims, dev, B):
    """Mean log-prob of the full-information expert action on memory-relevant steps (+0.25 all steps)."""
    rewards = []
    Bp = max(B, getattr(pol, "slots", 8))  # pad to the trained slot count
    for s in sims:
        c = s.c; T = len(c["a"])
        tok = torch.zeros(T, Bp, pol.d); feat = torch.zeros(T, Bp, FEAT); pres = torch.zeros(T, Bp, dtype=torch.bool)
        for t in range(T):
            sl = slots_at_step(s, t)
            cs = c["step_session"][t]
            for k, (ei, m) in enumerate(sl[:B]):
                tok[t, k] = c["base"][ei, m]; feat[t, k] = torch.tensor(feats(c, ei, cs, t)); pres[t, k] = True
        mt = pol.unit_finish(tok.to(dev), feat.to(dev))
        logits = pol.forward_tokens(c["ftok"].to(dev), c["ctok"].to(dev), mt, pres.to(dev), c["mtok"].to(dev))
        lp = F.log_softmax(logits, -1).gather(1, c["a"].to(dev).unsqueeze(1)).squeeze(1)
        rel = c["relmask"].to(dev)
        r_rel = lp[rel].mean() if rel.any() else lp.mean()
        rewards.append((r_rel + 0.25 * lp.mean()).item())
        s.acc = (logits.argmax(-1) == c["a"].to(dev)).float()[rel].mean().item() if rel.any() else float("nan")
    return torch.tensor(rewards)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["online", "sleep", "both", "split"])
    ap.add_argument("--policy", default="ckpt/policy.pt")
    ap.add_argument("--data", default="data/trials_rl.pkl")
    ap.add_argument("--val", default="data/trials_rl_val.pkl")
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--trials_per_update", type=int, default=16)
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sym_only", action="store_true")
    a = ap.parse_args()
    global SYM_ONLY
    SYM_ONLY = a.sym_only
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = device()
    pol, ck = load_policy(a.policy, dev)
    for p in pol.parameters():
        p.requires_grad_(False)
    if not os.path.exists(a.data):
        save(generate(a.n_train, 31), a.data); save(generate(400, 32), a.val)
    trials = load(a.data)[: a.n_train]
    val = load(a.val)
    cache, vcache = Cache(pol, dev), Cache(pol, dev)
    writer = OnlineWriter(pol.d, ONLINE_CTX, FEAT).to(dev)
    cons = Consolidator(pol.d, SLEEP_CTX, FEAT + 1).to(dev)
    params = list(writer.parameters()) + list(cons.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    out = a.out or f"ckpt/writer_{a.mode}_B{a.B}.pt"
    log = []
    t0 = time.time()
    for u in range(a.updates):
        idx = rng.choice(len(trials), a.trials_per_update, replace=False)
        sims = []
        for ti in idx:
            c = cache.get(int(ti), trials[int(ti)])
            sims += [Sim(int(ti), c, a.B) for _ in range(a.G)]
        simulate(a.mode, writer, cons, sims, dev, pol, greedy=False)
        R = student_reward(pol, sims, dev, a.B).view(a.trials_per_update, a.G).to(dev)
        adv = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-4)
        adv = adv.flatten()
        lps = torch.stack([torch.stack(s.logps).sum() if s.logps else torch.zeros((), device=dev) for s in sims])
        loss = -(adv.detach() * lps).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if u % 10 == 0:
            acc = np.nanmean([s.acc for s in sims])
            print(f"upd {u} R {R.mean().item():.3f} (std-within {R.std(1).mean().item():.3f}) rel-acc {acc:.3f} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
            log.append(dict(update=u, R=R.mean().item(), rel_acc=float(acc)))
        if (u + 1) % 50 == 0 or u == a.updates - 1:
            # greedy validation on 64 val trials
            vs = [Sim(i, vcache.get(i, val[i]), a.B) for i in range(64)]
            simulate(a.mode, writer, cons, vs, dev, pol, greedy=True)
            Rv = student_reward(pol, vs, dev, a.B)
            vacc = np.nanmean([s.acc for s in vs])
            print(f"== val (greedy) R {Rv.mean().item():.3f} rel-acc {vacc:.3f}", flush=True)
            log.append(dict(update=u, val_R=Rv.mean().item(), val_rel_acc=float(vacc)))
            torch.save({"writer": writer.state_dict(), "cons": cons.state_dict(), "mode": a.mode, "B": a.B, "log": log, "d": pol.d, "policy": a.policy, "sym_only": a.sym_only}, out)
    print("saved", out)


if __name__ == "__main__":
    main()
