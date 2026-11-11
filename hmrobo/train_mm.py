"""GRPO training of the memory manager against the frozen VLA, in-process (policy-learning venv).

Reward for a candidate 32-frame memory at execution step t of a demonstration: minus the frozen
model's flow-matching loss on the demonstration's next 20 actions given that memory (common random
numbers across candidates). The buffer is [lesson session][k distractor sessions][this episode up
to t] with true positions, exactly what the policy sees at evaluation.

    .venv/bin/python /app/ours/hmrobo/train_mm.py --ckpt <dir> --task VideoRepick --updates 100 --out /vol/mm_VideoRepick.pt
    .venv/bin/python /app/ours/hmrobo/train_mm.py --ckpt <dir> --task VideoRepick --probe 24 --out_probe /vol/probe.json
"""
from __future__ import annotations
import argparse, json, os, sys, time, random, zlib
from collections import OrderedDict
from pathlib import Path
import numpy as np, torch, h5py

sys.path.insert(0, "/app/ours")
from mme_vla_suite.policies import policy_config as _policy_config  # noqa: E402
from mme_vla_suite.training import config as _config  # noqa: E402
from hmrobo.buffer import load_episode, load_episode_npz, episode_keys, load_units, UNIT  # noqa: E402
import glob  # noqa: E402
from hmrobo.selector import FrameSelector, MEM, even_over, pos_features  # noqa: E402
from hmrobo.mm_policy import MemoryManager, encode_frames, set_buffer  # noqa: E402
from mme_vla_suite.shared.data_utils import even_sampling_indices  # noqa: E402

AH = 20  # model action horizon


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


class Data:
    def __init__(self, pol, a):
        self.pol, self.a = pol, a
        self.path = f"{a.data_dir}/record_dataset_{a.task}.h5"
        if a.demo_dir:   # held-out oracle demonstrations (npz) instead of the memorised train split
            self.files = {os.path.basename(p)[:-4]: p for p in sorted(glob.glob(f"{a.demo_dir}/{a.task}/ep*.npz"))}
            self.keys = list(self.files)[: a.n_train_eps]
        else:
            self.files = None
            with h5py.File(self.path, "r") as f:
                self.keys = episode_keys(f)[: a.n_train_eps]
        raw = load_units(a.data_dir, a.task, pool=a.distractor_pool)
        t0 = time.time()
        self.units = [dict(task=u["task"], key=u["key"], frames=encode_frames(pol, u["front"], u["state"])) for u in raw]
        print(f"{len(self.keys)} train episodes; {len(self.units)} distractor units encoded in {time.time()-t0:.0f}s", flush=True)
        self.cache = OrderedDict()
        self.sens = {}
        self.mm = None

    def episode(self, key):
        if key not in self.cache:
            if len(self.cache) >= self.a.cache_eps:
                self.cache.popitem(last=False)
            ep = load_episode_npz(self.files[key], self.a.task) if self.files else load_episode(self.path, key, self.a.task)
            t0 = time.time()
            frames = encode_frames(self.pol, ep.front, ep.state)
            if key not in self.sens and self.mm is not None and self.a.p_sens > 0:
                self.sens[key] = sensitivity(self.mm, self.pol, ep, frames, self.a.t_max)
            sv = [v for _, v in self.sens.get(key, [])]
            self.cache[key] = (ep, frames)
            print(f"  encoded {key}: T={len(ep.front)} demo={int(ep.is_demo.sum())} in {time.time()-t0:.0f}s"
                  + (f" sens mean {np.mean(sv):.4f} max {np.max(sv):.4f}" if sv else ""), flush=True)
        self.cache.move_to_end(key)
        return self.cache[key]

    def sample(self, rng, k, key=None):
        """-> dict(ep, frames(list for whole buffer up to the last exec frame), H, lesson_bounds, session_bounds, e0)."""
        key = key or rng.choice(self.keys)
        ep, frames = self.episode(key)
        e0 = ep.exec_start
        pool = [u for u in self.units if u["key"] != key or u["task"] != self.a.task]
        ds = rng.sample(pool, k) if k <= len(pool) else [rng.choice(pool) for _ in range(k)]
        sessions = [frames[:e0]] + [u["frames"] for u in ds]
        order = list(range(len(sessions)))
        if self.a.shuffle_sessions:
            rng.shuffle(order)
        buf, bounds, lesson = [], [], None
        for o in order:
            a_ = len(buf); buf += sessions[o]; bounds.append((a_, len(buf)))
            if o == 0:
                lesson = (a_, len(buf))
        H = len(buf)
        buf = buf + frames[e0:]
        return dict(ep=ep, key=key, frames=buf, H=H, lesson_bounds=lesson, session_bounds=bounds, e0=e0)


def sensitivity(mm, pol, ep, frames, t_max, stride=16, n_mc=2):
    """Memory sensitivity s(t) = loss(this episode only) - loss(lesson + this episode), k=0 layout.
    Decision-relevant steps (where the lesson changes the action) are sparse; we sample them more."""
    e0 = ep.exec_start
    set_buffer(pol, frames, e0)
    out = []
    for t in range(e0 + 4, min(len(ep.front) - AH, e0 + t_max), stride):
        T = t
        pol.step_idx = T
        sd = seed_of(ep.key, "sens", t)
        full = mm.flow_loss(obs_at(ep, t), ep.joint_action[t:t + AH], list(even_sampling_indices(T, MEM)), sd, n_mc)
        ex = mm.flow_loss(obs_at(ep, t), ep.joint_action[t:t + AH], even_over(list(range(e0, T + 1))), sd, n_mc)
        out.append((t, ex - full))
    return out


def obs_at(ep, t):
    return {"observation/image": ep.front[t], "observation/wrist_image": ep.wrist[t], "observation/state": ep.state[t], "prompt": ep.goal}


def pick_ts(rng, ep, S, t_max, sens=None, p_sens=0.7):
    e0 = ep.exec_start
    hi = min(len(ep.front) - AH, e0 + t_max)
    cand = list(range(e0, hi))
    if not cand:
        return []
    out = set()
    while len(out) < min(S, len(cand)):
        if sens and rng.random() < p_sens:
            w = np.array([max(v, 0.0) + 1e-4 for _, v in sens]); w = w / w.sum()
            t0 = sens[int(np.searchsorted(np.cumsum(w), rng.random()))][0]
            t = min(max(e0, t0 + rng.randint(-8, 8)), hi - 1)
        else:
            t = rng.choice(cand)
        out.add(t)
    return sorted(out)


def selector_inputs(mm, F, T, H, G=1):
    n = T + 1
    emb = torch.from_numpy(np.ascontiguousarray(F[:n])).float().unsqueeze(0).expand(G, -1, -1)
    pos = torch.from_numpy(pos_features(n, H)).unsqueeze(0).expand(G, -1, -1)
    return emb, pos, emb[:, T], torch.ones(G, n, dtype=torch.bool)


def sft_uniform(sel, mm, pol, data, rng, a):
    """Warm start: make the selector's scores reproduce the released policy's even sampling (no privileged
    information), so RL starts at the baseline's reward instead of at random subsets that miss recent frames."""
    opt = torch.optim.AdamW(sel.parameters(), lr=1e-3, weight_decay=0.01)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0))
    sel.train(); t0 = time.time(); ov = []
    for step in range(a.sft_steps):
        sm = data.sample(rng, rng.choice(a.k_choices))
        ep, H = sm["ep"], sm["H"]
        set_buffer(pol, sm["frames"], H)
        ts = pick_ts(rng, ep, 1, a.t_max, None, 0.0)
        if not ts:
            continue
        T = H + (ts[0] - sm["e0"])
        F = mm.features(range(T + 1))
        emb, pos, cur, mask = selector_inputs(mm, F, T, H)
        scores = sel(emb, pos, cur, mask)
        y = torch.zeros_like(scores); y[0, list(even_sampling_indices(T, MEM))] = 1.0
        loss = bce(scores, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        top = set(torch.topk(scores[0], min(MEM, T + 1)).indices.tolist()); ov.append(len(top & set(even_sampling_indices(T, MEM))) / MEM)
        if (step + 1) % 50 == 0:
            print(f"sft {step+1} loss {loss.item():.3f} top32-overlap {np.mean(ov[-50:]):.2f} {time.time()-t0:.0f}s", flush=True)


def fixed_probes(data, rng, n, a):
    out = []
    for i in range(n):
        sm = data.sample(rng, a.k_choices[i % len(a.k_choices)])
        ts = pick_ts(rng, sm["ep"], 1, a.t_max, data.sens.get(sm["key"]), 1.0)
        if ts:
            out.append((sm, ts[0]))
    return out


def eval_probes(mm, pol, probes, a, loss_for):
    res = {"learned": [], "uniform": [], "oracle": []}
    mm.sel.eval()
    with torch.no_grad():
        for sm, t in probes:
            H = sm["H"]; set_buffer(pol, sm["frames"], H); T = H + (t - sm["e0"]); sd = seed_of(sm["key"], t, "probe")
            F = mm.features(range(T + 1)); la, lb = sm["lesson_bounds"]; ex = list(range(H, T + 1))
            res["learned"].append(-loss_for(sm, lambda: mm.select("learned", T, H, F=F), t, T, sd))
            res["uniform"].append(-loss_for(sm, lambda: list(even_sampling_indices(T, MEM)), t, T, sd))
            res["oracle"].append(-loss_for(sm, lambda: even_over(list(range(la, lb)) + ex), t, T, sd))
    return {k: float(np.mean(v)) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="mme_vla_suite")
    ap.add_argument("--task", default="VideoRepick")
    ap.add_argument("--data_dir", default="/vol/data")
    ap.add_argument("--demo_dir", default="", help="npz demonstrations from gen_demos.py (held-out); empty = train h5")
    ap.add_argument("--updates", type=int, default=100)
    ap.add_argument("--eps_per_update", type=int, default=6)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--S", type=int, default=2)
    ap.add_argument("--n_mc", type=int, default=4)
    ap.add_argument("--t_max", type=int, default=240)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--k_choices", type=int, nargs="+", default=[0, 1, 3, 7])
    ap.add_argument("--distractor_pool", default="other", choices=["other", "same", "all"])
    ap.add_argument("--shuffle_sessions", action="store_true")
    ap.add_argument("--n_train_eps", type=int, default=60)
    ap.add_argument("--cache_eps", type=int, default=6)
    ap.add_argument("--init", default="")
    ap.add_argument("--out", default="/vol/mm.pt")
    ap.add_argument("--probe", type=int, default=0, help="probe N samples with baseline memories, then exit")
    ap.add_argument("--out_probe", default="/vol/probe.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sft_steps", type=int, default=300, help="supervised warm start to even sampling before RL")
    ap.add_argument("--n_probes", type=int, default=16, help="fixed (episode, step) probes for a stable progress metric")
    ap.add_argument("--probe_every", type=int, default=10)
    ap.add_argument("--p_sens", type=float, default=0.7, help="prob. of sampling a timestep from the memory-sensitivity profile")
    ap.add_argument("--min_spread", type=float, default=5e-4, help="skip GRPO groups whose reward range is below this (no signal)")
    a = ap.parse_args()

    pol = _policy_config.create_trained_policy(_config.get_config(a.config), Path(a.ckpt), seed=7)
    print("history config:", pol.config, flush=True)
    mm = MemoryManager(pol)
    data = Data(pol, a)
    data.mm = mm
    rng = random.Random(a.seed)

    def loss_for(sm, idx_fn, t, T, seed):
        """Loss at demo step t (buffer position T) for memory indices idx_fn()."""
        pol.step_idx = T
        return mm.flow_loss(obs_at(sm["ep"], t), sm["ep"].joint_action[t:t + AH], idx_fn(), seed, a.n_mc)

    if a.probe:
        rows = []
        for i in range(a.probe):
            k = a.k_choices[i % len(a.k_choices)]
            sm = data.sample(rng, k)
            ep, H = sm["ep"], sm["H"]
            set_buffer(pol, sm["frames"], H)
            for t in pick_ts(rng, ep, 1, a.t_max):
                T = H + (t - sm["e0"])
                F = mm.features(range(T + 1))
                sd = seed_of(sm["key"], t)
                la, lb = sm["lesson_bounds"]
                ex = list(range(H, T + 1))
                dis = [j for j in range(H) if not (la <= j < lb)]
                r = dict(k=k, key=sm["key"], t=t, T=T, D=lb - la)
                t0 = time.time()
                r["uniform"] = loss_for(sm, lambda: list(even_sampling_indices(T, MEM)), t, T, sd)
                r["uniform_rep"] = loss_for(sm, lambda: list(even_sampling_indices(T, MEM)), t, T, sd)
                r["uniform_seed2"] = loss_for(sm, lambda: list(even_sampling_indices(T, MEM)), t, T, sd + 1)
                r["lesson_oracle"] = loss_for(sm, lambda: even_over(list(range(la, lb)) + ex), t, T, sd)
                r["exec_only"] = loss_for(sm, lambda: even_over(ex), t, T, sd)
                r["lesson_only"] = loss_for(sm, lambda: even_over(list(range(la, lb))), t, T, sd)
                r["distractor_only"] = loss_for(sm, lambda: even_over(dis + ex) if dis else even_over(ex), t, T, sd)
                r["recency"] = loss_for(sm, lambda: list(range(max(0, T + 1 - MEM), T + 1)), t, T, sd)
                r["random"] = loss_for(sm, lambda: sorted(rng.sample(range(T + 1), min(MEM, T + 1))), t, T, sd)
                r["retrieval"] = loss_for(sm, lambda: mm.select("retrieval", T, H, sm["lesson_bounds"], sm["session_bounds"], F=F), t, T, sd)
                r["secs"] = time.time() - t0
                rows.append(r)
                print(" ".join(f"{kk}={v:.4f}" if isinstance(v, float) else f"{kk}={v}" for kk, v in r.items()), flush=True)
                json.dump(rows, open(a.out_probe, "w"), indent=1)
        keys = [kk for kk in rows[0] if isinstance(rows[0][kk], float) and kk != "secs"]
        print("PROBE mean loss by k:")
        for k in sorted({r["k"] for r in rows}):
            rs = [r for r in rows if r["k"] == k]
            print(f"  k={k} n={len(rs)} " + " ".join(f"{kk}={np.mean([r[kk] for r in rs]):.4f}" for kk in keys))
        return

    sel = FrameSelector()
    resumed = False
    if a.init:
        try:
            ck = torch.load(a.init, map_location="cpu"); sel.load_state_dict(ck["state"]); resumed = True
            print(f"resumed from {a.init} (log has {len(ck.get('log', []))} updates)", flush=True)
        except Exception as e:
            print(f"could not load {a.init}: {e}; starting fresh", flush=True)
    if not resumed and a.sft_steps > 0:
        sft_uniform(sel, mm, pol, data, rng, a)
    mm.sel = sel
    probes = fixed_probes(data, random.Random(12345), a.n_probes, a)
    pr = eval_probes(mm, pol, probes, a, loss_for)
    print(f"PROBE upd -1 learned {pr['learned']:.4f} uniform {pr['uniform']:.4f} oracle {pr['oracle']:.4f}", flush=True)
    opt = torch.optim.AdamW(sel.parameters(), lr=a.lr, weight_decay=0.01)
    log = [dict(update=-1, probe=pr)]; t0 = time.time()
    for u in range(a.updates):
        logps, advs, Rs, lf, ef, base = [], [], [], [], [], {"uniform": [], "lesson_oracle": []}
        n_groups, n_skipped = 0, 0
        for e in range(a.eps_per_update):
            k = rng.choice(a.k_choices)
            sm = data.sample(rng, k)
            ep, H = sm["ep"], sm["H"]
            set_buffer(pol, sm["frames"], H)
            la, lb = sm["lesson_bounds"]
            for t in pick_ts(rng, ep, a.S, a.t_max, data.sens.get(sm["key"]), a.p_sens):
                T = H + (t - sm["e0"])
                F = mm.features(range(T + 1))
                sd = seed_of(sm["key"], t)
                sel.train()
                idxs, logp = mm.learned(F, T, H, greedy=False, G=a.G)
                R = torch.tensor([-loss_for(sm, lambda g=g: [int(i) for i in idxs[g].tolist()], t, T, sd) for g in range(a.G)])
                n_groups += 1
                if (R.max() - R.min()).item() < a.min_spread:
                    n_skipped += 1
                    continue
                adv = (R - R.mean()) / (R.std() + 1e-4)
                logps.append(logp); advs.append(adv); Rs.append(R.mean().item())
                for g in range(a.G):
                    ii = idxs[g].numpy()
                    lf.append(float(np.mean((ii >= la) & (ii < lb)))); ef.append(float(np.mean(ii >= H)))
                if e == 0:
                    ex = list(range(H, T + 1))
                    base["uniform"].append(-loss_for(sm, lambda: list(even_sampling_indices(T, MEM)), t, T, sd))
                    base["lesson_oracle"].append(-loss_for(sm, lambda: even_over(list(range(la, lb)) + ex), t, T, sd))
        if not logps:
            continue
        loss = -(torch.cat(advs).detach() * torch.cat(logps)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(sel.parameters(), 1.0); opt.step()
        rec = dict(update=u, R=float(np.mean(Rs)), R_uniform=float(np.mean(base["uniform"])) if base["uniform"] else float("nan"),
                   R_oracle=float(np.mean(base["lesson_oracle"])) if base["lesson_oracle"] else float("nan"),
                   lesson_frac=float(np.mean(lf)), exec_frac=float(np.mean(ef)), loss=loss.item(), skipped=n_skipped / max(n_groups, 1), secs=time.time() - t0)
        log.append(rec)
        print(f"upd {u} R {rec['R']:.4f} (uniform {rec['R_uniform']:.4f} oracle {rec['R_oracle']:.4f}) lesson_frac {rec['lesson_frac']:.2f} exec_frac {rec['exec_frac']:.2f} skipped {rec['skipped']:.2f} loss {rec['loss']:.3f} {rec['secs']:.0f}s", flush=True)
        if (u + 1) % a.probe_every == 0 or u == a.updates - 1:
            pr = eval_probes(mm, pol, probes, a, loss_for); rec["probe"] = pr
            print(f"PROBE upd {u} learned {pr['learned']:.4f} uniform {pr['uniform']:.4f} oracle {pr['oracle']:.4f}", flush=True)
        if (u + 1) % 5 == 0 or u == a.updates - 1:
            torch.save({"state": sel.state_dict(), "cfg": sel.cfg, "log": log, "task": a.task, "args": vars(a)}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
