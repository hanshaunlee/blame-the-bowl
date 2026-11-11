"""Closed-loop evaluation sweep: every memory system x budget on a fixed set of reference trials.

    python -m hm.eval.sweep --n 400 --budgets 2 4 8 --learned ckpt/writer_both_B4.pt ...
"""
from __future__ import annotations
import argparse, json, time, os, numpy as np, torch
from pathlib import Path
from ..train.data import load, generate
from ..train.bc import make_policy, load_policy, device
from .rollout import run_batch, summarize
from .memory import *
from .learned_memory import LearnedMemory


def eval_trials(seed: int, n: int):
    """A dedicated evaluation set: 40% single tasks, 60% cross trials with k in {0,1,3,7}."""
    from ..env.session import make_trial
    from ..train.data import place_codes
    rng = np.random.default_rng(seed)
    specs = []
    for t in ("find", "count", "sequence", "adapt", "tidy"):
        specs += [{"kind": "single", "task": t}] * int(n * 0.08)
    per_k = (n - len(specs)) // 4
    for k in (0, 1, 3, 7):
        specs += [{"kind": "cross", "k": k}] * per_k
    out = []
    for i, sp in enumerate(specs):
        tr = make_trial(rng, sp)
        tr.place = place_codes(len(tr.layouts), seed * 7 + i)
        out.append(tr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="ckpt/policy.pt")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--learned", nargs="*", default=[], help="writer checkpoints (name inferred from file)")
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--out", default="results/sweep.json")
    ap.add_argument("--chunk", type=int, default=200)
    a = ap.parse_args()
    dev = device()
    pol, ck = load_policy(a.policy, dev)
    trials = eval_trials(a.seed, a.n)
    print(f"{len(trials)} eval trials on {dev}", flush=True)
    results = {}
    if os.path.exists(a.out):
        results = json.load(open(a.out))

    def systems_for(B):
        rng = np.random.default_rng(0)
        base = {
            "none": lambda r: NoMemory(B, rng),
            "recency_sym": lambda r: RecencySym(B, rng),
            "recency_frame": lambda r: RecencyFrame(B, rng),
            "random": lambda r: Reservoir(B, rng),
            "scripted_mem": lambda r: ScriptedMEM(B, rng, keep_fail=True),
            "scripted_mem_dropfail": lambda r: ScriptedMEM(B, rng, keep_fail=False),
            "oracle": lambda r: OracleMemory(B, rng, r.sessions[-1][1] if r.kind == "cross" else None),
        }
        for path in a.learned:
            ckw = torch.load(path, map_location="cpu")
            if ckw["B"] != B:
                continue
            name = f"learned_{ckw['mode']}" + ("_sym" if ckw.get("sym_only") else "")
            base[name] = (lambda p: (lambda r: LearnedMemory(B, p, dev=dev)))(path)
        if a.systems:
            base = {k: v for k, v in base.items() if k in a.systems}
        return base

    for B in a.budgets:
        for name, fac in systems_for(B).items():
            key = f"{name}/B{B}"
            if key in results:
                print("skip", key); continue
            t0 = time.time()
            lives = []
            for i in range(0, len(trials), a.chunk):
                lives += run_batch(pol, trials[i:i + a.chunk], fac, dev)
            summ = summarize(lives)
            # memory composition: fraction of frame units in the final slots of cross trials
            frac_frame = float(np.mean([np.mean([m for _, m in lt.memsys.contents()]) if lt.memsys.contents() else np.nan for lt in lives if lt.ref.kind == "cross"]))
            # what is kept: event-type histogram of the slots entering the query session (cross trials)
            from ..env.world import EVENT_NAMES
            comp = {}
            for lt in lives:
                if lt.ref.kind != "cross":
                    continue
                for ei, m in getattr(lt, "slots_at_query", []):
                    k2 = EVENT_NAMES[lt.events[ei].ev.type] + ("/frame" if m == 1 else "")
                    comp[k2] = comp.get(k2, 0) + 1
            results[key] = dict(summary=summ, frac_frame=frac_frame, composition=comp, secs=time.time() - t0)
            q = {k: v[0] for k, v in summ.items() if k.startswith("query/")}
            s = {k.split("/")[1]: round(v[0], 2) for k, v in summ.items() if k.startswith("single/")}
            print(f"{key:28s} query-by-k {q}  single {s}  ({time.time()-t0:.0f}s)", flush=True)
            Path(os.path.dirname(a.out)).mkdir(parents=True, exist_ok=True)
            json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
