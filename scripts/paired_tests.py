"""Exact McNemar tests between systems evaluated on the same episodes (common seeds).
    python scripts/paired_tests.py            # prints the comparisons quoted in the paper
Usage in code: mcnemar(rows_a, rows_b) with per-episode dicts {episode: success}."""
import json, glob, sys, collections
from scipy.stats import binomtest, fisher_exact
R = "results/robomme"
def load(f, system, k, same=None, split=None):
    d = {}
    for r in json.load(open(f"{R}/{f}")):
        if r.get("outcome") == "error" or r["system"] != system or r["k"] != k: continue
        if same is not None and bool(r.get("same_scene")) != same: continue
        if split is not None and r.get("split", "test") != split: continue
        d[r["episode"]] = int(bool(r["success"]))
    return d
def mcnemar(a, b):
    """a, b: {episode: 0/1}. Returns (n_common, mean_a, mean_b, b_only, a_only, p_exact)."""
    es = sorted(set(a) & set(b))
    x = sum(1 for e in es if a[e] == 0 and b[e] == 1); y = sum(1 for e in es if a[e] == 1 and b[e] == 0)
    p = binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
    return len(es), sum(a[e] for e in es) / len(es), sum(b[e] for e in es) / len(es), x, y, p
def fisher(a, b):
    sa, sb = sum(a.values()), sum(b.values()); return fisher_exact([[sa, len(a) - sa], [sb, len(b) - sb]])[1]
def merge(*ds):
    out = {}
    for i, d in enumerate(ds):
        for e, v in d.items(): out[(i, e)] = v
    return out
if __name__ == "__main__":
    C = []   # (name, a, b)
    # standard protocol: released k=7 vs k=0 (interference), retrieval_reindex k=7/15 vs released (fix)
    for task, f, rf in [("MoveCube", "MoveCube_v1.json", "MoveCube_v1.json"), ("PatternLock", "PatternLock_v1.json", "PatternLock_v1.json"),
                        ("RouteStick", "RouteStick_v1.json", "RouteStick_v1.json"), ("MoveCube-expert", "MoveCube_expert_v1.json", "MoveCube_expert_v1.json"),
                        ("VideoPlaceButton", "VideoPlaceButton_v3.json", "VideoPlaceButton_v3.json"), ("VideoRepick", "VideoRepick_v3.json", "VideoRepick_reindex.json")]:
        C.append((f"{task} std: released k0 vs k7", load(f, "uniform", 0), load(f, "uniform", 7)))
        C.append((f"{task} std: released k7 vs retrieval_reindex k7", load(f, "uniform", 7), load(rf, "retrieval_reindex", 7)))
        if task not in ("VideoPlaceButton", "VideoRepick"):
            C.append((f"{task} std: released k15 vs retrieval_reindex k15", load(f, "uniform", 15), load(rf, "retrieval_reindex", 15)))
            C.append((f"{task} std: retrieval k15 vs retrieval_reindex k15", load(f, "retrieval", 15), load(rf, "retrieval_reindex", 15)))
            C.append((f"{task} std: retrieval_reindex k7 vs k0", load(f, "uniform", 0), load(rf, "retrieval_reindex", 7)))
    for task in ["MoveCube", "PatternLock", "RouteStick", "VideoPlaceButton", "VideoRepick"]:
        f = {"VideoPlaceButton": "VideoPlaceButton_v3.json", "VideoRepick": "VideoRepick_v3.json"}.get(task, f"{task}_v1.json")
        C.append((f"{task} std: text_route k7 vs released k7", load(f, "uniform", 7), load(f"{task}_text_k7.json", "text_route_reindex", 7)))
        C.append((f"{task} std: text_route k7 vs k0", load(f, "uniform", 0), load(f"{task}_text_k7.json", "text_route_reindex", 7)))
    # hard condition
    H = {"MoveCube": ("MoveCube_ss_v1.json", "MoveCube_ss_v1_text03.json", "MoveCube_ss_v1.json"), "RouteStick": ("RouteStick_ss_v1.json", "RouteStick_ss_v1_text03.json", "RouteStick_ss_v1.json"),
         "PatternLock": ("PatternLock_ss_v1.json", "PatternLock_ss_v1_text03.json", "PatternLock_v1.json"), "VideoPlaceButton": ("VideoPlaceButton_ss_v1.json", "VideoPlaceButton_ss_v1.json", "VideoPlaceButton_v3.json"),
         "MoveCube-val": ("MoveCube_ss_val.json", "MoveCube_ss_val.json", "MoveCube_val.json"), "RouteStick-val": ("RouteStick_ss_val.json", "RouteStick_ss_val.json", "RouteStick_val.json"),
         "MoveCube-expert": ("MoveCube_expert_ss_v1.json", "MoveCube_expert_ss_v1_text03.json", "MoveCube_expert_v1.json")}
    for task, (f, tf, k0f) in H.items():
        rel = load(f, "uniform", 7, same=True); cos = load(f, "retrieval_reindex", 7, same=True); orc = load(f, "lesson_oracle_reindex", 7, same=True)
        txt = load(tf, "text_route_reindex", 7, same=True); k0 = load(k0f, "uniform", 0)
        C += [(f"{task} hard: released vs k0", k0, rel), (f"{task} hard: cosine vs released", rel, cos), (f"{task} hard: oracle vs cosine", cos, orc),
              (f"{task} hard: oracle vs k0", k0, orc), (f"{task} hard: text vs released", rel, txt), (f"{task} hard: text vs cosine", cos, txt), (f"{task} hard: text vs oracle", orc, txt), (f"{task} hard: text vs k0", k0, txt)]
        if task == "MoveCube":
            C.append(("MoveCube hard: loto6 router vs cosine", cos, load("MoveCube_ss_v1_loto3b.json", "learned_route_reindex", 7, same=True)))
        if task == "RouteStick":
            C.append(("RouteStick hard: in-task router vs oracle", orc, load("RouteStick_ss_v1_inrouter.json", "learned_route_reindex", 7, same=True)))
            C.append(("RouteStick hard: in-task router vs cosine", cos, load("RouteStick_ss_v1_inrouter.json", "learned_route_reindex", 7, same=True)))
    # fine-tuning study (test + val pooled where available)
    F = "results/robomme/ft"
    def lf(path, system, k):
        d = {}
        for r in json.load(open(path)):
            if r.get("outcome") != "error" and r["system"] == system and r["k"] == k: d[r["episode"]] = int(bool(r["success"]))
        return d
    for cellname, system, k, suf in [("MoveCube k0", "uniform", 0, "MoveCube_a"), ("MoveCube k7", "uniform", 7, "MoveCube_a"), ("MoveCube retrieval k15", "retrieval", 15, "MoveCube_b"),
                                     ("MoveCube retrieval_reindex k15", "retrieval_reindex", 15, "MoveCube_b"), ("RouteStick k0", "uniform", 0, "RouteStick_a"), ("RouteStick k7", "uniform", 7, "RouteStick_a"),
                                     ("VideoPlaceButton k0", "uniform", 0, "VideoPlaceButton_a"), ("VideoPlaceButton k7", "uniform", 7, "VideoPlaceButton_a")]:
        con = lf(f"{F}/control4k_{suf}.json", system, k)
        for m in ["shift", "interf"]:
            C.append((f"ft {cellname}: control vs {m}", con, lf(f"{F}/{m}4k_{suf}.json", system, k)))
    for sysname in ["retrieval", "retrieval_reindex"]:
        con = merge(lf(f"{F}/control4k_MoveCube_b.json", sysname, 15), lf(f"{F}/control4k_MoveCube_val_k15.json", sysname, 15))
        sh = merge(lf(f"{F}/shift4k_MoveCube_b.json", sysname, 15), lf(f"{F}/shift4k_MoveCube_val_k15.json", sysname, 15))
        rel = merge(load("MoveCube_v1.json", sysname, 15), lf(f"{F}/released_MoveCube_val_k15.json", sysname, 15))
        C.append((f"ft pooled {sysname} k15: control vs shift", con, sh)); C.append((f"ft pooled {sysname} k15: released vs shift", rel, sh))
    for tag, a, b in [("released", merge(load("MoveCube_v1.json", "retrieval", 15), lf(f"{F}/released_MoveCube_val_k15.json", "retrieval", 15)), merge(load("MoveCube_v1.json", "retrieval_reindex", 15), lf(f"{F}/released_MoveCube_val_k15.json", "retrieval_reindex", 15))),
                      ("control", merge(lf(f"{F}/control4k_MoveCube_b.json", "retrieval", 15), lf(f"{F}/control4k_MoveCube_val_k15.json", "retrieval", 15)), merge(lf(f"{F}/control4k_MoveCube_b.json", "retrieval_reindex", 15), lf(f"{F}/control4k_MoveCube_val_k15.json", "retrieval_reindex", 15))),
                      ("shift", merge(lf(f"{F}/shift4k_MoveCube_b.json", "retrieval", 15), lf(f"{F}/shift4k_MoveCube_val_k15.json", "retrieval", 15)), merge(lf(f"{F}/shift4k_MoveCube_b.json", "retrieval_reindex", 15), lf(f"{F}/shift4k_MoveCube_val_k15.json", "retrieval_reindex", 15)))]:
        C.append((f"ft pooled re-index gap {tag}: retrieval vs reindex", a, b))
    for name, a, b in C:
        if not a or not b: print(f"{name:60s} MISSING"); continue
        n, ma, mb, x, y, p = mcnemar(a, b)
        print(f"{name:60s} n={n:3d} {ma:.2f} -> {mb:.2f}  discordant +{x}/-{y}  McNemar p={p:.2g}  (Fisher {fisher(a,b):.2g})")
