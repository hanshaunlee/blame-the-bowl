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
def diff_ci(a, b, z=1.959964):
    """95% CI of the paired difference mean(b) - mean(a) on common episodes (Bonett-Price adjusted Wald)."""
    es = sorted(set(a) & set(b)); n = len(es)
    x = sum(1 for e in es if a[e] == 0 and b[e] == 1); y = sum(1 for e in es if a[e] == 1 and b[e] == 0)
    p1, p2 = (x + 1) / (n + 2), (y + 1) / (n + 2)
    h = z * ((p1 + p2 - (p1 - p2) ** 2) / (n + 2)) ** 0.5
    return (x - y) / n, max(-1.0, p1 - p2 - h), min(1.0, p1 - p2 + h)
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
    # tranche 7 controls: router ladder, comb isolation, index magnitude vs gap, boundary sensitivity, allocation dose-response
    import os
    for task in ["RouteStick", "PatternLock", "MoveCube"]:
        f, tf, _ = H[task]; lad = f"{task}_ss_t7_ladder.json"
        txt = load(tf, "text_route_reindex", 7, same=True); cos = load(f, "retrieval_reindex", 7, same=True); rel = load(f, "uniform", 7, same=True)
        for s in ["scene_route_reindex", "text_route_shuf_reindex", "retrieval_once_reindex"]:
            C.append((f"{task} hard ladder: {s} vs text", load(lad, s, 7, same=True), txt))
        C.append((f"{task} hard ladder: cosine per-step vs committed once", cos, load(lad, "retrieval_once_reindex", 7, same=True)))
        C.append((f"{task} hard ladder: released vs shuffled instruction", rel, load(lad, "text_route_shuf_reindex", 7, same=True)))
    for task, k0f in [("MoveCube", "MoveCube_v1.json"), ("VideoPlaceButton", "VideoPlaceButton_v3.json")]:
        de, st = load(f"{task}_t7_comb.json", "comb_dense", 0), load(f"{task}_t7_comb.json", "comb_stride4", 0)
        C += [(f"{task} comb k0: dense vs stride4", de, st), (f"{task} comb k0: released vs dense", load(k0f, "uniform", 0), de), (f"{task} comb k0: released vs stride4", load(k0f, "uniform", 0), st)]
    u0, sh, al = load("MoveCube_v1.json", "uniform", 0), load("MoveCube_t7_shift.json", "uniform_shift480", 0), load("MoveCube_t7_shift.json", "uniform_allshift480", 0)
    C += [("MoveCube k0 index: released vs episode+480", u0, sh), ("MoveCube k0 index: released vs all+480 (gap kept)", u0, al), ("MoveCube k0 index: episode+480 vs all+480", sh, al)]
    rt = load("RouteStick_ss_v1_text03.json", "text_route_reindex", 7, same=True)
    for b in ["split", "merge"]:
        C.append((f"RouteStick hard boundary: text vs lesson {b}", rt, load(f"RouteStick_ss_t7_{b}.json", "text_route_reindex", 7, same=True)))
    for task in ["MoveCube", "RouteStick"]:
        A = {}
        for suf in ["a", "b", "c", "d", "ri"]:
            p = f"{task}_t7_alloc_{suf}.json"
            if os.path.exists(f"{R}/{p}"):
                for s in {r["system"] for r in json.load(open(f"{R}/{p}"))}: A[s] = load(p, s, 7)
        for a_, b_ in [("alloc0", "alloc16"), ("alloc12", "alloc16"), ("alloc16", "alloc16r"), ("alloc0", "alloc16e8"), ("alloc16e8", "alloc20e8"), ("alloc20e8", "alloc24e8"), ("alloc16e8", "alloc24e8"), ("alloc24e8", "alloc24re8"), ("alloc16", "alloc16e8")]:
            if a_ in A and b_ in A: C.append((f"{task} k7 dose: {a_} vs {b_}", A[a_], A[b_]))
        for a_, b_ in [("alloc8", "alloc8_reindex"), ("alloc16", "alloc16_reindex"), ("alloc24e8", "alloc24e8_reindex"), ("alloc8_reindex", "alloc16_reindex"), ("alloc16_reindex", "alloc24e8_reindex"), ("alloc8_reindex", "alloc24e8_reindex")]:
            if a_ in A and b_ in A: C.append((f"{task} k7 dose: {a_} vs {b_}", A[a_], A[b_]))
        if "alloc24e8_reindex" in A: C.append((f"{task} k7 dose: alloc24e8_reindex vs retrieval_reindex", A["alloc24e8_reindex"], load(f"{task}_v1.json", "retrieval_reindex", 7)))
        ref = load(f"{task}_v1.json", "retrieval_reindex", 7)   # all history slots on the lesson (routing is ~1.0 on the standard protocol)
        if "alloc24e8" in A: C.append((f"{task} k7 dose: alloc24e8 vs retrieval_reindex", A["alloc24e8"], ref))
    st = {s_: load("MoveCube_ss_t7_same.json", s_, 7, same=True) for s_ in ["uniform", "retrieval_reindex", "text_route_reindex", "lesson_oracle_reindex"]}   # same-task same-scene competitors
    C += [("MoveCube same-task: released vs cosine", st["uniform"], st["retrieval_reindex"]), ("MoveCube same-task: released vs text", st["uniform"], st["text_route_reindex"]),
          ("MoveCube same-task: cosine vs text", st["retrieval_reindex"], st["text_route_reindex"]), ("MoveCube same-task: text vs oracle", st["text_route_reindex"], st["lesson_oracle_reindex"]),
          ("MoveCube same-task: oracle vs k0", load("MoveCube_v1.json", "uniform", 0), st["lesson_oracle_reindex"])]
    for task in ["MoveCube", "RouteStick"]:   # tranche 8: dose-response on retrieval's own schedule
        P = {f: load(f"{task}_t8_allocp.json", f"allocp{f}_reindex", 7) for f in (25, 50, 75)}; P[100] = load(f"{task}_v1.json", "retrieval_reindex", 7); u = load(f"{task}_v1.json", "uniform", 7)
        C += [(f"{task} k7 allocp: {x} vs {y}", P[x], P[y]) for x, y in [(25, 50), (50, 75), (75, 100), (25, 75), (25, 100), (50, 100)]] + [(f"{task} k7 allocp: released vs {x}", u, P[x]) for x in (25, 50, 75)]
    st = {s_: load("RouteStick_ss_t8_same.json", s_, 7, same=True) for s_ in ["uniform", "retrieval_reindex", "text_route_reindex", "lesson_oracle_reindex"]}
    C += [("RouteStick same-task: released vs cosine", st["uniform"], st["retrieval_reindex"]), ("RouteStick same-task: released vs text", st["uniform"], st["text_route_reindex"]),
          ("RouteStick same-task: cosine vs text", st["retrieval_reindex"], st["text_route_reindex"]), ("RouteStick same-task: text vs oracle", st["text_route_reindex"], st["lesson_oracle_reindex"]),
          ("RouteStick same-task: oracle vs k0", load("RouteStick_v1.json", "uniform", 0), st["lesson_oracle_reindex"])]
    for task, f_, nm in [("MoveCube", "inrouter", "in-task router"), ("RouteStick", "inrouter", "in-task router"), ("RouteStick", "loto", "loto router")]:   # tranche 9: learned routers (never trained on same-task competitors)
        sf = f"{task}_ss_t{7 if task == 'MoveCube' else 8}_same.json"; lr = load(f"{task}_ss_t9_same_{f_}.json", "learned_route_reindex", 7, same=True)
        C += [(f"{task} same-task: {x} vs {nm}", load(sf, y, 7, same=True), lr) for x, y in [("released", "uniform"), ("cosine", "retrieval_reindex"), ("text", "text_route_reindex"), ("oracle", "lesson_oracle_reindex")]]
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
        d, lo, hi = diff_ci(a, b)
        print(f"{name:60s} n={n:3d} {ma:.2f} -> {mb:.2f}  diff {d:+.2f} [{lo:+.2f},{hi:+.2f}]  discordant +{x}/-{y}  McNemar p={p:.2g}  (Fisher {fisher(a,b):.2g})")
    print("\nfine-tuning: absolute k=7 and the k=0 -> k=7 gap per model (paired within model)")
    rel = {"MoveCube": "MoveCube_v1.json", "RouteStick": "RouteStick_v1.json", "VideoPlaceButton": "VideoPlaceButton_v3.json"}
    for task in rel:
        for m in ["released", "control", "shift", "interf"]:
            k0 = load(rel[task], "uniform", 0) if m == "released" else lf(f"{F}/{m}4k_{task}_a.json", "uniform", 0)
            k7 = load(rel[task], "uniform", 7) if m == "released" else lf(f"{F}/{m}4k_{task}_a.json", "uniform", 7)
            d, lo, hi = diff_ci(k0, k7); n = len(set(k0) & set(k7))
            print(f"  {task:17s} {m:9s} k0 {sum(k0.values())/len(k0):.2f}  k7 {sum(k7.values())/len(k7):.2f}  gap {-d:.2f} [{-hi:+.2f},{-lo:+.2f}] (n={n})")
