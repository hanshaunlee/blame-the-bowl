"""Markdown summary of results/sweep.json (closed-loop) and results/distortion.json (policy-free)."""
import json, sys, os, numpy as np

sweep = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/sweep.json"))
dist_path = sys.argv[2] if len(sys.argv) > 2 else "results/distortion.json"
dist = json.load(open(dist_path)) if os.path.exists(dist_path) else {}
order = ["none", "random", "recency_frame", "recency_sym", "scripted_mem_dropfail", "scripted_mem",
         "learned_online", "learned_sleep", "learned_both", "learned_split", "learned_online_sym", "learned_sleep_sym", "learned_both_sym", "learned_split_sym", "oracle"]
budgets = sorted({int(k.split("/B")[1]) for k in sweep})
systems = [s for s in order if any(k.startswith(s + "/") for k in sweep)]
ks = [0, 1, 3, 7]
tasks = ["find", "count", "sequence", "adapt", "tidy"]

print("## Closed-loop query success by interference level k (cross-session trials)\n")
for B in budgets:
    print(f"**Budget B={B}**\n")
    print("| system | k=0 | k=1 | k=3 | k=7 | mean | single-task mean | frame frac |")
    print("|---|---|---|---|---|---|---|---|")
    for s in systems:
        key = f"{s}/B{B}"
        if key not in sweep:
            continue
        summ = sweep[key]["summary"]
        q = [summ.get(f"query/{k}", (np.nan,))[0] for k in ks]
        st = [summ.get(f"single/{t}", (np.nan,))[0] for t in tasks]
        print(f"| {s} | " + " | ".join(f"{v:.2f}" for v in q) + f" | {np.nanmean(q):.2f} | {np.nanmean(st):.2f} | {sweep[key]['frac_frame']:.2f} |")
    print()

print("## Single-session success (largest budget)\n")
B = budgets[-1]
print("| system | " + " | ".join(tasks) + " |")
print("|---|" + "---|" * len(tasks))
for s in systems:
    key = f"{s}/B{B}"
    if key in sweep:
        summ = sweep[key]["summary"]
        print(f"| {s} | " + " | ".join(f"{summ.get(f'single/{t}', (np.nan,))[0]:.2f}" for t in tasks) + " |")
print()

if dist:
    print("## Policy-free decision distortion: expert agreement given only the memory (query steps)\n")
    for B in budgets:
        print(f"**B={B}**\n")
        print("| system | k=0 | k=1 | k=3 | k=7 | within-session |")
        print("|---|---|---|---|---|---|")
        for s in systems:
            key = f"{s}/B{B}"
            if key not in dist:
                continue
            d = dist[key]
            print(f"| {s} | " + " | ".join(f"{d.get(f'query/k{k}', (np.nan,))[0]:.2f}" for k in ks) + f" | {d.get('within_session', (np.nan,))[0]:.2f} |")
        print()

print("## What the memories hold at query time (event-type histogram, cross trials)\n")
for B in budgets:
    for s in systems:
        key = f"{s}/B{B}"
        comp = sweep.get(key, {}).get("composition", {})
        if comp:
            tot = sum(comp.values())
            top = sorted(comp.items(), key=lambda kv: -kv[1])[:6]
            print(f"- {s} B={B}: " + ", ".join(f"{k} {v/tot:.0%}" for k, v in top))
