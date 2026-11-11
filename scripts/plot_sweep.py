"""Figures from results/sweep.json: query success vs interference k, vs budget B, single-task bars."""
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "results/sweep.json"
R = json.load(open(path))
systems = sorted({k.split("/")[0] for k in R})
budgets = sorted({int(k.split("/B")[1]) for k in R})
order = ["none", "random", "recency_frame", "recency_sym", "scripted_mem_dropfail", "scripted_mem",
         "learned_online", "learned_sleep", "learned_both", "learned_split", "learned_online_sym", "learned_sleep_sym", "learned_both_sym", "learned_split_sym", "oracle"]
systems = [s for s in order if s in systems] + [s for s in systems if s not in order]
colors = {s: c for s, c in zip(systems, plt.cm.tab10(np.linspace(0, 1, len(systems))))}
ks = [0, 1, 3, 7]

fig, axes = plt.subplots(1, len(budgets) + 1, figsize=(4.2 * (len(budgets) + 1), 3.8))
for ax, B in zip(axes, budgets):
    for s in systems:
        key = f"{s}/B{B}"
        if key not in R:
            continue
        summ = R[key]["summary"]
        ys = [summ.get(f"query/{k}", (np.nan,))[0] for k in ks]
        ax.plot(ks, ys, marker="o", label=s, color=colors[s], lw=2 if s.startswith("learned") else 1.2,
                ls="--" if s == "oracle" else "-")
    ax.set_title(f"query success, budget B={B}"); ax.set_xlabel("distractor sessions k"); ax.set_ylim(0, 1.02); ax.set_xticks(ks); ax.grid(alpha=.3)
axes[0].set_ylabel("success rate")
# budget curve averaged over k
ax = axes[-1]
for s in systems:
    ys = []
    for B in budgets:
        key = f"{s}/B{B}"
        ys.append(np.nanmean([R[key]["summary"].get(f"query/{k}", (np.nan,))[0] for k in ks]) if key in R else np.nan)
    ax.plot(budgets, ys, marker="s", label=s, color=colors[s], lw=2 if s.startswith("learned") else 1.2, ls="--" if s == "oracle" else "-")
ax.set_xscale("log", base=2); ax.set_xticks(budgets); ax.set_xticklabels(budgets); ax.set_xlabel("memory budget B (slots)")
ax.set_title("query success, mean over k"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
ax.legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig(path.replace(".json", "_query.png"), dpi=130)

# single-session tasks
tasks = ["find", "count", "sequence", "adapt", "tidy"]
fig, axes = plt.subplots(1, len(budgets), figsize=(4.6 * len(budgets), 3.6), squeeze=False)
for ax, B in zip(axes[0], budgets):
    w = 0.8 / len(systems)
    for i, s in enumerate(systems):
        key = f"{s}/B{B}"
        if key not in R:
            continue
        summ = R[key]["summary"]
        ys = [summ.get(f"single/{t}", (np.nan,))[0] for t in tasks]
        ax.bar(np.arange(len(tasks)) + i * w, ys, w, label=s, color=colors[s])
    ax.set_xticks(np.arange(len(tasks)) + 0.4); ax.set_xticklabels(tasks); ax.set_ylim(0, 1.02); ax.set_title(f"single-session tasks, B={B}"); ax.grid(alpha=.3, axis="y")
axes[0][0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(path.replace(".json", "_single.png"), dpi=130)

# table
print(f"{'system':24s} " + " ".join(f"B{B}:k0  k1  k3  k7 |" for B in budgets))
for s in systems:
    row = f"{s:24s} "
    for B in budgets:
        key = f"{s}/B{B}"
        if key in R:
            summ = R[key]["summary"]
            row += " ".join(f"{summ.get(f'query/{k}', (np.nan,))[0]:.2f}" for k in ks) + " | "
        else:
            row += "  -    -    -    -  | "
    print(row)
print("\nsingle-session (B = max):")
B = budgets[-1]
for s in systems:
    key = f"{s}/B{B}"
    if key in R:
        summ = R[key]["summary"]
        print(f"{s:24s} " + " ".join(f"{t}={summ.get(f'single/{t}', (np.nan,))[0]:.2f}" for t in tasks) + f"  frame-frac={R[key]['frac_frame']:.2f}")
