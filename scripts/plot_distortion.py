"""Figure for the policy-free decision-distortion table (results/distortion.json)."""
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "results/distortion.json"
R = json.load(open(path))
order = ["none", "random", "recency_frame", "recency_sym", "scripted_mem_dropfail", "scripted_mem",
         "learned_online_sym", "learned_sleep_sym", "learned_both_sym", "learned_split_sym",
         "learned_online", "learned_sleep", "learned_both", "learned_split", "oracle"]
systems = [s for s in order if any(k.startswith(s + "/") for k in R)]
budgets = sorted({int(k.split("/B")[1]) for k in R})
ks = [0, 1, 3, 7]
colors = {s: c for s, c in zip(systems, plt.cm.tab20(np.linspace(0, 1, len(systems))))}
fig, axes = plt.subplots(1, len(budgets), figsize=(4.4 * len(budgets), 3.8), squeeze=False)
for ax, B in zip(axes[0], budgets):
    for s in systems:
        key = f"{s}/B{B}"
        if key not in R:
            continue
        ys = [R[key].get(f"query/k{k}", (np.nan,))[0] for k in ks]
        ax.plot(ks, ys, marker="o", label=s, color=colors[s], lw=2.2 if "learned" in s else 1.2, ls="--" if s == "oracle" else "-")
    ax.set_title(f"expert agreement given memory, B={B}"); ax.set_xlabel("distractor sessions k"); ax.set_xticks(ks); ax.set_ylim(0.3, 1.02); ax.grid(alpha=.3)
axes[0][0].set_ylabel("agreement with full-history expert (query steps)")
axes[0][-1].legend(fontsize=6.5, loc="lower left")
plt.tight_layout(); plt.savefig(path.replace(".json", ".png"), dpi=130)
print("saved", path.replace(".json", ".png"))
