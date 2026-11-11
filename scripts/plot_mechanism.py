"""Mechanism figure: success relative to the task's k=0 level against the lesson's mean share of the 32 memory slots,
one point per evaluated cell (system x k x task x condition x checkpoint), from every n=50 result file.
    python scripts/plot_mechanism.py paper/fig/mechanism.png"""
import json, glob, sys, collections, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "results/robomme"
cells = collections.defaultdict(list)     # key -> list of (success, lesson_frac)
for f in sorted(glob.glob(f"{R}/*.json")):
    base = os.path.basename(f)
    if "screen" in base or "pilot" in base or "_v0" in base or "text05" in base or "loto.json" in base or "loto2" in base: continue
    try: rows = json.load(open(f))
    except Exception: continue
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict) or "system" not in rows[0]: continue
    for r in rows:
        if r.get("outcome") == "error" or r.get("lesson_frac") is None: continue
        ck = "expert" if "expert" in base else "modul"
        split = r.get("split") or "test"
        key = (r["task"], ck, split, bool(r.get("same_scene")), r["system"], r["k"])
        cells[key].append((bool(r["success"]), float(r["lesson_frac"])))
agg = {k: (np.mean([s for s, _ in v]), np.mean([l for _, l in v]), len(v)) for k, v in cells.items() if len(v) >= 50}
k0 = {}
for (task, ck, split, same, system, k), (s, l, n) in agg.items():
    if k == 0 and system == "uniform": k0[(task, ck, split)] = s
groups = {"released (even sampling)": ("C0", "o"), "cosine retrieval": ("C1", "o"), "routers (learned, instruction)": ("C6", "s"),
          "lesson oracle": ("C2", "^"), "other fixed layouts (recency, two-scale, late lesson, demo-loss selector)": ("0.6", "x")}
def group(system):
    if system == "uniform": return "released (even sampling)"
    if system.startswith("retrieval"): return "cosine retrieval"
    if system.startswith("learned_route") or system.startswith("text_route"): return "routers (learned, instruction)"
    if system.startswith("lesson_oracle"): return "lesson oracle"
    return "other fixed layouts (recency, two-scale, late lesson, demo-loss selector)"
fig, ax = plt.subplots(figsize=(8.5, 3.0))
done = set(); npts = 0
for (task, ck, split, same, system, k), (s, l, n) in agg.items():
    if (task, ck, split) not in k0 or k0[(task, ck, split)] < 0.25 or k == 0: continue   # VideoRepick's ceiling is too low to normalise
    g = group(system); col, mk = groups[g]
    ax.scatter(l, s / k0[(task, ck, split)], color=col, marker=mk, s=28 if not same else 46, alpha=.85,
               edgecolor="k" if same else "none", linewidth=.6, label=g if g not in done else None); done.add(g); npts += 1
ax.axhline(1.0, color="k", ls="--", lw=.8); ax.axvspan(0.68, 0.92, color="C2", alpha=.08)
ax.set_xlabel("lesson's mean share of the 32 memory slots"); ax.set_ylabel("success / success at $k{=}0$")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.25); ax.grid(alpha=.3)
ax.legend(fontsize=6.5, loc="lower right", frameon=True, framealpha=.9)
ax.set_title(f"{npts} cells, $n{{=}}50$ each; black edge: hard condition", fontsize=9)
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=170, bbox_inches="tight"); print("saved", sys.argv[1], npts, "cells")
