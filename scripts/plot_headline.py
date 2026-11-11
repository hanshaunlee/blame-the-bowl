"""Headline figure: at k=7 the released sampler collapses, a lesson-routed budget is at the k=0 level, on every task,
under the standard protocol (left) and the hard condition with three same-scene replays (right).
    python scripts/plot_headline.py paper/fig/headline.png
"""
import json, sys, math, collections
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "results/robomme"
def wilson(p, n, z=1.96):
    d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return c-h, c+h
def cell(files, system, k, same=None, split="test"):
    v = []
    for f in files:
        for r in json.load(open(f"{R}/{f}")):
            if r.get("outcome") == "error" or r["system"] != system or r["k"] != k: continue
            if same is not None and bool(r.get("same_scene")) != same: continue
            if same is False and r.get("same_scene") is None and r["k"] > 0 and False: continue
            if r.get("split", "test") not in (split, None): continue
            v.append(bool(r["success"]))
    return (np.mean(v), len(v)) if v else (np.nan, 0)
std = {  # task: (k0 files, released k7 files, retrieval_reindex files, text files)
 "MoveCube": (["MoveCube_v1.json"], ["MoveCube_v1.json"], ["MoveCube_v1.json"], ["MoveCube_text_k7.json"]),
 "RouteStick": (["RouteStick_v1.json"], ["RouteStick_v1.json"], ["RouteStick_v1.json"], ["RouteStick_text_k7.json"]),
 "PatternLock": (["PatternLock_v1.json"], ["PatternLock_v1.json"], ["PatternLock_v1.json"], ["PatternLock_text_k7.json"]),
 "VideoPlaceButton": (["VideoPlaceButton_v3.json"], ["VideoPlaceButton_v3.json"], ["VideoPlaceButton_v3.json"], ["VideoPlaceButton_text_k7.json"]),
 "VideoRepick": (["VideoRepick_v3.json"], ["VideoRepick_v3.json"], ["VideoRepick_reindex.json"], ["VideoRepick_text_k7.json"]),
}
hard = {  # task: (main ss file, text file, k0 file)
 "MoveCube": ("MoveCube_ss_v1.json", "MoveCube_ss_v1_text03.json", "MoveCube_ss_v1.json"),
 "RouteStick": ("RouteStick_ss_v1.json", "RouteStick_ss_v1_text03.json", "RouteStick_ss_v1.json"),
 "PatternLock": ("PatternLock_ss_v1.json", "PatternLock_ss_v1_text03.json", "PatternLock_v1.json"),
 "VideoPlaceButton": ("VideoPlaceButton_ss_v1.json", "VideoPlaceButton_ss_v1.json", "VideoPlaceButton_v3.json"),
}
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), gridspec_kw={"width_ratios": [5, 4]})
cols = {"released": "C0", "cosine": "C1", "text": "C6", "oracle": "C2"}
def panel(ax, tasks, bars, title):
    w = 0.8 / len(bars); xs = np.arange(len(tasks))
    for j, (lab, col, getter) in enumerate(bars):
        for i, t in enumerate(tasks):
            p, n = getter(t)
            if n == 0: continue
            lo, hi = wilson(p, n)
            ax.bar(xs[i] + (j - (len(bars)-1)/2) * w, p, width=w*0.92, color=col, alpha=.9, label=lab if i == 0 else None)
            ax.errorbar(xs[i] + (j - (len(bars)-1)/2) * w, p, yerr=[[p-lo], [hi-p]], color="k", capsize=2, lw=.8)
    for i, t in enumerate(tasks):
        p0, n0 = k0s[t]
        ax.hlines(p0, xs[i]-0.45, xs[i]+0.45, color="k", ls="--", lw=1.2, label="$k{=}0$ (no distractors)" if i == 0 else None)
    ax.set_xticks(xs); ax.set_xticklabels(tasks, fontsize=9); ax.set_ylim(0, 1); ax.set_ylabel("success at $k{=}7$")
    ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=.3)
k0s = {t: cell(std[t][0], "uniform", 0) for t in std}
panel(axes[0], list(std),
      [("released FrameSamp (even sampling)", cols["released"], lambda t: cell(std[t][1], "uniform", 7, same=False)),
       ("cosine retrieval + re-index", cols["cosine"], lambda t: cell(std[t][2], "retrieval_reindex", 7, same=False)),
       ("instruction router + re-index (label-free)", cols["text"], lambda t: cell(std[t][3], "text_route_reindex", 7, same=False))],
      "Standard protocol: 7 distractor sessions of other tasks")
k0s = {t: cell([hard[t][2]], "uniform", 0) for t in hard}
panel(axes[1], list(hard),
      [("released FrameSamp (even sampling)", cols["released"], lambda t: cell([hard[t][0]], "uniform", 7, same=True)),
       ("cosine retrieval + re-index", cols["cosine"], lambda t: cell([hard[t][0]], "retrieval_reindex", 7, same=True)),
       ("instruction router + re-index (label-free)", cols["text"], lambda t: cell([hard[t][1]], "text_route_reindex", 7, same=True)),
       ("lesson oracle + re-index", cols["oracle"], lambda t: cell([hard[t][0]], "lesson_oracle_reindex", 7, same=True))],
      "Hard condition: 3 of the 7 distractors recorded in the query scene")
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout(rect=(0, 0.05, 1, 1)); plt.savefig(sys.argv[1], dpi=170, bbox_inches="tight"); print("saved", sys.argv[1])
