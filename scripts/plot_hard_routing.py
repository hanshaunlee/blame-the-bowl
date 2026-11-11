"""Hard-routing condition (same-scene distractors): success per system with Wilson 95% intervals, one row per task.
    python scripts/plot_hard_routing.py results/robomme/hard_routing.png \
        "MoveCube|results/robomme/MoveCube_ss_v1.json" \
        "RouteStick|results/robomme/RouteStick_ss_v1.json|results/robomme/RouteStick_ss_v1_inrouter.json"
Panel spec: task|main json[|in-task-router json[|leave-one-task-out json]]. With a third field, learned_route_reindex rows
of the main json are the cross-task router (trained on another task) and those of the extra json the router trained on
this task; a fourth json holds the router trained on three OTHER tasks (leave-one-task-out); a fifth json holds the
label-free instruction router (text_route_reindex). Use "-" to skip a field.
"""
import json, sys, os, collections, math
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def wilson(p, n, z=1.96):
    d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return c-h, c+h

def load(path, agg, rename=None, only=None):
    """path may be 'a.json+b.json'; `only` restricts to (system, k) pairs (used to pull the k=0 cell from a standard-protocol file)."""
    for pth in path.split("+"):
        if not os.path.exists(pth): continue
        for r in json.load(open(pth)):
            if r.get("outcome") == "error": continue
            if only is not None and (r["system"], r["k"]) not in only: continue
            if r.get("same_scene") is False or (r.get("same_scene") in (None, 0) and r["k"] > 0): continue   # standard-protocol rows other than k=0 don't belong here
            _add(r, agg, rename)

def _add(r, agg, rename):
    s = rename.get(r["system"], r["system"]) if rename else r["system"]
    agg[(s, r["k"])].append((bool(r["success"]), r["lesson_frac"], r.get("same_frac") or 0.0))

out = sys.argv[1]; panels = [p.split("|") for p in sys.argv[2:]]
COMPACT = os.environ.get("COMPACT") == "1"   # one row of success panels only (paper main figure)
if COMPACT:
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.4), squeeze=False); axes = axes.T
else:
    fig, axes = plt.subplots(len(panels), 2, figsize=(16, 3.8 * len(panels)), squeeze=False)
for row, spec in enumerate(panels):
    task, main = spec[0], spec[1]; extra = spec[2] if len(spec) > 2 and spec[2] != "-" else None
    loto = spec[3] if len(spec) > 3 and spec[3] != "-" else None
    text = spec[4] if len(spec) > 4 and spec[4] != "-" else None
    agg = collections.defaultdict(list)
    ren = {}
    if extra: ren["learned_route_reindex"] = "router_cross"
    if text: ren["text_route_reindex"] = "text_main"   # a separate instruction-router file supersedes the main file's rows
    load(main, agg, ren or None)
    if extra: load(extra, agg, {"learned_route_reindex": "router_in"})
    if loto: load(loto, agg, {"learned_route_reindex": "router_loto"})
    if text: load(text, agg)
    order = [("uniform", 0, "k=0\n(no distractors)", "C0"), ("uniform", 7, "released\nFrameSamp", "C0"), ("retrieval_reindex", 7, "cosine\nretrieval", "C1")]
    if extra:
        order += [("router_cross", 7, "router trained\non MoveCube", "C4"), ("router_in", 7, f"router trained\non {task}", "C3")]
    elif any(k_[0] == "learned_route_reindex" for k_ in agg):
        order += [("learned_route_reindex", 7, "learned\nrouter", "C3")]
    if loto:
        n_other = {"MoveCube": 6, "RouteStick": 3}.get(task, "other")
        order += [("router_loto", 7, f"router trained on\n{n_other} other tasks", "C5")]
    if text or any(k_[0] == "text_route_reindex" for k_ in agg):
        order += [("text_route_reindex", 7, "instruction\nrouter (label-free)", "C6")]
    order += [("lesson_oracle_reindex", 7, "lesson\noracle", "C2")]
    xs = np.arange(len(order))
    specs = [("success", 0, "success")] if COMPACT else [("success", 0, "success"), ("share of the 32 memory slots", None, "share")]
    for ax, (title, idx, ylabel) in zip(axes[row], specs):
        for i, (s, k, lab, col) in enumerate(order):
            v = agg[(s, k)]; n = len(v)
            if n == 0:
                ax.text(i, 0.5, "pending", ha="center", va="center", fontsize=8, color="gray"); continue
            if idx == 0:
                p = np.mean([x[0] for x in v]); lo, hi = wilson(p, n)
                ax.bar(i, p, color=col, alpha=.85); ax.errorbar(i, p, yerr=[[p-lo], [hi-p]], color="k", capsize=3)
            else:
                les = np.mean([x[1] for x in v]); same = np.mean([x[2] for x in v])
                ax.bar(i-0.18, les, width=.36, color="C2", alpha=.85, label="lesson session" if i == 0 else None)
                ax.bar(i+0.18, same, width=.36, color="C7", alpha=.85, label="same-scene sessions" if i == 0 else None)
        ax.set_xticks(xs); ax.set_xticklabels([o[2] for o in order], fontsize=7); ax.set_ylim(0, 1); ax.set_ylabel(ylabel)
        ax.set_title(f"{task}: {title}", fontsize=10); ax.grid(axis="y", alpha=.3)
    if not COMPACT: axes[row][1].legend(fontsize=8, loc="upper right")
if not COMPACT:
    fig.suptitle("Hard routing: 3 of the 7 distractor sessions are recorded in the query scene (k=7, n=50 per cell; all routed systems use re-indexing)", fontsize=10)
plt.tight_layout(); plt.savefig(out, dpi=170, bbox_inches="tight"); print("saved", out)
