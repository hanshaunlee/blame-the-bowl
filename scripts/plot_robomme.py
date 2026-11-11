"""Success vs interference level per memory system, with Wilson 95% intervals.

    python scripts/plot_robomme.py results/robomme/VideoRepick_v3.json results/robomme/VideoPlaceButton_v3.json
"""
import json, sys, collections, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["uniform", "recency", "retrieval", "retrieval_reindex", "text_route_reindex", "lesson_oracle", "lesson_oracle_reindex", "lesson_late", "two_scale", "learned"]
COLOR = {"uniform": "C0", "recency": "C7", "retrieval": "C1", "retrieval_reindex": "C2", "lesson_oracle": "C4", "lesson_oracle_reindex": "C5",
         "lesson_late": "C6", "two_scale": "C8", "learned": "C3", "text_route_reindex": "C9"}
LABEL = {"uniform": "FrameSamp (released)", "recency": "recency", "retrieval": "retrieval", "lesson_oracle": "lesson session (oracle)", "retrieval_reindex": "retrieval + re-index", "lesson_oracle_reindex": "lesson session (oracle) + re-index",
         "lesson_late": "lesson, late segment", "two_scale": "two-scale (lesson + recent window)", "learned": "learned (demo-loss GRPO)", "text_route_reindex": "instruction router (label-free) + re-index"}


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def table(path):
    """path may be 'a.json+b.json' to merge several result files into one panel."""
    agg = collections.defaultdict(list); seen = set()
    for pth in path.split("+"):   # earlier files take precedence when a (system, k, episode) cell was rerun
        for x in json.load(open(pth)):
            key = (x["system"], x["k"], x["episode"])
            if x.get("outcome") != "error" and key not in seen:
                seen.add(key); agg[(x["system"], x["k"])].append(bool(x["success"]))
    # at k=0 the history-only systems see exactly the released frames and positions (verified cell-by-cell), so a
    # missing k=0 cell is filled from the released baseline for plotting
    for s in ("retrieval", "retrieval_reindex", "lesson_oracle", "lesson_oracle_reindex", "text_route_reindex"):
        if any(k_ == s for k_, _ in agg) and (s, 0) not in agg and ("uniform", 0) in agg:
            agg[(s, 0)] = list(agg[("uniform", 0)])
    return agg


def main(paths):
    import os
    systems = os.environ.get("SYSTEMS", "").split(",") if os.environ.get("SYSTEMS") else ORDER
    ncol = min(3, len(paths)); nrow = (len(paths) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow), squeeze=False)
    for ax, path in zip(axes.flat, paths):
        agg = table(path); stem = path.split("+")[0].split("/")[-1][:-5]; task = stem.split("_")[0] + (" (same-task distractors)" if stem.endswith("_same") else "") + (" (expert checkpoint)" if "_expert" in stem else "")
        ks = sorted({k for _, k in agg})
        for s in systems:
            if not any((s, k) in agg for k in ks):
                continue
            ps = [np.mean(agg[(s, k)]) if (s, k) in agg else np.nan for k in ks]
            lo, hi = zip(*[wilson(p, len(agg.get((s, k), []))) for p, k in zip(ps, ks)])
            ax.errorbar(ks, ps, yerr=[np.array(ps) - np.array(lo), np.array(hi) - np.array(ps)], marker="o", capsize=3, label=LABEL.get(s, s), color=COLOR.get(s))
        ax.set_title(f"{task}, n={max(len(v) for v in agg.values())}", fontsize=10); ax.set_xlabel("distractor sessions k"); ax.set_ylabel("success")
        ax.set_xticks(ks); ax.set_ylim(0, 1); ax.grid(alpha=.3)
        print(f"\n{task}")
        for s in ORDER:
            row = [f"{np.mean(agg[(s, k)]):.2f} (n={len(agg[(s, k)])})" if (s, k) in agg else "-" for k in ks]
            if any(c != "-" for c in row):
                print(f"  {s:14s} " + "  ".join(row))
    for ax in list(axes.flat)[len(paths):]: ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    for ax in list(axes.flat)[1:len(paths)]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l: h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=min(5, len(l)), fontsize=9, frameon=False)
    out = os.environ.get("OUT", "results/robomme/success_vs_k.png"); plt.tight_layout(rect=(0, 0.06 if nrow > 1 else 0.12, 1, 1)); plt.savefig(out, dpi=150); print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1:])
