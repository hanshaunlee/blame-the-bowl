"""Appendix assets for the matched controls, generated from the per-episode result files so no number is typed by hand.
    python scripts/make_controls_assets.py     # writes paper/fig/controls.png and paper/ci_table.tex"""
import json, math, sys
sys.path.insert(0, "scripts")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from paired_tests import load, mcnemar, diff_ci

def wilson(p, n, z=1.96):
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
def rate(d):
    p = sum(d.values()) / len(d); lo, hi = wilson(p, len(d)); return p, p - lo, hi - p

# ---------------------------------------------------------------- figure: dose-response and router ladder
fig, ax = plt.subplots(1, 2, figsize=(11, 3.3), gridspec_kw=dict(width_ratios=[1, 1.5]))
for task, c in [("MoveCube", "C0"), ("RouteStick", "C3")]:
    P = {f: load(f"{task}_t8_allocp.json", f"allocp{f}_reindex", 7) for f in (25, 50, 75)}; P[100] = load(f"{task}_v1.json", "retrieval_reindex", 7)
    xs = [25, 50, 75, 100]; r = [rate(P[x]) for x in xs]
    ax[0].errorbar(xs, [a for a, _, _ in r], yerr=[[b for _, b, _ in r], [b for _, _, b in r]], color=c, marker="o", capsize=3, label=task)
    ax[0].axhline(rate(load(f"{task}_v1.json", "uniform", 7))[0], color=c, ls=":", lw=1)
    ax[0].axhline(rate(load(f"{task}_v1.json", "uniform", 0))[0], color=c, ls="--", lw=1)
ax[0].set_xticks([25, 50, 75, 100]); ax[0].set_xlabel("lesson slots kept (% of retrieval + re-index)"); ax[0].set_ylabel("success"); ax[0].set_ylim(0, 1.05)
ax[0].set_title("(a) dose-response, $k{=}7$ (dotted: released; dashed: $k{=}0$)", fontsize=10); ax[0].legend(frameon=False, loc="upper left", ncol=2, fontsize=9, bbox_to_anchor=(0, 1.02), borderaxespad=0.2)
H = {"RouteStick": ("RouteStick_ss_v1.json", "RouteStick_ss_v1_text03.json"), "PatternLock": ("PatternLock_ss_v1.json", "PatternLock_ss_v1_text03.json"), "MoveCube": ("MoveCube_ss_v1.json", "MoveCube_ss_v1_text03.json")}
names = ["released", "cosine,\nper step", "cosine,\ncommitted", "scene\nonly", "wrong\ninstruction", "instruction\nrouter", "oracle"]; cols = ["0.55", "C1", "C1", "C4", "C5", "C6", "C2"]
for j, (task, (f, tf)) in enumerate(H.items()):
    lad = f"{task}_ss_t7_ladder.json"
    cells = [load(f, "uniform", 7, same=True), load(f, "retrieval_reindex", 7, same=True), load(lad, "retrieval_once_reindex", 7, same=True), load(lad, "scene_route_reindex", 7, same=True),
             load(lad, "text_route_shuf_reindex", 7, same=True), load(tf, "text_route_reindex", 7, same=True), load(f, "lesson_oracle_reindex", 7, same=True)]
    for i, d in enumerate(cells):
        p, lo, hi = rate(d)
        ax[1].bar(j + (i - 3) * .12, p, .11, color=cols[i], yerr=[[lo], [hi]], capsize=1.5, error_kw=dict(lw=.7), label=names[i].replace("\n", " ") if j == 0 else None,
                  hatch="//" if i in (2, 3, 4) else None, edgecolor="white")
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(list(H)); ax[1].set_ylim(0, 1); ax[1].set_title("(b) router ladder, hard condition (hatched: controls)", fontsize=10)
ax[1].legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center")
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("paper/fig/controls.png", dpi=200)

# ---------------------------------------------------------------- table: paired differences with intervals for the main tables
def cell(a, b):
    d, lo, hi = diff_ci(a, b); f = lambda x: f"{x:+.2f}".replace("0.", ".")
    return f"${f(d)}$ $[{f(lo)},{f(hi)}]$"
rows = []
STD = [("MoveCube, modul", "MoveCube_v1.json", "MoveCube_v1.json"), ("MoveCube, expert", "MoveCube_expert_v1.json", "MoveCube_expert_v1.json"), ("PatternLock", "PatternLock_v1.json", "PatternLock_v1.json"),
       ("RouteStick", "RouteStick_v1.json", "RouteStick_v1.json"), ("VideoPlaceButton", "VideoPlaceButton_v3.json", "VideoPlaceButton_v3.json"), ("VideoRepick", "VideoRepick_v3.json", "VideoRepick_reindex.json")]
out = ["\\begin{tabular}{lccc}", "\\toprule", "standard protocol, $k{=}7$ & released $-$ $k{=}0$ & fix $-$ released & fix $-$ $k{=}0$\\\\", "\\midrule"]
for nm, f, rf in STD:
    k0, rel, fix = load(f, "uniform", 0), load(f, "uniform", 7), load(rf, "retrieval_reindex", 7)
    out.append(f"{nm} & {cell(k0, rel)} & {cell(rel, fix)} & {cell(k0, fix)}\\\\")
out += ["\\bottomrule", "\\end{tabular}", "", "\\vspace{4pt}", "\\begin{tabular}{lcccc}", "\\toprule", "hard condition & released $-$ $k{=}0$ & oracle $-$ $k{=}0$ & router $-$ released & router $-$ oracle\\\\", "\\midrule"]
HH = [("MoveCube, test", "MoveCube_ss_v1.json", "MoveCube_ss_v1_text03.json", "MoveCube_ss_v1.json"), ("MoveCube, val", "MoveCube_ss_val.json", "MoveCube_ss_val.json", "MoveCube_val.json"),
      ("MoveCube, expert", "MoveCube_expert_ss_v1.json", "MoveCube_expert_ss_v1_text03.json", "MoveCube_expert_v1.json"), ("RouteStick, test", "RouteStick_ss_v1.json", "RouteStick_ss_v1_text03.json", "RouteStick_ss_v1.json"),
      ("RouteStick, val", "RouteStick_ss_val.json", "RouteStick_ss_val.json", "RouteStick_val.json"), ("PatternLock, test", "PatternLock_ss_v1.json", "PatternLock_ss_v1_text03.json", "PatternLock_v1.json"),
      ("VideoPlaceButton, test", "VideoPlaceButton_ss_v1.json", "VideoPlaceButton_ss_v1.json", "VideoPlaceButton_v3.json")]
for nm, f, tf, k0f in HH:
    k0, rel, orc, txt = load(k0f, "uniform", 0), load(f, "uniform", 7, same=True), load(f, "lesson_oracle_reindex", 7, same=True), load(tf, "text_route_reindex", 7, same=True)
    out.append(f"{nm} & {cell(k0, rel)} & {cell(k0, orc)} & {cell(rel, txt)} & {cell(orc, txt)}\\\\")
out += ["\\bottomrule", "\\end{tabular}"]
open("paper/ci_table.tex", "w").write("% generated by scripts/make_controls_assets.py\n" + "\n".join(out) + "\n")
print("\n".join(out))

# ---------------------------------------------------------------- table: fine-tuning as absolute success and as the k=0 -> k=7 gap
FT = "results/robomme/ft"
def lf(path, k):
    return {r["episode"]: int(bool(r["success"])) for r in json.load(open(path)) if r.get("outcome") != "error" and r["system"] == "uniform" and r["k"] == k}
f2 = lambda x: f"{x:.2f}".replace("0.", ".")
out = ["\\begin{tabular}{llcccc}", "\\toprule", "task & model & $k{=}0$ & $k{=}7$ & gap $[95\\%]$ & $k{=}7$ vs \\sys{control}\\\\", "\\midrule"]
for task, relf in [("MoveCube", "MoveCube_v1.json"), ("RouteStick", "RouteStick_v1.json"), ("VideoPlaceButton", "VideoPlaceButton_v3.json")]:
    c7 = lf(f"{FT}/control4k_{task}_a.json", 7)
    for m in ["released", "control", "shift", "interf"]:
        k0, k7 = (load(relf, "uniform", 0), load(relf, "uniform", 7)) if m == "released" else (lf(f"{FT}/{m}4k_{task}_a.json", 0), lf(f"{FT}/{m}4k_{task}_a.json", 7))
        d, lo, hi = diff_ci(k7, k0)
        vs = "--" if m in ("released", "control") else cell(c7, k7)
        out.append(f"{task if m == 'released' else ''} & {m if m == 'released' else chr(92) + 'sys{' + m + '}'} & {f2(np.mean(list(k0.values())))} & {f2(np.mean(list(k7.values())))} & ${f2(d)}$ $[{f2(lo)},{f2(hi)}]$ & {vs}\\\\")
    out.append("\\addlinespace")
out = out[:-1] + ["\\bottomrule", "\\end{tabular}"]
open("paper/ft_gap_table.tex", "w").write("% generated by scripts/make_controls_assets.py\n" + "\n".join(out) + "\n")
