"""Method schematic: the buffer, even sampling vs lesson routing, the sampling comb, positional re-indexing, the hard condition.
    python scripts/plot_schematic.py paper/fig/schematic.png"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
fig, axes = plt.subplots(1, 3, figsize=(13, 3.0), gridspec_kw={"width_ratios": [1.25, 1, 1]})
LES, DIS, EPI, SAME = "C2", "0.75", "C0", "C3"
def buffer(ax, y, segs, label, slots=None, slot_color=None, h=0.42):
    x = 0
    for L, col, txt in segs:
        ax.add_patch(Rectangle((x, y), L, h, facecolor=col, edgecolor="k", lw=.6))
        if txt: ax.text(x + L/2, y + h/2, txt, ha="center", va="center", fontsize=7, color="k" if col != EPI else "w")
        x += L
    ax.text(-0.3, y + h/2, label, ha="right", va="center", fontsize=8)
    if slots is not None:
        for s in slots: ax.plot([s, s], [y - 0.14, y - 0.02], color=slot_color if slot_color else "k", lw=1.2)
    return x
# panel A: buffer + allocation
ax = axes[0]; ax.set_title("(a) the buffer and who gets the 32 slots", fontsize=9, loc="left")
D, U, E = 2.4, 0.4, 1.3; k = 7
segs = [(D, LES, "lesson"), *[(U, DIS, "") for _ in range(k)], (E, EPI, "episode")]
total = D + k*U + E
top = buffer(ax, 2.6, segs, "buffer at $k{=}7$")
ev = np.linspace(0.05, total - 0.05, 32); buffer(ax, 1.5, segs, "released: even sampling", slots=ev, slot_color="k")
ax.text(total + 0.2, 1.5 + 0.21, f"lesson share {D/total:.2f}", fontsize=7, va="center")
le = np.concatenate([np.linspace(0.05, D - 0.05, 22), np.linspace(D + k*U + 0.05, total - 0.05, 10)])
buffer(ax, 0.4, segs, "routed: lesson session", slots=le, slot_color="k")
ax.text(total + 0.2, 0.4 + 0.21, "lesson share .69", fontsize=7, va="center")
ax.text(D + k*U/2, 2.6 + 0.6, "$k$ distractor sessions (other tasks, 32 frames each)", ha="center", fontsize=7)
ax.set_xlim(-3.2, total + 2.6); ax.set_ylim(0.1, 3.6); ax.axis("off")
# panel B: comb and positions
ax = axes[1]; ax.set_title("(b) two constraints of the frozen policy", fontsize=9, loc="left")
yA = 2.6; ax.text(0, yA + 0.5, "sampling comb: the episode's own frames must stay evenly spread", fontsize=7)
ax.add_patch(Rectangle((0, yA), 6, 0.35, facecolor=EPI, edgecolor="k", lw=.6))
for s in np.linspace(0.1, 5.9, 10): ax.plot([s, s], [yA - 0.12, yA - 0.02], color="k", lw=1.2)
ax.text(6.15, yA + 0.17, "ok", fontsize=7, va="center", color="C2")
ax.add_patch(Rectangle((0, yA - 0.75), 6, 0.35, facecolor=EPI, edgecolor="k", lw=.6))
for s in np.linspace(4.6, 5.9, 10): ax.plot([s, s], [yA - 0.87, yA - 0.77], color="k", lw=1.2)
ax.text(6.15, yA - 0.58, "recency: stalls (0%)", fontsize=7, va="center", color="C3")
yB = 0.9; ax.text(0, yB + 0.55, "positions: absolute steps shift the episode away from the lesson", fontsize=7)
for i, (lab, lpos, epos, col) in enumerate([("trained layout", (0, 1.8), (1.8, 3.0), "C2"), ("routed, $k{=}7$", (0, 1.8), (4.6, 5.8), "C3"), ("re-indexed", (0, 1.8), (1.8, 3.0), "C2")]):
    y = yB - 0.45*i
    ax.add_patch(Rectangle((lpos[0], y), lpos[1]-lpos[0], 0.3, facecolor=LES, edgecolor="k", lw=.6))
    ax.add_patch(Rectangle((epos[0], y), epos[1]-epos[0], 0.3, facecolor=EPI, edgecolor="k", lw=.6))
    ax.text(6.15, y + 0.15, lab, fontsize=7, va="center", color=col)
    if i == 1: ax.annotate("", xy=(4.6, y + 0.15), xytext=(1.8, y + 0.15), arrowprops=dict(arrowstyle="->", lw=.8)); ax.text(3.2, y + 0.33, "$+32k$ steps, never seen in training", fontsize=6.5, ha="center")
ax.set_xlim(-0.2, 9.2); ax.set_ylim(-0.2, 3.4); ax.axis("off")
# panel C: hard condition
ax = axes[2]; ax.set_title("(c) hard condition: replays recorded in the query scene", fontsize=9, loc="left")
segs = [(D, LES, "lesson"), (U, DIS, ""), (U, SAME, ""), (U, DIS, ""), (U, SAME, ""), (U, DIS, ""), (U, SAME, ""), (U, DIS, ""), (E, EPI, "episode")]
buffer(ax, 2.2, segs, "")
ax.text(D + 3.5*U, 2.2 + 0.65, "3 of 7 distractors: unrelated motions replayed in this scene", ha="center", fontsize=7, color=SAME)
ax.text(0, 1.5, "cosine retrieval picks a same-scene replay on 3 of 4 tasks\n(MoveCube .44, RouteStick .22, PatternLock .14 vs oracle .78 / .70 / .54)", fontsize=7, va="top")
ax.text(0, 0.75, "instruction router: gate sessions by scene, pick by the task text\n(no training; RouteStick .66, PatternLock .50, MoveCube .58)", fontsize=7, va="top", color="C6")
ax.set_xlim(-0.3, total + 0.3); ax.set_ylim(0.0, 3.2); ax.axis("off")
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=170, bbox_inches="tight"); print("saved", sys.argv[1])
