"""Qualitative figure: what a router sees. Test episode 3 of RouteStick and MoveCube: the current view, the lesson, an unrelated
session (standard protocol), a replay of another task recorded in the query scene (hard condition) and a replay of the same task
recorded in the query scene. Frames are copied from the caches the evaluator reads (/vol/samescene, /vol/samescene_st, /vol/units)
into results/robomme/qual_frames.npz.
    python scripts/plot_qualitative.py     # writes paper/fig/qualitative.png"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
Z = np.load("results/robomme/qual_frames.npz")
ROWS = [("lesson", "lesson", "C2"), ("unrelated", "unrelated session\n(standard protocol)", "0.45"), ("other", "other task, replayed\nin the query scene", "C1"), ("same", "same task, replayed\nin the query scene", "C3")]
N = 6
W, H = 13.5, 4.15                     # inches; cells are square, placed by hand
c = .88 / W * 1.0                     # cell width in figure fraction
ch = c * W / H
x0s = [.125, .125 + 7 * c + .012]
fig = plt.figure(figsize=(W, H))
def cell(x, y, img, col, lw=1.5):
    a = fig.add_axes([x, y, c * .97, ch * .97]); a.imshow(img); a.set_xticks([]); a.set_yticks([])
    for s_ in a.spines.values(): s_.set_color(col); s_.set_linewidth(lw)
    return a
for b, task in enumerate(["RouteStick", "MoveCube"]):
    for r, (key, name, col) in enumerate(ROWS):
        y = .93 - (r + 1) * ch * 1.03
        if b == 0: fig.text(x0s[0] - .008, y + ch / 2, name, ha="right", va="center", fontsize=8.5, color=col, fontweight="bold")
        if r == 0: cell(x0s[b], y, Z[f"{task}_current"], "k", 2.2).set_title("current view", fontsize=8, pad=2)
        for j in range(N): cell(x0s[b] + (j + 1) * c, y, Z[f"{task}_{key}"][j], col)
    goal = str(Z[f"{task}_goal"]).split("then ")[-1].replace("use the stick attached to the robot to ", "")
    fig.text(x0s[b] + c + 3 * c, .975, f"{task}: \u201c{goal}\u201d", ha="center", va="top", fontsize=8.5)
plt.savefig("paper/fig/qualitative.png", dpi=170, bbox_inches="tight")
