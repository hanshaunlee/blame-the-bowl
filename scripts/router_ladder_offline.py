"""Routing ladder from the instruction router's logged decisions (free; no new rollouts).
Every text_route call logs, per session, the SigLIP scene score (max cosine to the current view), the gate, and the
instruction score. Session 0 is the lesson (eval_cross.py does not shuffle sessions; the router never reads the order).
From the same numbers: scene-only argmax (same encoder, same once-per-episode commitment, no language), instruction-only
argmax (no gate), and the full router.   python scripts/router_ladder_offline.py
"""
import re, glob, json, numpy as np
RUNS = {  # serve log -> cell (results/robomme/serve_logs, copied from the volume)
    "RouteStick_1794296686.log": "RouteStick hard test", "RouteStick_1794302055.log": "RouteStick hard val",
    "PatternLock_1794298281.log": "PatternLock hard test", "MoveCube_1794296847.log": "MoveCube hard test",
    "MoveCube_1794340618.log": "MoveCube hard val", "VideoPlaceButton_1794342187.log": "VideoPlaceButton hard test",
    "RouteStick_1794296526.log": "RouteStick standard k=7", "PatternLock_1794296529.log": "PatternLock standard k=7",
    "MoveCube_1794297725.log": "MoveCube standard k=7,15", "VideoPlaceButton_1794297460.log": "VideoPlaceButton standard k=7",
    "VideoRepick_1794296566.log": "VideoRepick standard k=7"}
print(f"{'cell':32s} n   scene-only  instr-only  gate+instr  gate size  lesson in gate")
for f, name in RUNS.items():
    rows = []
    for l in open(f"results/robomme/serve_logs/{f}", errors="ignore"):
        m = re.search(r"text_route\(m=[\d.]+\).*scene (\[.*?\]) gate (\[.*?\]) text (\[.*?\]) -> session (\d+)", l)
        if m:
            rows.append((np.array(json.loads(m.group(1))), np.array(json.loads(m.group(2))), np.array(json.loads(m.group(3))), int(m.group(4))))
    acc = lambda g: np.mean([g(r) == 0 for r in rows])
    print(f"{name:32s} {len(rows):3d}  {acc(lambda r: np.argmax(r[0])):.2f}        {acc(lambda r: np.argmax(r[2])):.2f}        {acc(lambda r: r[3]):.2f}        "
          f"{np.mean([r[1].sum() for r in rows]):.1f}        {np.mean([r[1][0] for r in rows]):.2f}")
