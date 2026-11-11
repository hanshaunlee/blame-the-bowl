"""Watch fine-tuning runs on Modal: pull /vol/ft_logs/<exp>.log, parse "Step N: loss=..." lines, print a compact
loss trace, and (with --kill-if) cancel the FunctionCall when a rule fires. Rules are checked on the latest window.

    python3 scripts/ft_monitor.py --exp ft-interf-v1 --call fc-... [--every 300] [--kill-if "nan|loss>0.02@200|flat@1500"]

nan             loss is NaN/inf
loss>X@N        smoothed loss above X after step N (diverging / not learning)
flat@N          after step N the smoothed loss of the last 200 steps is not below the smoothed loss 500 steps earlier
"""
from __future__ import annotations
import argparse, re, subprocess, time, math, json, os
import numpy as np

def pull(exp: str, dst: str) -> str:
    subprocess.run(["modal", "volume", "get", "--force", "robomme", f"ft_logs/{exp}.log", dst], capture_output=True)
    return open(dst).read() if os.path.exists(dst) else ""

def parse(text: str):
    steps, losses = [], []
    for m in re.finditer(r"Step (\d+): (.*)", text):
        kv = dict(re.findall(r"(\w+)=([-\d.einfa]+)", m.group(2)))
        if "loss" in kv:
            steps.append(int(m.group(1))); losses.append(float(kv["loss"]))
    return np.array(steps), np.array(losses)

def smooth(steps, losses, lo, hi):
    m = (steps >= lo) & (steps < hi)
    return float(np.mean(losses[m])) if m.any() else math.nan

def check(rules, steps, losses):
    if len(steps) == 0: return None
    last = int(steps[-1])
    for r in rules:
        if r == "nan" and (not np.isfinite(losses[-5:]).all()): return "nan loss"
        m = re.match(r"loss>([\d.e-]+)@(\d+)", r)
        if m and last >= int(m.group(2)) and smooth(steps, losses, last - 200, last + 1) > float(m.group(1)): return f"loss above {m.group(1)} at step {last}"
        m = re.match(r"flat@(\d+)", r)
        if m and last >= int(m.group(1)):
            now, before = smooth(steps, losses, last - 200, last + 1), smooth(steps, losses, last - 700, last - 500)
            if np.isfinite(now) and np.isfinite(before) and now >= before: return f"flat: {now:.5f} vs {before:.5f} 500 steps earlier"
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True); ap.add_argument("--call", default=""); ap.add_argument("--every", type=int, default=300)
    ap.add_argument("--kill-if", default="nan"); ap.add_argument("--once", action="store_true"); ap.add_argument("--tmp", default="/tmp")
    a = ap.parse_args()
    rules = a.kill_if.split("|") if a.kill_if else []
    while True:
        text = pull(a.exp, f"{a.tmp}/{a.exp}.log"); steps, losses = parse(text)
        tail = "\n".join(text.strip().splitlines()[-3:])
        if len(steps):
            marks = [int(x) for x in np.linspace(0, len(steps) - 1, min(12, len(steps)))]
            print(time.strftime("%H:%M"), a.exp, "steps", int(steps[-1]), "trace", " ".join(f"{int(steps[i])}:{losses[i]:.4f}" for i in marks), flush=True)
        else:
            print(time.strftime("%H:%M"), a.exp, "no loss lines yet |", tail.replace("\n", " | ")[-300:], flush=True)
        why = check(rules, steps, losses)
        if why and a.call:
            import modal
            print("KILL:", why, flush=True); modal.FunctionCall.from_id(a.call).cancel(); print("cancelled", a.call, flush=True); break
        if a.once or "train_ft exit" in text or "DONE" in text or "Traceback" in text or "Error" in text.splitlines()[-1:][0] if text else False:
            print(tail, flush=True); break
        time.sleep(a.every)

if __name__ == "__main__":
    main()
