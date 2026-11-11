"""Instruction-conditioned routing by a VLM verifier, zero-shot: for each candidate session, show the VLM a few frames and ask
whether the clip shows the robot doing what the instruction says; route to the session with the highest yes-vs-no logit.
Offline pilot on the cached same-scene data (lesson + same-scene replays + other-task units).

    .venv/bin/python /app/ours/hmrobo/vlm_verify.py --tasks MoveCube RouteStick --out_dir /vol/vlm
"""
from __future__ import annotations
import argparse, glob, os, re, time, random
import numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", nargs="+", required=True)
ap.add_argument("--split", default="train")
ap.add_argument("--root", default="/vol/samescene")
ap.add_argument("--units_dir", default="/vol/units")
ap.add_argument("--out_dir", default="/vol/vlm")
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--n_frames", type=int, default=6)
ap.add_argument("--mode", default="both", help="yesno | compare | both")
ap.add_argument("--n_other", type=int, default=4)
ap.add_argument("--scenes", type=int, default=50)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
from transformers import AutoModelForImageTextToText, AutoProcessor
dev = "cuda"
model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
proc = AutoProcessor.from_pretrained(a.model, min_pixels=64 * 28 * 28, max_pixels=640 * 28 * 28)
tok = proc.tokenizer
YES = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes")]
NO = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No")]
DIG = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(1, 9)]
LM_W = model.get_output_embeddings().weight


def token_logits(inp, ids):
    """Logits of the given token ids at the answer position, computed in float32 from the last hidden state
    (bf16 logits are quantised to ~0.125 and produce ties)."""
    out = model(**inp, output_hidden_states=True)
    h = out.hidden_states[-1][0, -1].float()
    return (LM_W[ids].float() @ h)


def strip(frames, n):
    ims = pick(frames, n); w, hgt = ims[0].size
    canvas = Image.new("RGB", (w * n, hgt))
    for i, im in enumerate(ims): canvas.paste(im, (i * w, 0))
    return canvas
PREFIX = re.compile(r"^\s*watch the video carefully,?\s*then\s*", re.I)


def pick(frames, n):
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [Image.fromarray(np.asarray(frames[i], dtype=np.uint8)) for i in idx]


@torch.no_grad()
def yes_score(frames, instruction) -> float:
    ims = pick(frames, a.n_frames)
    msg = [{"role": "user", "content": [{"type": "image"} for _ in ims] + [{"type": "text", "text":
            f"These {len(ims)} frames are taken in order from one video of a robot arm. In this video, does the robot {instruction}? Answer Yes or No."}]}]
    text = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    inp = proc(text=[text], images=ims, return_tensors="pt").to(dev)
    lg = token_logits(inp, YES + NO)
    return float(torch.logsumexp(lg[:len(YES)], 0) - torch.logsumexp(lg[len(YES):], 0))


@torch.no_grad()
def compare(clips, instruction) -> int:
    """clips: list of frame arrays, shown as one filmstrip image each; -> index of the clip the VLM picks."""
    ims = [strip(c, a.n_frames) for c in clips]
    msg = [{"role": "user", "content": [{"type": "image"} for _ in ims] + [{"type": "text", "text":
            f"Each of these {len(ims)} images is a filmstrip of {a.n_frames} frames, in order, from one video of a robot arm; all videos are recorded in the same scene. "
            f"In exactly one of them the robot does the following: {instruction}. Which video is it? Answer with its number (1-{len(ims)}) only."}]}]
    text = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    inp = proc(text=[text], images=ims, return_tensors="pt").to(dev)
    return int(torch.argmax(token_logits(inp, DIG[:len(ims)])))


units = {}
for up in sorted(glob.glob(f"{a.units_dir}/*.npz")):
    units[os.path.basename(up)[:-4]] = np.load(up)["front"]
t0 = time.time()
for task in a.tasks:
    d = f"{a.root}/{task}/{a.split}"
    lessons = sorted(glob.glob(f"{d}/ep*_lesson.npz"), key=lambda q: int(os.path.basename(q)[2:].split("_")[0]))[: a.scenes]
    pool = [(t, i) for t, arr in units.items() if t != task for i in range(len(arr))]
    rng = random.Random(0); rows = []; hit = {"same": 0, "other": 0, "full": 0}; n = 0
    for lp in lessons:
        ep = int(os.path.basename(lp)[2:].split("_")[0])
        sess = sorted(glob.glob(f"{d}/ep{ep}_[0-9]*.npz"), key=lambda q: int(q[:-4].split("_")[-1]))
        if len(sess) < 2:
            continue
        z = np.load(lp); instr = PREFIX.sub("", str(z["goal"])).rstrip(". ")
        same_frames = [np.load(q)["front"] for q in sess[:3]]; n += 1; row = dict(ep=ep)
        if a.mode in ("yesno", "both"):
            s_les = yes_score(z["front"], instr); s_same = [yes_score(f, instr) for f in same_frames]
            s_oth = [yes_score(units[t][i], instr) for t, i in rng.sample(pool, a.n_other)]
            hit["same"] += int(s_les > max(s_same)); hit["other"] += int(s_les > max(s_oth)); hit["full"] += int(s_les > max(s_same + s_oth))
            row.update(lesson=s_les, same=s_same, other=s_oth)
            print(f"  {task} ep {ep}: yes/no lesson {s_les:+.2f} same {np.round(s_same, 2).tolist()} other {np.round(s_oth, 2).tolist()} | acc same {hit['same']/n:.2f} other {hit['other']/n:.2f} full {hit['full']/n:.2f} ({time.time()-t0:.0f}s)", flush=True)
        if a.mode in ("compare", "both"):
            clips = [z["front"]] + same_frames; order = list(range(len(clips))); rng.shuffle(order)
            pred = compare([clips[o] for o in order], instr); ok = int(order[pred] == 0); hit["compare"] = hit.get("compare", 0) + ok; row.update(compare_ok=ok)
            print(f"  {task} ep {ep}: compare picked {pred+1} (lesson at {order.index(0)+1}) -> {'ok' if ok else 'WRONG'} | acc compare {hit['compare']/n:.2f} ({time.time()-t0:.0f}s)", flush=True)
        rows.append(row)
    np.savez_compressed(f"{a.out_dir}/{task}_{a.split}.npz", rows=np.array(rows, dtype=object))
    print(f"RESULT {task}: n={n} yes/no lesson-vs-3same {hit['same']/max(n,1):.3f} lesson-vs-4other {hit['other']/max(n,1):.3f} full {hit['full']/max(n,1):.3f} | compare(lesson+3same) {hit.get('compare',0)/max(n,1):.3f}", flush=True)
print("done", flush=True)
