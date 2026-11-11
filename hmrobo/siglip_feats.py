"""SigLIP (HF, joint image-text space) embeddings of the same-scene data and of the distractor units, for
instruction-conditioned routing experiments (zero-shot text-frame matching and a text-conditioned router).

    .venv/bin/python /app/ours/hmrobo/siglip_feats.py --tasks MoveCube RouteStick --split train --out_dir /vol/siglip
Writes <out_dir>/<task>_<split>.npz with eps={ep: dict(lesson (n,D), exec (D,), sessions (m,32,D), goal, text (D,), text_stripped (D,))}
and <out_dir>/units.npz with {task: (n_units, 32, D)} from /vol/units/<task>.npz. Embeddings are L2-normalised float16.
"""
from __future__ import annotations
import argparse, glob, os, re, time
import numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", nargs="+", required=True)
ap.add_argument("--split", default="train")
ap.add_argument("--root", default="/vol/samescene")
ap.add_argument("--units_dir", default="/vol/units")
ap.add_argument("--out_dir", default="/vol/siglip")
ap.add_argument("--model", default="google/siglip-so400m-patch14-384")
ap.add_argument("--batch", type=int, default=64)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
from transformers import AutoModel, AutoProcessor
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
proc = AutoProcessor.from_pretrained(a.model)
PREFIX = re.compile(r"^\s*watch the video carefully,?\s*then\s*", re.I)


def _vec(e):
    """Feature call output -> (B, D) tensor across transformers versions (tensor, or an output object/dict)."""
    if torch.is_tensor(e):
        return e
    keys = list(e.keys()) if hasattr(e, "keys") else []
    for k in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
        v = e[k] if k in keys else getattr(e, k, None)
        if torch.is_tensor(v):
            return v.mean(1) if v.dim() == 3 else v
    for v in (list(e.values()) if hasattr(e, "values") else list(e)):
        if torch.is_tensor(v) and v.dim() == 2:
            return v
    raise TypeError(f"cannot find features in {type(e)} keys={keys}")


@torch.no_grad()
def embed_images(frames: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, len(frames), a.batch):
        ims = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames[i:i + a.batch]]
        px = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev, dtype=model.dtype)
        e = _vec(model.get_image_features(pixel_values=px))
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy().astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0, model.config.projection_dim if hasattr(model.config, "projection_dim") else 1152), np.float16)


@torch.no_grad()
def embed_text(s: str) -> np.ndarray:
    tok = proc(text=[s], padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(dev)
    e = _vec(model.get_text_features(**tok))
    return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()[0].astype(np.float16)


t0 = time.time()
for task in a.tasks:
    d = f"{a.root}/{task}/{a.split}"
    lessons = sorted(glob.glob(f"{d}/ep*_lesson.npz"), key=lambda q: int(os.path.basename(q)[2:].split("_")[0]))
    eps = {}
    for lp in lessons:
        ep = int(os.path.basename(lp)[2:].split("_")[0])
        sess = sorted(glob.glob(f"{d}/ep{ep}_[0-9]*.npz"), key=lambda q: int(q[:-4].split("_")[-1]))
        if len(sess) < 2:
            continue
        z = np.load(lp)
        goal = str(z["goal"])
        eps[ep] = dict(lesson=embed_images(z["front"]), exec=embed_images(z["exec_front"][None])[0],
                       sessions=np.stack([embed_images(np.load(q)["front"]) for q in sess]), goal=goal,
                       text=embed_text(goal), text_stripped=embed_text(PREFIX.sub("", goal)))
        print(f"  {task} ep {ep}: lesson {len(z['front'])} frames, {len(sess)} sessions ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(f"{a.out_dir}/{task}_{a.split}.npz", eps=np.array(eps, dtype=object))
    print(f"{task}: {len(eps)} scenes -> {a.out_dir}/{task}_{a.split}.npz", flush=True)

units = {}
for up in sorted(glob.glob(f"{a.units_dir}/*.npz")):
    t = os.path.basename(up)[:-4]
    fr = np.load(up)["front"]   # (n, 32, H, W, 3)
    units[t] = np.stack([embed_images(u) for u in fr])
    print(f"units {t}: {units[t].shape} ({time.time()-t0:.0f}s)", flush=True)
np.savez_compressed(f"{a.out_dir}/units.npz", **units)
print("done", flush=True)
