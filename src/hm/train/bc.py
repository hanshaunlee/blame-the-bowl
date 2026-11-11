"""Behaviour cloning of the information-matched expert."""
from __future__ import annotations
import argparse, time, json, os, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from .data import generate, save, load, BCDataset, collate, CTX_DIM, PLACE_DIM
from ..models import policy as P


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_dev(b, dev):
    out = {k: v.to(dev, non_blocking=True) for k, v in b.items() if k != "mem"}
    out["mem"] = {k: v.to(dev, non_blocking=True) for k, v in b["mem"].items()}
    return out


def make_policy(d=128, n_layers=3, coord=False, n_heads=4):
    return P.Policy(d=d, n_layers=n_layers, n_heads=n_heads, ctx_dim=CTX_DIM, age_dim=3 + PLACE_DIM, coord=coord)


def load_policy(path, dev):
    ck = torch.load(path, map_location="cpu")
    cfg = ck.get("cfg", {})
    pol = make_policy(**cfg).to(dev)
    pol.load_state_dict(ck["state"]); pol.eval()
    pol.slots = ck.get("B", 8)
    return pol, ck


@torch.no_grad()
def evaluate(pol, dl, dev, max_batches=60):
    pol.eval()
    acc, accf, n = 0.0, 0.0, 0
    for bi, b in enumerate(dl):
        if bi >= max_batches:
            break
        b = to_dev(b, dev)
        logits = pol(b["frames"], b["ctx"], b["mem"], b["mapt"])
        pred = logits.argmax(-1)
        acc += (pred == b["y"]).float().sum().item()
        accf += (pred == b["y_full"]).float().sum().item()
        n += len(pred)
    pol.train()
    return acc / n, accf / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trials_train.pkl")
    ap.add_argument("--val", default="data/trials_val.pkl")
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_val", type=int, default=400)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="ckpt/policy.pt")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--coord", action="store_true")
    ap.add_argument("--init", default="", help="checkpoint to continue from")
    ap.add_argument("--decision_weight", type=float, default=1.0, help="loss weight for non-movement actions")
    ap.add_argument("--extra_data", nargs="*", default=[], help="additional trial pickles (DAgger)")
    a = ap.parse_args()
    dev = device()
    Path("data").mkdir(exist_ok=True); Path(os.path.dirname(a.out) or ".").mkdir(exist_ok=True, parents=True)
    if not os.path.exists(a.data):
        t0 = time.time(); save(generate(a.n_train, 1), a.data); save(generate(a.n_val, 2), a.val)
        print(f"generated data in {time.time()-t0:.0f}s", flush=True)
    tr, va = load(a.data), load(a.val)
    for ex in a.extra_data:
        tr = tr + load(ex)
        print(f"added {ex}: now {len(tr)} trials", flush=True)
    ds, dv = BCDataset(tr, B=a.B, seed=0), BCDataset(va, B=a.B, seed=1)
    print(f"train steps {len(ds)}  val steps {len(dv)}  device {dev}", flush=True)
    dl = DataLoader(ds, a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate, persistent_workers=a.workers > 0, drop_last=True)
    dlv = DataLoader(dv, a.bs, shuffle=True, num_workers=max(a.workers // 2, 1), collate_fn=collate)
    cfg = dict(d=a.d, n_layers=a.layers, coord=a.coord, n_heads=max(4, a.d // 32))
    if a.init:
        pol, ck0 = load_policy(a.init, dev); cfg = ck0.get("cfg", cfg); pol.train()
    else:
        pol = make_policy(**cfg).to(dev)
    print(f"params {sum(p.numel() for p in pol.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, weight_decay=0.01)
    total = a.epochs * len(dl) if not a.max_steps else a.max_steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total, pct_start=0.05)
    step, t0 = 0, time.time()
    log = []
    for ep in range(a.epochs):
        for b in dl:
            b = to_dev(b, dev)
            logits = pol(b["frames"], b["ctx"], b["mem"], b["mapt"])
            if a.decision_weight != 1.0:
                w = torch.where(b["y"] >= 4, torch.full_like(b["y"], a.decision_weight, dtype=torch.float), torch.ones_like(b["y"], dtype=torch.float))
                loss = (F.cross_entropy(logits, b["y"], reduction="none") * w).sum() / w.sum()
            else:
                loss = F.cross_entropy(logits, b["y"])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                print(f"ep {ep} step {step}/{total} loss {loss.item():.3f}  {(time.time()-t0)/step:.2f}s/it", flush=True)
            if step >= total:
                break
        acc, accf = evaluate(pol, dlv, dev)
        log.append(dict(epoch=ep, step=step, val_acc_matched=acc, val_acc_full=accf))
        print(f"== epoch {ep}: val acc (info-matched) {acc:.3f}  agreement with full-history expert {accf:.3f}", flush=True)
        torch.save({"state": pol.state_dict(), "B": a.B, "log": log, "cfg": cfg}, a.out)
        if step >= total:
            break


if __name__ == "__main__":
    main()
