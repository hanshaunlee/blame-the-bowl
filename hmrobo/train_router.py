"""Train the session router on the train split (session labels available there), report offline
routing accuracy of the router and of cosine retrieval on held-out train episodes, for same-task and
other-task distractors. Features come from the frozen VLA's encoder (cached on the volume).

    .venv/bin/python /app/ours/hmrobo/train_router.py --ckpt <dir> --task VideoPlaceButton --out /vol/router_VideoPlaceButton.pt
"""
from __future__ import annotations
import argparse, os, sys, time, random, json
from pathlib import Path
import numpy as np, torch, h5py

sys.path.insert(0, "/app/ours")
from mme_vla_suite.policies import policy_config as _policy_config  # noqa: E402
from mme_vla_suite.training import config as _config  # noqa: E402
from hmrobo.buffer import load_episode, episode_keys, load_units  # noqa: E402
from hmrobo.mm_policy import encode_frames, _feat  # noqa: E402
from hmrobo.router import SessionRouter, cosine_route  # noqa: E402


def pooled(pol, front, state):
    return np.stack([_feat(f) for f in encode_frames(pol, front, state)]).astype(np.float16)


def build_features(pol, a):
    """-> dict(eps={key: dict(lesson=(n,2048), exec=(m,2048))}, units=[dict(task,key,feat=(32,2048))]); cached."""
    cache = f"/vol/router_feats_{a.task}.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["eps"].item(), list(z["units"])
    path = a.task_h5 or f"{a.data_dir}/record_dataset_{a.task}.h5"
    with h5py.File(path, "r") as f:
        keys = episode_keys(f)[: a.n_eps]
    eps = {}
    t0 = time.time()
    for key in keys:
        ep = load_episode(path, key, a.task)
        e0 = ep.exec_start
        ex = np.arange(e0, len(ep.front), a.exec_stride)
        eps[key] = dict(lesson=pooled(pol, ep.front[:e0], ep.state[:e0]), exec=pooled(pol, ep.front[ex], ep.state[ex]))
        print(f"  {key}: lesson {e0} exec {len(ex)} frames ({time.time()-t0:.0f}s)", flush=True)
    raw = load_units(a.data_dir, a.task, pool="all")
    if a.task_h5 and os.path.dirname(a.task_h5) != a.data_dir.rstrip("/"):   # task h5 kept outside the eval distractor pool
        raw += load_units(os.path.dirname(a.task_h5), a.task, pool="same")
    units = [dict(task=u["task"], key=u["key"], feat=pooled(pol, u["front"], u["state"])) for u in raw]
    print(f"{len(eps)} episodes, {len(units)} units encoded in {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(cache, eps=np.array(eps, dtype=object), units=np.array(units, dtype=object))
    return eps, units


def ss_task(d):
    """/vol/samescene/<task>/<split> -> task."""
    return os.path.basename(os.path.dirname(d.rstrip("/")))


def build_ss_features(pol, a):
    """All same-scene dirs (--same_scene_dir and --ss_dirs) -> {(task, ep): ...}; see build_ss_features_dir."""
    out = {}
    for d in ([a.same_scene_dir] if a.same_scene_dir else []) + list(a.ss_dirs or []):
        t = ss_task(d)
        for ep, v in build_ss_features_dir(pol, d, t).items():
            out[(t, ep)] = dict(v, task=t)
    return out


def build_ss_features_dir(pol, d, task):
    """Same-scene data generated in sim (hmrobo/gen_same_scene.py): -> {ep: dict(lesson=(n,2048), queries=(q,2048), same=[(32,2048)...])}.
    queries = the execution-start view + frames of the last same-scene session (views of this scene with the robot moving),
    which is held out from the negatives. Cached."""
    import glob
    cache = f"/vol/router_feats_ss_{task}_{os.path.basename(d.rstrip('/'))}.npz"
    if os.path.exists(cache):
        cached = np.load(cache, allow_pickle=True)["eps"].item()
        if cached:
            print(f"same-scene features from cache {cache}: {len(cached)} episodes", flush=True)
            return cached
        print(f"ignoring empty same-scene cache {cache}", flush=True)   # written by a run that could not see the data
    out = {}; t0 = time.time()
    lessons = sorted(glob.glob(f"{d}/ep*_lesson.npz"), key=lambda q: int(os.path.basename(q)[2:].split("_")[0]))
    print(f"same-scene dir {d}: exists={os.path.isdir(d)} files={len(os.listdir(d)) if os.path.isdir(d) else 0} lessons={len(lessons)}", flush=True)
    if not lessons:
        raise SystemExit(f"--same_scene_dir {d} given but no ep*_lesson.npz found")
    for lp in lessons:
        ep = int(os.path.basename(lp)[2:].split("_")[0])
        z = np.load(lp)
        sess = sorted(glob.glob(f"{d}/ep{ep}_[0-9]*.npz"), key=lambda q: int(q[:-4].split("_")[-1]))
        if len(sess) < 2:
            continue
        feats = [pooled(pol, np.load(q)["front"], np.load(q)["state"]) for q in sess]
        queries = np.concatenate([pooled(pol, z["exec_front"][None], z["exec_state"][None]), feats[-1]])
        out[ep] = dict(lesson=pooled(pol, z["front"], z["state"]), queries=queries, same=feats[:-1], fingerprint=int(z["fingerprint"]))
        print(f"  ss ep {ep}: lesson {len(z['front'])} frames, {len(feats)-1} same-scene sessions ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(cache, eps=np.array(out, dtype=object))
    return out


def make_example_ss(rng, ss, units, ep, k, n_same, a):
    """Hard-routing example: lesson + n_same same-scene sessions + other-task units up to k."""
    e = ss[ep]
    q = e["queries"][rng.randrange(len(e["queries"]))].astype(np.float32)
    same = list(e["same"]); rng.shuffle(same); same = same[:n_same]
    other = [u for u in units if u["task"] != e.get("task", a.task)]
    ds = rng.sample(other, max(0, k - len(same)))
    sessions = [e["lesson"].astype(np.float32)] + [x.astype(np.float32) for x in same] + [d["feat"].astype(np.float32) for d in ds]
    order = list(range(len(sessions))); rng.shuffle(order)
    return q, [sessions[o] for o in order], order.index(0)


def accuracy_ss(router, rng, ss, units, eps_, k, n_same, a, n=200, dev="cpu"):
    hit_r = hit_c = 0
    for _ in range(n):
        q, sess, y = make_example_ss(rng, ss, units, rng.choice(eps_), k, n_same, a)
        F = np.concatenate([q[None]] + sess); bounds, off = [], 1
        for s_ in sess:
            bounds.append((off, off + len(s_))); off += len(s_)
        hit_c += cosine_route(F, bounds, 0) == y
        with torch.no_grad():
            logits = router(torch.from_numpy(q).to(dev), [torch.from_numpy(s_).to(dev) for s_ in sess])
        hit_r += int(torch.argmax(logits).item()) == y
    return hit_r / n, hit_c / n


def make_example(rng, eps, units, key, k, same_only, a):
    """-> (query (2048,), sessions list of (n,2048) float32, label)."""
    e = eps[key]
    q = e["exec"][rng.randrange(len(e["exec"]))].astype(np.float32)
    same = [u for u in units if u["task"] == a.task and u["key"] != key]
    other = [u for u in units if u["task"] != a.task]
    pool = same if same_only else same + other
    ds = rng.sample(pool, k) if k <= len(pool) else [rng.choice(pool) for _ in range(k)]
    sessions = [e["lesson"].astype(np.float32)] + [d["feat"].astype(np.float32) for d in ds]
    order = list(range(len(sessions))); rng.shuffle(order)
    return q, [sessions[o] for o in order], order.index(0)


def accuracy(router, rng, eps, units, keys, k, same_only, a, n=200, dev="cpu"):
    hit_r = hit_c = 0
    for _ in range(n):
        q, sess, y = make_example(rng, eps, units, rng.choice(keys), k, same_only, a)
        F = np.concatenate([q[None]] + sess)   # cosine_route wants a frame matrix: put query at row 0, sessions after
        bounds, off = [], 1
        for s in sess:
            bounds.append((off, off + len(s))); off += len(s)
        hit_c += cosine_route(F, bounds, 0) == y
        with torch.no_grad():
            logits = router(torch.from_numpy(q).to(dev), [torch.from_numpy(s).to(dev) for s in sess])
        hit_r += int(torch.argmax(logits).item()) == y
    return hit_r / n, hit_c / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="mme_vla_suite")
    ap.add_argument("--task", default="VideoPlaceButton")
    ap.add_argument("--data_dir", default="/vol/data")
    ap.add_argument("--task_h5", default="", help="path of the task's train h5 if it is not in --data_dir (keeps the eval distractor pool unchanged)")
    ap.add_argument("--n_eps", type=int, default=60)
    ap.add_argument("--n_val", type=int, default=12)
    ap.add_argument("--exec_stride", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--p_same", type=float, default=0.5)
    ap.add_argument("--out", default="/vol/router.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--same_scene_dir", default="", help="/vol/samescene/<task>/train: adds hard-routing examples (same-scene negatives)")
    ap.add_argument("--p_ss", type=float, default=0.5, help="fraction of training examples drawn from the same-scene data")
    ap.add_argument("--exclude_tasks", nargs="*", default=[], help="tasks whose distractor units are removed from the negative pool (held-out task of a leave-one-task-out router)")
    ap.add_argument("--ss_dirs", nargs="*", default=[], help="additional same-scene dirs of OTHER tasks (/vol/samescene/<task>/train): multi-task / leave-one-task-out router")
    ap.add_argument("--n_val_ss", type=int, default=10)
    a = ap.parse_args()
    rng = random.Random(a.seed); torch.manual_seed(a.seed)
    pol = _policy_config.create_trained_policy(_config.get_config(a.config), Path(a.ckpt), seed=7)
    eps, units = build_features(pol, a)
    if a.exclude_tasks:   # leave-one-task-out: the held-out task must not appear as a negative either
        n0 = len(units); units = [u for u in units if u["task"] not in a.exclude_tasks]
        print(f"excluded units of {a.exclude_tasks}: {n0} -> {len(units)}", flush=True)
    ss = build_ss_features(pol, a)
    del pol
    ss_val, ss_train = [], []
    for t in sorted({k[0] for k in ss}):   # hold out the last n_val_ss scenes of every task
        eps_t = sorted(k for k in ss if k[0] == t); ss_val += eps_t[-a.n_val_ss:]; ss_train += eps_t[:-a.n_val_ss]
    if ss:
        print(f"same-scene episodes: train {len(ss_train)} val {len(ss_val)} from tasks {sorted({k[0] for k in ss})}", flush=True)
    keys = sorted(eps, key=lambda s: int(s.split("_")[1]))
    val, train = keys[-a.n_val:], keys[:-a.n_val]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    router = SessionRouter().to(dev)
    opt = torch.optim.AdamW(router.parameters(), lr=a.lr, weight_decay=1e-2)
    print(f"train {len(train)} val {len(val)} episodes; same-task units {sum(u['task']==a.task for u in units)}, other {sum(u['task']!=a.task for u in units)}; device {dev}", flush=True)
    t0 = time.time(); log = []
    for step in range(1, a.steps + 1):
        router.train(); loss = 0.0
        for _ in range(a.batch):
            k = rng.choice([1, 3, 7]); same_only = rng.random() < a.p_same
            if ss and rng.random() < a.p_ss:
                q, sess, y = make_example_ss(rng, ss, units, rng.choice(ss_train), k, rng.choice([1, 2, 3]) if k > 1 else 1, a)
            else:
                q, sess, y = make_example(rng, eps, units, rng.choice(train), k, same_only, a)
            logits = router(torch.from_numpy(q).to(dev), [torch.from_numpy(s).to(dev) for s in sess])
            loss = loss + torch.nn.functional.cross_entropy(logits[None], torch.tensor([y], device=dev))
        loss = loss / a.batch
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0); opt.step()
        if step % 100 == 0 or step == 1:
            router.eval(); vr = random.Random(123)
            accs = {f"{'same' if so else 'all'}_k{k}": accuracy(router, vr, eps, units, val, k, so, a, n=100, dev=dev) for k in (3, 7) for so in (True, False)}
            if ss:
                accs["ss_k7_3same"] = accuracy_ss(router, vr, ss, units, ss_val, 7, 3, a, n=100, dev=dev)
            msg = " ".join(f"{n} router {r:.2f}/cos {c:.2f}" for n, (r, c) in accs.items())
            print(f"step {step} loss {loss.item():.3f} | val {msg} | {time.time()-t0:.0f}s", flush=True)
            log.append(dict(step=step, loss=loss.item(), **{n: dict(router=r, cosine=c) for n, (r, c) in accs.items()}))
            torch.save({"state": {k: v.cpu() for k, v in router.state_dict().items()}, "cfg": {}, "log": log, "task": a.task, "args": vars(a)}, a.out)
    router.eval(); vr = random.Random(7)
    final = {f"{'same' if so else 'all'}_k{k}": accuracy(router, vr, eps, units, val, k, so, a, n=400, dev=dev) for k in (1, 3, 7) for so in (True, False)}
    if ss:
        for ns in (1, 3):
            final[f"ss_k7_{ns}same"] = accuracy_ss(router, vr, ss, units, ss_val, 7, ns, a, n=400, dev=dev)
    for n, (r, c) in final.items():
        print(f"FINAL {n}: router {r:.3f} cosine {c:.3f}", flush=True)
    torch.save({"state": {k: v.cpu() for k, v in router.state_dict().items()}, "cfg": {}, "log": log, "final": final, "task": a.task, "ss_tasks": sorted({k[0] for k in ss}), "args": vars(a)}, a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
