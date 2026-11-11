"""Offline instruction-conditioned routing on SigLIP embeddings of the same-scene data (hmrobo/siglip_feats.py output).
Zero-shot scores (scene similarity, text-frame similarity, scene-gated text) and a leave-one-task-out *relational*
router (tiny MLP over per-session scalar relations to the query view and the instruction).
    python scripts/textroute_offline.py <dir with <task>_train.npz and units.npz> [--test MoveCube RouteStick ...]
"""
import argparse, glob, os, random, sys, collections
import numpy as np, torch, torch.nn as nn

ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--test", nargs="*", default=None)
ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--save", default=""); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n_other", type=int, default=4)
a = ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed); np.random.seed(a.seed)

data = {os.path.basename(p)[:-len("_train.npz")]: np.load(p, allow_pickle=True)["eps"].item() for p in sorted(glob.glob(f"{a.dir}/*_train.npz"))}
units = dict(np.load(f"{a.dir}/units.npz")) if os.path.exists(f"{a.dir}/units.npz") else {}
print("tasks:", {t: len(v) for t, v in data.items()}, "| unit pools:", {t: v.shape[0] for t, v in units.items()})
f32 = lambda x: np.asarray(x, dtype=np.float32)


def relations(F, q, t):
    """Scalar relations between a session (frames F (f,D), unit-norm) and the query view q / instruction t."""
    sq, st = F @ q, F @ t
    d = F[-1] - F[0]; nd = np.linalg.norm(d) + 1e-6
    half = len(F) // 2
    return np.array([sq.max(), sq.mean(), sq[:half].mean(), sq[half:].mean(), st.max(), st.mean(), st[:half].mean(), st[half:].mean(),
                     st[half:].mean() - st[:half].mean(), nd, float(d @ t) / nd, float(d @ q) / nd, float(np.std(sq)), float(np.std(st))], np.float32)


def episodes(task, n_same=3, n_other=4, rng=None, split="all"):
    """-> list of (query q, text t, sessions [F...], label). Queries: exec view + frames of the held-out last same-scene session."""
    rng = rng or random.Random(0); out = []
    pool = [u for t2, arr in units.items() if t2 != task for u in arr]
    for ep, e in sorted(data[task].items()):
        sess = list(e["sessions"]); hold = sess[-1]; negs = sess[:-1]
        t = f32(e["text_stripped"]); qs = [f32(e["exec"])] + [f32(x) for x in hold[::8]]
        for q in qs:
            same = negs[:]; rng.shuffle(same); same = same[:n_same]
            others = rng.sample(pool, n_other) if n_other and len(pool) >= n_other else []
            S = [f32(e["lesson"])] + [f32(x) for x in same] + [f32(x) for x in others]
            order = list(range(len(S))); rng.shuffle(order)
            out.append((q, t, [S[o] for o in order], order.index(0), ep))
    return out


def zero_shot(task):
    for name, (ns, no) in {"lesson+3same": (3, 0), "lesson+4other": (0, a.n_other), "lesson+3same+4other": (3, a.n_other)}.items():
        eps = episodes(task, ns, no); acc = collections.Counter(); n = 0
        for q, t, S, y, _ in eps:
            scene = np.array([(F @ q).max() for F in S]); text = np.array([(F @ t).max() for F in S])
            st = [F @ t for F in S]; sq = [F @ q for F in S]
            prog = np.array([x[len(x)//2:].mean() - x[:len(x)//2].mean() for x in st])          # instruction similarity rises over the session
            prog_max = np.array([x[len(x)//4:].max() - x[:max(1, len(x)//4)].mean() for x in st])
            change = np.array([1 - x[-max(1, len(x)//4):].mean() for x in sq])                  # scene looks different from the reset view at the end
            top3 = np.array([np.sort(x)[-3:].mean() for x in st])
            gate = scene >= scene.max() - 0.05
            acc["scene(cosine)"] += int(scene.argmax() == y); acc["text max"] += int(text.argmax() == y); acc["text top3"] += int(top3.argmax() == y)
            acc["gate+text"] += int(np.where(gate, text, -9).argmax() == y)
            acc["text progress"] += int(prog.argmax() == y); acc["text progress-max"] += int(prog_max.argmax() == y)
            acc["change"] += int(change.argmax() == y)
            acc["gate+text+prog"] += int(np.where(gate, text + prog_max, -9).argmax() == y)
            acc["gate+text+change"] += int(np.where(gate, text + change, -9).argmax() == y)
            acc["gate+text+prog+change"] += int(np.where(gate, text + prog_max + change, -9).argmax() == int(y))
            n += 1
        print(f"  zero-shot {task:16s} {name:20s} " + "  ".join(f"{k} {v/n:.3f}" for k, v in acc.items()))


def relative(R):
    """Scale-free, per-candidate-set features: each session's relations minus the set mean / max, and ranks. (S, 3d)."""
    R = np.asarray(R, np.float32); sd = R.std(0) + 1e-6
    ranks = np.argsort(np.argsort(R, 0), 0) / max(1, len(R) - 1)
    return np.concatenate([(R - R.mean(0)) / sd, (R - R.max(0)) / sd, ranks], 1).astype(np.float32)


class RelRouter(nn.Module):
    def __init__(self, d=42, h=64):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, R):   # R (S, d) -> (S,)
        return self.net(R).squeeze(-1)


def train_loto(test_task):
    train_tasks = [t for t in data if t != test_task]
    rng = random.Random(a.seed)
    train = [ex for t in train_tasks for ex in episodes(t, 3, a.n_other, rng)] + [ex for t in train_tasks for ex in episodes(t, 3, 0, rng)] + [ex for t in train_tasks for ex in episodes(t, 1, a.n_other, rng)]
    rng.shuffle(train)
    feats = lambda q, t, S: torch.from_numpy(relative(np.stack([relations(F, q, t) for F in S])))
    m = RelRouter(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    for step in range(a.steps):
        batch = rng.sample(train, 32); loss = 0
        for q, t, S, y, _ in batch:
            loss = loss + nn.functional.cross_entropy(m(feats(q, t, S))[None], torch.tensor([y]))
        opt.zero_grad(); (loss / len(batch)).backward(); opt.step()
    m.eval(); res = {}
    with torch.no_grad():
        for name, (ns, no) in {"lesson+3same": (3, 0), "lesson+4other": (0, a.n_other), "lesson+3same+4other": (3, a.n_other)}.items():
            eps = episodes(test_task, ns, no); hit = sum(int(m(feats(q, t, S)).argmax() == y) for q, t, S, y, _ in eps); res[name] = hit / len(eps)
        # exec-view query only (the decision a per-episode router makes once at the start)
        eps = [e for e in episodes(test_task, 3, a.n_other)]; first = {}
        for q, t, S, y, ep in eps: first.setdefault(ep, (q, t, S, y))
        res["exec-query only"] = np.mean([int(m(feats(q, t, S)).argmax() == y) for q, t, S, y in first.values()])
    print(f"  relational router (relative feats), trained on {len(train_tasks)} tasks, tested on {test_task}: " + "  ".join(f"{k} {v:.3f}" for k, v in res.items()), flush=True)
    if a.save:
        torch.save({"state": m.state_dict(), "test_task": test_task, "train_tasks": train_tasks}, f"{a.save}/relrouter_not_{test_task}.pt")
    return m


tests = a.test or list(data)
for t in tests:
    zero_shot(t)
if not os.environ.get("ZERO_SHOT_ONLY"):
    for t in tests:
        train_loto(t)
