"""Cross-session (RoboMME-Interference) evaluation client for the memory-managed policy server.

Runs inside the simulator venv against hmrobo/serve_mm.py. Mirrors examples/robomme/eval.py (same
EnvRunner, 16-step chunks, incremental add_buffer, the current observation is in the buffer when
the policy is queried) and builds the buffer as

    [lesson video][distractor session 1]...[distractor session k][this episode, streaming]

with true positions. Every memory system fills the policy's fixed 32-frame budget; the server picks
the frames (see hmrobo/mm_policy.py): uniform (released default) | recency | exec_only |
lesson_oracle | retrieval | learned.

    python hmrobo/eval_cross.py --task VideoRepick --k 0 3 7 --systems uniform retrieval learned \
        --episodes 20 --port 8011 --out /vol/eval_cross/VideoRepick.json
"""
from __future__ import annotations
import argparse, collections, json, os, sys, time, random, traceback
from collections import deque
import numpy as np

sys.path.insert(0, "/app/robomme_policy_learning/examples/robomme")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_runner import EnvRunner  # noqa: E402  (official helper)
from openpi_client import websocket_client_policy as wcp  # noqa: E402
from hmrobo.buffer import load_units, load_actions, episode_keys  # noqa: E402
import glob, h5py  # noqa: E402

CHUNK = 64


def add_frames(client, fronts, states, exec_start_idx=0):
    for s in range(0, len(fronts), CHUNK):
        client.infer({"images": np.stack(fronts[s:s + CHUNK]).astype(np.uint8)[:, None], "state": np.stack(states[s:s + CHUNK]).astype(np.float32),
                      "add_buffer": True, "exec_start_idx": exec_start_idx if s == 0 else 0})


def scene_fingerprint(img):
    return int(np.asarray(img, dtype=np.int64)[::8, ::8].sum())


def same_scene_session(runner, episode_id, task, split, j, data_dir, cache_dir, n_src=256, stride=8):
    """A 32-frame session recorded in the *query episode's own scene*: the robot replays an unrelated motion
    (the post-demonstration joint actions of a recorded episode of another task) in this scene, after the lesson
    video has played. Cached per (task, split, episode, j). Returns dict(front, state, fingerprint, src)."""
    path = f"{cache_dir}/{task}/{split}/ep{episode_id}_{j}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return dict(front=list(z["front"]), state=list(z["state"]), fingerprint=int(z["fingerprint"]), src=str(z["src"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    srcs = [q for q in sorted(glob.glob(f"{data_dir}/record_dataset_*.h5")) if os.path.basename(q)[len("record_dataset_"):-3] != task]
    rng = random.Random(7919 * episode_id + j)
    src = rng.choice(srcs)
    with h5py.File(src, "r") as f:
        key = rng.choice(episode_keys(f)[:12])
    acts = load_actions(src, key, n=n_src)
    runner.make_env(episode_id)
    pre = runner.get_init_obs()
    fp = scene_fingerprint(pre["images"][0])
    fronts, states = [pre["images"][-1].copy()], [pre["states"][-1].copy()]
    for i, a in enumerate(acts):
        (img, wimg, st), stop, outcome = runner.step(a)
        if img is None:
            break
        if (i + 1) % stride == 0:
            fronts.append(img.copy()); states.append(st.copy())
        if stop:
            break
    runner.close_env()
    while len(fronts) < n_src // stride:
        fronts.append(fronts[-1]); states.append(states[-1])
    fronts, states = fronts[:n_src // stride], states[:n_src // stride]
    np.savez_compressed(path, front=np.stack(fronts), state=np.stack(states), fingerprint=fp, src=f"{os.path.basename(src)}:{key}")
    print(f"  same-scene session cached: ep={episode_id} j={j} src={os.path.basename(src)}:{key} frames={len(fronts)}", flush=True)
    return dict(front=fronts, state=states, fingerprint=fp, src=f"{os.path.basename(src)}:{key}")


def run_episode(client, runner, episode_id, k, system, units, rng, max_steps, shuffle_sessions=False, obs_horizon=16, same_scene=0, same_scene_kw=None):
    extra = []
    if k > 0 and same_scene > 0:
        extra = [same_scene_session(runner, episode_id, j=j, **same_scene_kw) for j in range(min(same_scene, k))]
    client.reset()
    runner.make_env(episode_id)
    pre = runner.get_init_obs()
    goal = pre["task_goal"]
    scene_match = all(scene_fingerprint(pre["images"][0]) == e["fingerprint"] for e in extra) if extra else None
    video_f, video_s = list(pre["images"][:-1]), list(pre["states"][:-1])   # the last pre frame is the first execution observation
    img, wimg, st = pre["images"][-1], pre["wrist_images"][-1], pre["states"][-1]
    k_other = k - len(extra)
    ds = rng.sample(units, k_other) if k_other <= len(units) else [rng.choice(units) for _ in range(k_other)]
    dist = [(list(d["front"]), list(d["state"]), False) for d in ds] + [(e["front"], e["state"], True) for e in extra]
    if extra:
        rng.shuffle(dist)   # same-scene sessions at random positions among the distractors
    sessions = [(video_f, video_s, False)] + dist
    order = list(range(len(sessions)))
    if shuffle_sessions:
        rng.shuffle(order)
    hf, hs, bounds, lesson, same_b = [], [], [], (0, 0), []
    for o in order:
        a_ = len(hf); hf += sessions[o][0]; hs += sessions[o][1]; bounds.append([a_, len(hf)])
        if o == 0:
            lesson = [a_, len(hf)]
        if sessions[o][2]:
            same_b.append((a_, len(hf)))
    H = len(hf)
    if H:
        add_frames(client, hf, hs, exec_start_idx=H)
    extra = {"memory_system": system, "phase_start": H, "lesson_bounds": lesson, "session_bounds": bounds}
    new_f, new_s = [img.copy()], [st.copy()]
    plan = deque(); count = 0; outcome = "timeout"; lf, ef, sf, n_inf = [], [], [], 0
    while True:
        if not plan:
            add_frames(client, new_f, new_s); new_f, new_s = [], []
            out = client.infer({"observation/image": img, "observation/wrist_image": wimg, "observation/state": st, "prompt": goal, **extra})
            plan.extend(list(out["actions"][:obs_horizon])); n_inf += 1
            mi = np.asarray(out.get("memory_indices", []))
            if len(mi):
                lf.append(float(np.mean((mi >= lesson[0]) & (mi < lesson[1])))); ef.append(float(np.mean(mi >= H)))
                sf.append(float(np.mean(np.any([(mi >= a_) & (mi < b_) for a_, b_ in same_b], axis=0))) if same_b else 0.0)
        a = plan.popleft()
        (img, wimg, st), stop, outcome = runner.step(a)
        count += 1
        if stop or count > max_steps or img is None:
            break
        new_f.append(img.copy()); new_s.append(st.copy())
    runner.close_env()
    return dict(outcome=outcome, success=(outcome == "success"), steps=count, lesson_frac=float(np.mean(lf)) if lf else float("nan"),
                exec_frac=float(np.mean(ef)) if ef else float("nan"), same_frac=float(np.mean(sf)) if sf else float("nan"), n_same=len(extra),
                scene_match=scene_match, n_hist=H, D=lesson[1] - lesson[0], n_infer=n_inf, goal=goal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="VideoRepick")
    ap.add_argument("--k", type=int, nargs="+", default=[0, 3, 7])
    ap.add_argument("--systems", nargs="+", default=["uniform", "retrieval"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--data_dir", default="/vol/data")
    ap.add_argument("--distractor_pool", default="other", choices=["other", "same", "all"])
    ap.add_argument("--shuffle_sessions", action="store_true")
    ap.add_argument("--split", default="test", help="benchmark split: test (default) or val (a second, independent set of scene instances)")
    ap.add_argument("--k0_once", action="store_true", help="run k=0 only for the first listed system (all systems share the k=0 memory)")
    ap.add_argument("--out", default="/vol/eval_cross/results.json")
    ap.add_argument("--max_steps", type=int, default=1300)
    ap.add_argument("--same_scene", type=int, default=0, help="number of the k distractor sessions that are recorded in the query scene itself (hard routing)")
    ap.add_argument("--same_scene_cache", default="/vol/samescene")
    ap.add_argument("--video_dir", default="/vol/eval_cross/videos")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    results = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["system"], r["k"], r["task"], r["episode"]) for r in results}
    client = wcp.MMEVLAWebsocketClientPolicy(host="0.0.0.0", port=a.port)
    runner = EnvRunner(a.task, a.video_dir, max_steps=a.max_steps)
    if a.split != "test":
        from robomme.env_record_wrapper import BenchmarkEnvBuilder
        runner.env_builder = BenchmarkEnvBuilder(env_id=a.task, dataset=a.split, action_space="joint_angle", gui_render=False, max_steps=a.max_steps)
        print(f"split {a.split}: {runner.num_episodes} episodes", flush=True)
    units = load_units(a.data_dir, a.task, pool=a.distractor_pool)
    print(f"{len(units)} distractor units from {sorted({u['task'] for u in units})}", flush=True)
    for ep in range(min(a.episodes, runner.num_episodes)):
        for k in a.k:
            for system in a.systems:
                key = (system, k, a.task, ep)
                if key in done or (a.k0_once and k == 0 and system != a.systems[0]):
                    continue
                rng = random.Random(1000 * ep + k)   # same distractors for every system
                t0 = time.time()
                try:
                    r = run_episode(client, runner, ep, k, system, units, rng, a.max_steps, a.shuffle_sessions, same_scene=a.same_scene,
                                    same_scene_kw=dict(task=a.task, split=a.split, data_dir=a.data_dir, cache_dir=a.same_scene_cache))
                except Exception as e:
                    traceback.print_exc()
                    r = dict(outcome="error", success=False, steps=0, lesson_frac=float("nan"), exec_frac=float("nan"), n_hist=0, D=0, n_infer=0, goal=str(e)[:200])
                r.update(system=system, k=k, task=a.task, episode=ep, secs=time.time() - t0, pool=a.distractor_pool, shuffled=a.shuffle_sessions, split=a.split, same_scene=a.same_scene)
                results.append(r)
                json.dump(results, open(a.out, "w"), indent=1)
                print(f"{system:13s} k={k} ep={ep} -> {r['outcome']:8s} steps={r['steps']} lesson_frac={r['lesson_frac']:.2f} exec_frac={r['exec_frac']:.2f} same_frac={r.get('same_frac', float('nan')):.2f} match={r.get('scene_match')} ({r['secs']:.0f}s)", flush=True)
    agg = collections.defaultdict(list)
    for r in results:
        agg[(r["system"], r["k"])].append(r["success"])
    for (s, k), v in sorted(agg.items()):
        print(f"SUMMARY {s:13s} k={k}: {np.mean(v):.2f} (n={len(v)})")


if __name__ == "__main__":
    main()
