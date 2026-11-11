"""Generate held-out oracle demonstrations (val split) with the benchmark's motion-planning solver.

The released checkpoints were trained on the 100-episode train split (the public h5 files), on
which their loss is ~1e-3 (memorised), so a memory-selection reward needs unseen demonstrations.
The benchmark ships the oracle solve() callables per subtask and a 50-episode val split with
fixed seeds; this script runs them and records exactly what the h5 files hold.

    .venv/bin/python hmrobo/gen_demos.py --task VideoRepick --split val --out /vol/demos/val
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
import numpy as np, torch

sys.path.insert(0, "/app/robomme_policy_learning/examples/robomme")
from robomme.env_record_wrapper import BenchmarkEnvBuilder  # noqa: E402
try:
    from robomme.robomme_env.utils.planner_fail_safe import (FailAwarePandaArmMotionPlanningSolver as Arm,  # noqa: E402
                                                             FailAwarePandaStickMotionPlanningSolver as Stick, ScrewPlanFailure)
except Exception:  # pragma: no cover
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver as Arm  # noqa: E402
    Stick = Arm
    ScrewPlanFailure = RuntimeError


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _retry_screw(planner, n_screw=3, n_rrt=3):
    screw, rrt = planner.move_to_pose_with_screw, planner.move_to_pose_with_RRTStar

    def f(*args, **kw):
        for _ in range(n_screw):
            try:
                r = screw(*args, **kw)
            except ScrewPlanFailure:
                continue
            if not (isinstance(r, int) and r == -1):
                return r
        for _ in range(n_rrt):
            try:
                r = rrt(*args, **kw)
            except Exception:
                continue
            if not (isinstance(r, int) and r == -1):
                return r
        return -1
    planner.move_to_pose_with_screw = f


def run_episode(task, split, ep, max_steps):
    builder = BenchmarkEnvBuilder(env_id=task, dataset=split, action_space="joint_angle", gui_render=False, max_steps=max_steps)
    env = builder.make_env_for_episode(ep)
    obs, info = env.reset()
    goal = info["task_goal"][0] if isinstance(info["task_goal"], (list, tuple)) else str(info["task_goal"])
    seed = builder.resolve_episode(ep)[0]
    fronts = [_np(x).astype(np.uint8) for x in obs["front_rgb_list"]]
    wrists = [_np(x).astype(np.uint8) for x in obs["wrist_rgb_list"]]
    joints = [_np(x).astype(np.float32).reshape(-1) for x in obs["joint_state_list"]]
    grips = [_np(x).astype(np.float32).reshape(-1) for x in obs["gripper_state_list"]]
    n_pre = len(fronts)
    is_demo = [i < n_pre - 1 for i in range(n_pre)]
    actions = [None] * n_pre
    u = env.unwrapped
    stick = u.spec.id in ("PatternLock", "RouteStick")
    kw = dict(debug=False, vis=False, base_pose=u.agent.robot.pose, visualize_target_grasp_pose=False, print_env_info=False)
    if stick:
        kw["joint_vel_limits"] = 0.3
    planner = (Stick if stick else Arm)(env, **kw)
    _retry_screw(planner)
    rec = dict(status="ongoing", done=False, err=None)
    orig = planner.env.step

    def step(action):
        if rec["done"]:
            raise StopIteration("episode finished")
        out = orig(action)
        o, r, term, trunc, inf = out
        a = np.asarray(_np(action), np.float32).reshape(-1)[:8]
        for i in range(len(o["front_rgb_list"])):
            fronts.append(_np(o["front_rgb_list"][i]).astype(np.uint8)); wrists.append(_np(o["wrist_rgb_list"][i]).astype(np.uint8))
            joints.append(_np(o["joint_state_list"][i]).astype(np.float32).reshape(-1)); grips.append(_np(o["gripper_state_list"][i]).astype(np.float32).reshape(-1))
            is_demo.append(False); actions.append(a)
        rec["status"] = inf.get("status", "unknown")
        if bool(_np(term).any()) or bool(_np(trunc).any()) or rec["status"] in ("success", "fail", "timeout", "error"):
            rec["done"] = True
        return out
    planner.env.step = step
    failed = False
    try:
        for te in u.task_list:
            if te.get("demonstration", False) or rec["done"]:
                continue
            try:
                res = te["solve"](env, planner)
            except StopIteration:
                break
            except ScrewPlanFailure:
                failed = True; break
            except Exception as e:
                rec["err"] = f"{type(e).__name__}: {e}"; failed = True; break
            bad = (isinstance(res, int) and res == -1) or (isinstance(res, (list, tuple)) and any(isinstance(x, int) and x == -1 for x in res))
            if bad:
                failed = True; break
    finally:
        planner.env.step = orig
        try:
            planner.close()
        except Exception:
            pass
        env.close()
    for i in range(n_pre):
        actions[i] = np.concatenate([joints[i][:7], [1.0 if grips[i][0] > 0.02 else -1.0]]).astype(np.float32)
    T = len(fronts)
    data = dict(front=np.stack(fronts), wrist=np.stack(wrists), joint=np.stack(joints)[:, :7], gripper=np.stack(grips)[:, :2],
                joint_action=np.stack(actions)[:, :8].astype(np.float32), is_demo=np.array(is_demo, bool), goal=np.array(goal), seed=np.array(seed if seed is not None else -1),
                status=np.array(rec["status"]), n_pre=np.array(n_pre))
    return data, dict(status=rec["status"], failed=failed, err=rec["err"], T=T, n_pre=n_pre, goal=goal, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="VideoRepick")
    ap.add_argument("--split", default="val")
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="episode indices (default: all)")
    ap.add_argument("--max_steps", type=int, default=1300)
    ap.add_argument("--out", default="/vol/demos/val")
    a = ap.parse_args()
    out = f"{a.out}/{a.task}"; os.makedirs(out, exist_ok=True)
    n = BenchmarkEnvBuilder(env_id=a.task, dataset=a.split, action_space="joint_angle", max_steps=a.max_steps).get_episode_num()
    eps = a.episodes if a.episodes else list(range(n))
    log_path = f"{out}/log.json"
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    for ep in eps:
        if str(ep) in log and (log[str(ep)].get("status") == "success" or log[str(ep)].get("attempts", 0) >= 2):
            continue
        t0 = time.time()
        try:
            data, meta = run_episode(a.task, a.split, ep, a.max_steps)
        except Exception as e:
            traceback.print_exc()
            meta = dict(status="exception", failed=True, err=f"{type(e).__name__}: {e}"[:300], T=0)
            data = None
        meta["secs"] = time.time() - t0
        meta["attempts"] = log.get(str(ep), {}).get("attempts", 0) + 1
        if data is not None and meta["status"] == "success" and not meta["failed"]:
            np.savez_compressed(f"{out}/ep{ep:03d}.npz", **data)
        log[str(ep)] = meta
        json.dump(log, open(log_path, "w"), indent=1)
        print(f"{a.task} {a.split} ep={ep} -> {meta['status']} failed={meta['failed']} T={meta['T']} ({meta['secs']:.0f}s) {meta.get('err') or ''}", flush=True)
    ok = sum(1 for m in log.values() if m.get("status") == "success" and not m.get("failed"))
    print(f"DONE {a.task}: {ok}/{len(log)} successful demonstrations in {out}")


if __name__ == "__main__":
    main()
