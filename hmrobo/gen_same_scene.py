"""Generate and cache same-scene sessions (hard-routing condition) for a split, plus each episode's lesson video
and execution-start view, for router training. Simulator venv, no policy server needed.

    python hmrobo/gen_same_scene.py --task MoveCube --split train --episodes 50 --m 4
Writes /vol/samescene/<task>/<split>/ep<i>_<j>.npz (j < m) and ep<i>_lesson.npz (lesson frames at --lesson_stride, states, exec-start frame).
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, "/app/robomme_policy_learning/examples/robomme")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_runner import EnvRunner  # noqa: E402
from hmrobo.eval_cross import same_scene_session, scene_fingerprint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="MoveCube")
    ap.add_argument("--split", default="train")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--lesson_stride", type=int, default=4)
    ap.add_argument("--data_dir", default="/vol/data")
    ap.add_argument("--cache", default="/vol/samescene")
    ap.add_argument("--max_steps", type=int, default=1300)
    a = ap.parse_args()
    runner = EnvRunner(a.task, "/tmp/videos", max_steps=a.max_steps)
    if a.split != "test":
        from robomme.env_record_wrapper import BenchmarkEnvBuilder
        runner.env_builder = BenchmarkEnvBuilder(env_id=a.task, dataset=a.split, action_space="joint_angle", gui_render=False, max_steps=a.max_steps)
    n = min(a.episodes, runner.num_episodes)
    print(f"{a.task} {a.split}: {runner.num_episodes} episodes, generating {n} x ({a.m} sessions + lesson)", flush=True)
    t0 = time.time()
    for ep in range(n):
        lp = f"{a.cache}/{a.task}/{a.split}/ep{ep}_lesson.npz"
        if not os.path.exists(lp):
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            runner.make_env(ep); pre = runner.get_init_obs(); runner.close_env()
            L = len(pre["images"]) - 1
            idx = np.arange(0, L, a.lesson_stride)
            np.savez_compressed(lp, front=np.stack([pre["images"][i] for i in idx]), state=np.stack([pre["states"][i] for i in idx]),
                                exec_front=pre["images"][-1], exec_state=pre["states"][-1], fingerprint=scene_fingerprint(pre["images"][0]), goal=str(pre["task_goal"]), L=L)
        for j in range(a.m):
            same_scene_session(runner, ep, a.task, a.split, j, a.data_dir, a.cache)
        print(f"ep {ep} done ({time.time()-t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
