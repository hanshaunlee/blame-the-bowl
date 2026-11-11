"""Build fine-tuning data for the frozen-policy study (runs in the policy-learning venv, GPU).

Output layout (compatible with mme_vla_suite.training.dataset.SampleDataset plus our per-episode features):
  <out>/data/<id>.pkl            one sample per execution step, RoboMME pickle format
  <out>/features/ep<g>.npz       per-episode SigLIP 4x4 features of ALL frames (video demo + execution):
                                 img (T,16,2048) bf16 stored as uint16 bit pattern, state (T,8) f32
  <out>/meta/stats.json          {"execution_samples": N, "total_samples": M}
  <out>/meta/episodes.json       [{"epis_idx", "task", "exec_start", "T", "n_exec", "goal"}]

    .venv/bin/python /app/ours/hmrobo/build_ft_data.py --h5 /vol/data/record_dataset_MoveCube.h5 /vol/data_extra/record_dataset_RouteStick.h5 --out /vol/ftdata [--max_episodes 4]
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, time
import h5py, numpy as np

sys.path.insert(0, "/app/ours")
from mme_vla_suite.shared.mem_buffer import MemoryBuffer  # noqa: E402
from mme_vla_suite.dataset_builder.build_robomme_dataset import get_action_chunk, ACTION_CHUNK_HORIZON  # noqa: E402
from mme_vla_suite.dataset_builder.robomme_h5_utils import first_execution_step  # noqa: E402

CHUNK = 64


def encode(mb: MemoryBuffer, front: np.ndarray, states: np.ndarray) -> np.ndarray:
    """SigLIP 4x4 pooled features for every frame, (T,16,2048) bf16, fixed-shape chunks (one jit)."""
    out = []
    for s in range(0, len(front), CHUNK):
        imgs, sts = front[s:s + CHUNK], states[s:s + CHUNK]
        n = len(imgs)
        if n < CHUNK:
            imgs = np.concatenate([imgs, np.repeat(imgs[-1:], CHUNK - n, 0)]); sts = np.concatenate([sts, np.repeat(sts[-1:], CHUNK - n, 0)])
        mb.clear()
        mb.add_buffer(imgs.astype(np.uint8)[:, None], sts.astype(np.float32), list(range(CHUNK)))
        out += [mb._history_feats[i]["image_emb_4x4"][0] for i in range(n)]
    mb.clear()
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", nargs="+", required=True)
    ap.add_argument("--out", default="/vol/ftdata")
    ap.add_argument("--max_episodes", type=int, default=None)
    a = ap.parse_args()
    for d in ["data", "features", "meta"]:
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    mb = MemoryBuffer(num_views=1, prepare_buffer=True, pool_type="mean")
    episodes, g, sid, total = [], 0, 0, 0
    t0 = time.time()
    for path in a.h5:
        task = os.path.basename(path).replace("record_dataset_", "").replace(".h5", "")
        with h5py.File(path, "r") as f:
            keys = sorted([k for k in f.keys() if k.startswith("episode_")], key=lambda s: int(s.split("_")[1]))
            if a.max_episodes:
                keys = keys[: a.max_episodes]
            for key in keys:
                ep = f[key]
                goal = ep["setup"]["task_goal"][()][0].decode()
                T = sum(1 for k in ep.keys() if k.startswith("timestep_"))
                D = first_execution_step(ep)
                front = np.stack([ep[f"timestep_{i}"]["obs"]["front_rgb"][()] for i in range(T)])
                wrist = np.stack([ep[f"timestep_{i}"]["obs"]["wrist_rgb"][()] for i in range(T)])
                state = np.stack([np.concatenate([ep[f"timestep_{i}"]["obs"]["joint_state"][()], ep[f"timestep_{i}"]["obs"]["gripper_state"][()][:1]]) for i in range(T)]).astype(np.float32)
                feats = encode(mb, front, state)
                np.savez(f"{a.out}/features/ep{g}.npz", img=feats.view(np.uint16), state=state, exec_start=D, task=task, goal=goal)
                n_exec = 0
                for i in range(D, T):
                    ts = ep[f"timestep_{i}"]
                    sg = {k: (ts["info"][k][()].decode().lower() if k in ts["info"] else "") for k in ["simple_subgoal", "grounded_subgoal", "simple_subgoal_online", "grounded_subgoal_online"]}
                    sample = {"image": front[i], "wrist_image": wrist[i], "state": state[i],
                              "actions": get_action_chunk(ep, i, horizon=ACTION_CHUNK_HORIZON),
                              "is_demo": np.array([False]), "exec_start_idx": np.array([D], dtype=np.int32),
                              "step_idx": np.array([i], dtype=np.int32), "epis_idx": np.array([g], dtype=np.int32),
                              "prompt": goal.lower(), **sg}
                    with open(f"{a.out}/data/{sid}.pkl", "wb") as fh:
                        pickle.dump(sample, fh)
                    sid += 1; n_exec += 1
                total += T
                episodes.append({"epis_idx": g, "task": task, "exec_start": int(D), "T": int(T), "n_exec": n_exec, "goal": goal})
                print(f"[{time.time()-t0:6.0f}s] ep{g} {task} {key}: T={T} exec_start={D} samples so far {sid}", flush=True)
                g += 1
                if g % 10 == 0:
                    json.dump(episodes, open(f"{a.out}/meta/episodes.json", "w"))
    json.dump(episodes, open(f"{a.out}/meta/episodes.json", "w"))
    json.dump({"execution_samples": sid, "total_samples": total}, open(f"{a.out}/meta/stats.json", "w"))
    print(f"DONE episodes={g} exec_samples={sid} total_frames={total} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
