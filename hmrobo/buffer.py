"""Cross-session history construction for RoboMME-Interference, from the released h5 demos.

Protocol (Rathi 2026): buffer = [lesson] + k x [distractor unit] (+ the rollout's own observations
accumulate afterwards). The lesson is the query episode's own demonstration section
(frames with info/is_video_demo == True). A distractor unit is the first 256 frames of an episode
from a *different* task family, subsampled at stride 8 (32 frames). FrameSamp then samples 32
frames evenly over the whole buffer, which is why it decays with k.
"""
from __future__ import annotations
import h5py, numpy as np, random, os, glob
from dataclasses import dataclass

LESSON_TASKS = ["MoveCube", "InsertPeg", "PatternLock", "RouteStick", "VideoRepick", "VideoPlaceButton",
                "VideoPlaceOrder", "VideoUnmask", "VideoUnmaskSwap"]
UNIT = 32           # frames per distractor unit (= FrameSamp budget)
UNIT_SRC = 256      # raw frames taken from the distractor episode
UNIT_STRIDE = 8


def episode_keys(f: h5py.File):
    return sorted([k for k in f.keys() if k.startswith("episode")], key=lambda s: int(s.split("_")[1]))


def timestep_keys(ep):
    return sorted([k for k in ep.keys() if k.startswith("timestep")], key=lambda s: int(s.split("_")[1]))


@dataclass
class Episode:
    task: str
    key: str
    front: np.ndarray        # (T,256,256,3) uint8
    wrist: np.ndarray
    joint: np.ndarray        # (T,7)
    gripper: np.ndarray      # (T,2)
    joint_action: np.ndarray # (T,8)
    is_demo: np.ndarray      # (T,) bool  -- demonstration (video) section
    goal: str

    @property
    def exec_start(self):
        idx = np.where(~self.is_demo)[0]
        return int(idx[0]) if len(idx) else 0

    def lesson_frames(self):
        return self.front[self.is_demo], self.wrist[self.is_demo], self.state[self.is_demo]

    @property
    def state(self):
        return np.concatenate([self.joint, self.gripper[:, :1]], 1).astype(np.float32)  # (T,8) as pack_state does


def load_episode(path: str, key: str, task: str, stride: int = 1, max_T: int | None = None) -> Episode:
    with h5py.File(path, "r") as f:
        ep = f[key]
        ts = timestep_keys(ep)[::stride]
        if max_T:
            ts = ts[:max_T]
        front = np.stack([ep[t]["obs"]["front_rgb"][()] for t in ts])
        wrist = np.stack([ep[t]["obs"]["wrist_rgb"][()] for t in ts])
        joint = np.stack([ep[t]["obs"]["joint_state"][()] for t in ts]).astype(np.float32)
        grip = np.stack([ep[t]["obs"]["gripper_state"][()] for t in ts]).astype(np.float32)
        act = np.stack([ep[t]["action"]["joint_action"][()] for t in ts]).astype(np.float32)
        demo = np.array([bool(ep[t]["info"]["is_video_demo"][()]) for t in ts])
        goal = ep["setup"]["task_goal"][()]
        goal = goal[0] if isinstance(goal, (list, np.ndarray)) else goal
        goal = goal.decode() if isinstance(goal, bytes) else str(goal)
    return Episode(task, key, front, wrist, joint, grip, act, demo, goal)


def load_actions(path: str, key: str, n: int = 256, skip_demo: bool = True) -> np.ndarray:
    """(n,8) joint actions of one recorded episode, starting after its demonstration section (cheap: no images)."""
    with h5py.File(path, "r") as f:
        ep = f[key]; ts = timestep_keys(ep)
        if skip_demo:
            demo = [bool(ep[t]["info"]["is_video_demo"][()]) for t in ts]
            start = demo.index(False) if False in demo else 0
            ts = ts[start:]
        return np.stack([ep[t]["action"]["joint_action"][()] for t in ts[:n]]).astype(np.float32)


def distractor_unit(path: str, key: str, task: str) -> dict:
    """First UNIT_SRC frames at stride UNIT_STRIDE -> 32 frames (front, wrist, state)."""
    ep = load_episode(path, key, task, stride=UNIT_STRIDE, max_T=UNIT_SRC // UNIT_STRIDE)
    return dict(front=ep.front[:UNIT], wrist=ep.wrist[:UNIT], state=ep.state[:UNIT], task=task)


def build_history(query: Episode, distractors: list[dict], k: int):
    """Returns (front, wrist, state, session_id) arrays for lesson + k distractor units."""
    lf, lw, ls = query.lesson_frames()
    fronts, wrists, states, sess = [lf], [lw], [ls], [np.zeros(len(lf), dtype=np.int64)]
    for i, d in enumerate(distractors[:k]):
        fronts.append(d["front"]); wrists.append(d["wrist"]); states.append(d["state"]); sess.append(np.full(len(d["front"]), i + 1))
    return np.concatenate(fronts), np.concatenate(wrists), np.concatenate(states), np.concatenate(sess)


def even_indices(n: int, m: int) -> np.ndarray:
    """FrameSamp's default: m evenly spaced indices over n frames (identity when n <= m)."""
    if n <= m:
        return np.arange(n)
    return np.linspace(0, n - 1, m).round().astype(int)


def load_units(data_dir: str, task: str, pool: str = "other", n_per_task: int = 12, cache_dir: str = "/vol/units"):
    """Distractor units (front, wrist, state, task, key) from cached npz. pool: other tasks | same task | all."""
    os.makedirs(cache_dir, exist_ok=True)
    units = []
    for path in sorted(glob.glob(f"{data_dir}/record_dataset_*.h5")):
        t = os.path.basename(path)[len("record_dataset_"):-3]
        if (pool == "other" and t == task) or (pool == "same" and t != task):
            continue
        cpath = f"{cache_dir}/{t}.npz"
        if not os.path.exists(cpath):
            with h5py.File(path, "r") as f:
                keys = episode_keys(f)[:n_per_task]
            us = [(k, distractor_unit(path, k, t)) for k in keys]
            us = [(k, u) for k, u in us if len(u["front"]) == UNIT]
            np.savez_compressed(cpath, front=np.stack([u["front"] for _, u in us]), wrist=np.stack([u["wrist"] for _, u in us]),
                                state=np.stack([u["state"] for _, u in us]), keys=np.array([k for k, _ in us]))
        z = np.load(cpath)
        keys = list(z["keys"]) if "keys" in z else [f"episode_{i}" for i in range(len(z["front"]))]
        for i in range(len(z["front"])):
            units.append(dict(front=z["front"][i], wrist=z["wrist"][i], state=z["state"][i], task=t, key=str(keys[i])))
    return units


def load_episode_npz(path: str, task: str) -> Episode:
    """Episode from hmrobo/gen_demos.py output (held-out oracle demonstrations)."""
    z = np.load(path, allow_pickle=False)
    return Episode(task, os.path.basename(path)[:-4], z["front"], z["wrist"], z["joint"].astype(np.float32), z["gripper"].astype(np.float32),
                   z["joint_action"].astype(np.float32), z["is_demo"].astype(bool), str(z["goal"]))
