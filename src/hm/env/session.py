"""Trials: sequences of sessions with persistent layouts and a running event history.

A trial is either
  single(task)    : one session on a fresh layout, or
  cross(k)        : session 0 on layout L0 (tidy / find / adapt, which reveals drawer contents or
                    the cabinet latch), then k distractor sessions on fresh layouts, then a QUERY
                    session back on L0 (find or adapt) with a tight step limit so that only an
                    agent that remembers session 0 can succeed.  This copies the protocol of
                    RoboMME-Interference.

Everything is stored compactly (render states, not pixels); frames are re-rendered on demand.
"""
from __future__ import annotations
import copy
import numpy as np
from dataclasses import dataclass, field
from .world import *
from .oracle import expert_action

W_FRAMES = 3  # working memory: previous frames shown to the policy
DISTRACTOR_TASKS = ["count", "sequence", "find", "adapt", "tidy"]


@dataclass
class EventRec:
    session: int
    t: int
    step: int          # global step index in the trial (frame to use for the perceptual unit)
    layout: int
    ev: Event

    def vec(self):
        return self.ev.vec()


@dataclass
class StepRec:
    session: int
    t: int
    layout: int
    state: tuple
    proprio: np.ndarray
    instr: np.ndarray
    action: int
    n_avail: int        # events with index < n_avail are observable at this step
    rel: tuple          # event indices the expert used


@dataclass
class Trial:
    kind: str
    k: int
    layouts: list
    sessions: list       # list of (layout_idx, TaskSpec)
    steps: list = field(default_factory=list)
    events: list = field(default_factory=list)
    successes: list = field(default_factory=list)
    session_bounds: list = field(default_factory=list)  # (first_step, last_step+1) per session
    init_layouts: list = field(default_factory=list)    # deep copies of layouts before any session ran
    query_restock: int = -2                              # cabinet colour restocked before an adapt query

    @property
    def query_session(self):
        return len(self.sessions) - 1 if self.kind == "cross" else None


def sample_task(name: str, L: Layout, rng, history_belief_colors=None) -> TaskSpec:
    if name == "find":
        avail = [c for c in L.drawer_contents if c >= 0]
        return TaskSpec("find", target_color=int(rng.choice(avail)), max_steps=MAX_STEPS["find"])
    if name == "count":
        return TaskSpec("count", count_k=int(rng.integers(1, 4)), max_steps=MAX_STEPS["count"])
    if name == "sequence":
        seq = tuple(int(x) for x in rng.permutation(N_BUTTONS)[:3])
        return TaskSpec("sequence", seq=seq, max_steps=MAX_STEPS["sequence"])
    return TaskSpec(name, max_steps=MAX_STEPS[name])


def run_session(trial: Trial, li: int, task: TaskSpec, rng, layout_history: dict, actor=None):
    """Run one session with the expert (or a given actor) and append records to the trial."""
    L = trial.layouts[li]
    w = World(L, task, rng)
    s = len(trial.sessions)
    trial.sessions.append((li, task))
    first = len(trial.steps)
    hist = layout_history.setdefault(li, [])  # list of (global_event_idx, Event) for this layout
    obs = w.obs()
    while True:
        # register events emitted with this observation
        for e in obs["events"]:
            gi = len(trial.events)
            trial.events.append(EventRec(s, w.t, len(trial.steps), li, e))
            hist.append((gi, e))
        a, rel = expert_action(w, hist)
        trial.steps.append(StepRec(s, w.t, li, w.render_state(), obs["proprio"], obs["instr"], int(a),
                                   len(trial.events), tuple(sorted(rel))))
        obs, done = w.step(a)
        if done:
            for e in obs["events"]:  # terminal events (e.g. delivered) still count as history
                gi = len(trial.events)
                trial.events.append(EventRec(s, w.t, len(trial.steps) - 1, li, e))
                hist.append((gi, e))
            break
    trial.successes.append(bool(w.success))
    trial.session_bounds.append((first, len(trial.steps)))
    return w


def expert_steps(L: Layout, task: TaskSpec, hist, rng) -> int:
    """Number of steps the expert needs on this layout given history `hist`."""
    w = World(copy.deepcopy(L), task, rng)
    h = list(hist)
    obs = w.obs()
    n = 0
    while True:
        for e in obs["events"]:
            h.append((-1, e))
        a, _ = expert_action(w, h)
        obs, done = w.step(a)
        n += 1
        if done:
            return n if w.success else 10 ** 6


def make_single(rng, task_name: str) -> Trial:
    L = Layout.random(rng)
    tr = Trial("single", 0, [L], [])
    tr.init_layouts = [copy.deepcopy(L)]
    run_session(tr, 0, sample_task(task_name, L, rng), rng, {})
    return tr


def make_cross(rng, k: int, first_task: str | None = None, query_task: str | None = None) -> Trial:
    for _ in range(20):
        tr = _make_cross(rng, k, first_task, query_task)
        if tr is not None:
            return tr
    return _make_cross(rng, k, "tidy", query_task)


def _make_cross(rng, k, first_task, query_task):
    L0 = Layout.random(rng)
    tr = Trial("cross", k, [L0], [])
    tr.init_layouts = [copy.deepcopy(L0)]
    hist = {}
    first_task = first_task or str(rng.choice(["tidy", "find", "adapt", "tidy"]))
    run_session(tr, 0, sample_task(first_task, L0, rng), rng, hist)
    for _ in range(k):
        Ld = Layout.random(rng)
        tr.layouts.append(Ld)
        tr.init_layouts.append(copy.deepcopy(Ld))
        run_session(tr, len(tr.layouts) - 1, sample_task(str(rng.choice(DISTRACTOR_TASKS)), Ld, rng), rng, hist)
    # query on L0
    from .oracle import Belief
    B = Belief(hist[0])
    known_colors = [c for d, c in B.drawer_color.items() if c >= 0 and tr.layouts[0].drawer_contents[d] == c]
    # drawer contents may have changed in session 0 (object delivered); use the live layout state
    options = []
    if known_colors:
        options.append("find")
    if B.ok is not None:
        options.append("adapt")  # the cabinet is restocked below; the latch persists
    if query_task in options:
        q = query_task
    elif options:
        q = str(rng.choice(options))
    else:  # nothing learned in session 0: not a valid memory query
        return None
    if q == "find":
        task = TaskSpec("find", target_color=int(rng.choice(known_colors)) if known_colors
                        else int(rng.choice([c for c in L0.drawer_contents if c >= 0])))
    else:
        L0.cabinet_content = int(rng.integers(N_COLORS))  # restocked
        tr.query_restock = L0.cabinet_content
        task = TaskSpec("adapt")
    # tight limit: expert-with-memory steps + 2, and the query must be unsolvable without memory
    n = expert_steps(L0, task, hist[0], np.random.default_rng(0))
    task.max_steps = int(min(n + 2, MAX_STEPS["find"])) if n < 10 ** 6 else MAX_STEPS["find"]
    n0 = expert_steps(L0, task, [], np.random.default_rng(0))
    if n0 <= task.max_steps:
        return None
    run_session(tr, 0, task, rng, hist)
    return tr


def make_trial(rng, spec: dict) -> Trial:
    if spec["kind"] == "single":
        return make_single(rng, spec["task"])
    return make_cross(rng, spec["k"], spec.get("first_task"), spec.get("query_task"))


def frames_for_step(trial: Trial, i: int, n_prev: int = W_FRAMES) -> list[np.ndarray]:
    """Current frame plus n_prev previous frames from the same session (padded by repetition)."""
    st = trial.steps[i]
    out = []
    for j in range(i, i - n_prev - 1, -1):
        jj = j
        while jj < i and (jj < 0 or trial.steps[jj].session != st.session):
            jj += 1
        jj = max(jj, 0)
        if trial.steps[jj].session != st.session:
            jj = i
        s = trial.steps[jj]
        out.append(render_from(trial.layouts[s.layout], trial.sessions[s.session][1], s.state))
    return out  # [t, t-1, t-2, ...]


def frame_for_event(trial: Trial, ei: int) -> np.ndarray:
    e = trial.events[ei]
    s = trial.steps[e.step]
    return render_from(trial.layouts[s.layout], trial.sessions[s.session][1], s.state)
