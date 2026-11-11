"""Long-term memory systems. Each keeps <= B slots of (event_index, modality) across the
sessions of a trial. modality 0 = symbolic fact, 1 = frame snapshot.

Decision points:
  on_event(trial, ei)          an event was just observed (online write opportunity)
  on_session_end(trial, s)     the session ended (consolidation opportunity)
"""
from __future__ import annotations
import numpy as np
from ..env.world import *

SYM, FRM = 0, 1
SESSION_LOCAL = {EV_CUE_COUNT, EV_CUE_SEQ, EV_PLACED_BIN, EV_PRESSED, EV_SESSION, EV_DELIVERED}


class MemorySystem:
    name = "base"
    learned = False

    def __init__(self, B: int, rng=None):
        self.B = B
        self.rng = rng or np.random.default_rng(0)
        self.slots: list[tuple[int, int]] = []

    def reset_trial(self):
        self.slots = []

    def on_event(self, trial, ei: int):
        pass

    def on_session_end(self, trial, s: int):
        pass

    def contents(self):
        return list(self.slots)


class NoMemory(MemorySystem):
    name = "none"


class RecencySym(MemorySystem):
    name = "recency_sym"

    def on_event(self, trial, ei):
        self.slots.append((ei, SYM))
        self.slots = self.slots[-self.B:]


class RecencyFrame(MemorySystem):
    """Keyframe memory: the frame at every event, most recent B (EventVLA/KEMO-style without learning)."""
    name = "recency_frame"

    def on_event(self, trial, ei):
        self.slots.append((ei, FRM))
        self.slots = self.slots[-self.B:]


class Reservoir(MemorySystem):
    name = "random"

    def reset_trial(self):
        super().reset_trial(); self.n = 0

    def on_event(self, trial, ei):
        self.n += 1
        if len(self.slots) < self.B:
            self.slots.append((ei, SYM))
        else:
            j = int(self.rng.integers(self.n))
            if j < self.B:
                self.slots[j] = (ei, SYM)


class ScriptedMEM(MemorySystem):
    """Hand-written stand-in for MEM's LLM compressor: symbolic facts only; one fact per drawer
    (latest), drop a drawer fact when its object is picked, keep latest progress fact only,
    drop failed strategies once the latch is known (and optionally always), drop session-local
    facts at session end, FIFO when over budget."""

    def __init__(self, B, rng=None, keep_fail=True):
        super().__init__(B, rng)
        self.keep_fail = keep_fail
        self.name = "scripted_mem" + ("" if keep_fail else "_dropfail")

    def _drop(self, pred):
        self.slots = [(i, m) for (i, m) in self.slots if not pred(i)]

    def on_event(self, trial, ei):
        e = trial.events[ei].ev
        lay = trial.events[ei].layout
        E = trial.events
        if e.type in (EV_DRAWER, EV_DRAWER_EMPTY):
            self._drop(lambda i: E[i].layout == lay and E[i].ev.type in (EV_DRAWER, EV_DRAWER_EMPTY) and E[i].ev.a1 == e.a1)
        elif e.type == EV_PICKED:
            if e.a1 < N_COLORS and e.a2 == 0:
                self._drop(lambda i: E[i].layout == lay and E[i].ev.type == EV_DRAWER and E[i].ev.a2 == e.a1)
            return
        elif e.type in (EV_PLACED_BIN, EV_PRESSED):
            self._drop(lambda i: E[i].ev.type == e.type)
        elif e.type == EV_STRAT_OK:
            self._drop(lambda i: E[i].layout == lay and E[i].ev.type in (EV_STRAT_FAIL, EV_STRAT_OK))
        elif e.type == EV_STRAT_FAIL:
            if not self.keep_fail:
                return
        elif e.type in (EV_SESSION, EV_DELIVERED):
            return
        self.slots.append((ei, SYM))
        self.slots = self.slots[-self.B:]

    def on_session_end(self, trial, s):
        E = trial.events
        self._drop(lambda i: E[i].ev.type in SESSION_LOCAL)


class OracleMemory(MemorySystem):
    """Cheating upper bound: knows which facts the future query will need. Those facts are
    protected (never evicted); remaining slots behave like recency_sym."""
    name = "oracle"

    def __init__(self, B, rng=None, query_task=None, query_layout=0):
        super().__init__(B, rng)
        self.query_task = query_task
        self.query_layout = query_layout
        self.protected = []

    def reset_trial(self):
        super().reset_trial(); self.protected = []; self.n_seen = 0

    def _compose(self, recent):
        self.slots = [(i, SYM) for i in self.protected][: self.B]
        for ei, m in recent:
            if len(self.slots) >= self.B:
                break
            if all(ei != j for j, _ in self.slots):
                self.slots.append((ei, m))

    def on_event(self, trial, ei):
        self.n_seen = max(self.n_seen, ei + 1)
        recent = [x for x in self.slots if x[0] not in self.protected] + [(ei, SYM)]
        recent = recent[-self.B:]
        self._compose(recent)

    def on_session_end(self, trial, s):
        E = trial.events
        q = self.query_task
        if q is None:
            return
        latest_drawer, ok = {}, []
        for i, e in enumerate(E[: self.n_seen]):
            if e.layout != self.query_layout:
                continue
            if e.ev.type in (EV_DRAWER, EV_DRAWER_EMPTY):
                latest_drawer[e.ev.a1] = i
            elif e.ev.type == EV_PICKED and e.ev.a1 < N_COLORS and e.ev.a2 == 0:
                for d, j in list(latest_drawer.items()):
                    if E[j].ev.type == EV_DRAWER and E[j].ev.a2 == e.ev.a1:
                        latest_drawer[d] = i  # stale: the picked event supersedes it
            elif e.ev.type == EV_STRAT_OK:
                ok = [i]
        keep = []
        if q.name == "find":
            tgt = [i for d, i in latest_drawer.items() if E[i].ev.type == EV_DRAWER and E[i].ev.a2 == q.target_color]
            rest = [i for d, i in latest_drawer.items() if i not in tgt and E[i].ev.type in (EV_DRAWER, EV_DRAWER_EMPTY)]
            keep = tgt + rest
        elif q.name == "adapt":
            keep = ok
        self.protected = keep[: self.B]
        recent = [x for x in self.slots if x[0] not in self.protected]
        self._compose(recent)


def heuristic_systems(B, rng, query_task=None):
    return [NoMemory(B, rng), RecencySym(B, rng), RecencyFrame(B, rng), Reservoir(B, rng),
            ScriptedMEM(B, rng, keep_fail=True), ScriptedMEM(B, rng, keep_fail=False),
            OracleMemory(B, rng, query_task)]
