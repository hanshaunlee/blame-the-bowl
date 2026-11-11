"""Belief-state expert.

The expert never reads hidden state. It acts on the *history of events* it has observed on
the current layout (across sessions) plus the current observable state (position, holding,
which drawer is open). This makes expert behaviour a function of history, so a student with
the right memory can imitate it exactly, and "what to remember" has a ground truth:
`relevant_events` returns the indices of the history events the expert's decision used.
"""
from __future__ import annotations
from .world import *


def path_action(src, dst):
    """One greedy Manhattan step (rows first, then columns)."""
    if src[0] < dst[0]:
        return A_DOWN
    if src[0] > dst[0]:
        return A_UP
    if src[1] < dst[1]:
        return A_RIGHT
    if src[1] > dst[1]:
        return A_LEFT
    return None


class Belief:
    """Facts the expert can derive from the event history on one layout."""

    def __init__(self, history: list[tuple[int, Event]]):
        # history: list of (global_event_index, Event) in temporal order, this layout only
        self.drawer_color = {}      # drawer -> colour (latest knowledge)
        self.drawer_src = {}        # drawer -> event index that established the knowledge
        self.visited = {}           # drawer -> event index
        self.failed = {}            # strategy -> event index
        self.ok = None              # (strategy, idx)
        self.cue_k = None           # (k, idx)
        self.cue_seq = {}           # position -> (button, idx)
        self.placed = None          # (count, idx)
        self.pressed = []           # list of (button, idx) in this session (reset by session event)
        self.pressed_counts = []    # list of (count, idx)
        self.picked = {}            # colour -> idx
        for gi, e in history:
            t = e.type
            if t == EV_SESSION:
                self.pressed = []
                self.pressed_counts = []
                self.placed = None
                self.cue_k = None
                self.cue_seq = {}
                self.picked = {}
            elif t == EV_DRAWER:
                self.drawer_color[e.a1] = e.a2
                self.drawer_src[e.a1] = gi
                self.visited[e.a1] = gi
            elif t == EV_DRAWER_EMPTY:
                self.drawer_color[e.a1] = -1
                self.drawer_src[e.a1] = gi
                self.visited[e.a1] = gi
            elif t == EV_PICKED:
                if e.a1 < N_COLORS and e.a2 == 0:
                    # the object left its drawer: that drawer is now empty
                    for d, c in list(self.drawer_color.items()):
                        if c == e.a1:
                            self.drawer_color[d] = -1
                            self.drawer_src[d] = gi
                    self.picked[e.a1] = gi
            elif t == EV_STRAT_FAIL:
                self.failed[e.a1] = gi
            elif t == EV_STRAT_OK:
                self.ok = (e.a1, gi)
            elif t == EV_CUE_COUNT:
                self.cue_k = (e.a1, gi)
            elif t == EV_CUE_SEQ:
                # packed: a1 = b*2+pos_flag, a2 = b*2+1 or 7(=none)
                self.cue_seq_add(e, gi)
            elif t == EV_PLACED_BIN:
                self.placed = (e.a1, gi)
            elif t == EV_PRESSED:
                self.pressed.append((e.a1, gi))
                self.pressed_counts.append((e.a2, gi))

    def cue_seq_add(self, e, gi):
        self.cue_seq[e.a1] = (e.a2, gi)


class _S:
    """Observable state the expert needs: pos, holding, open_drawer, cabinet_open."""
    __slots__ = ("pos", "holding", "open_drawer", "cabinet_open")

    def __init__(self, pos, holding, open_drawer, cabinet_open):
        self.pos, self.holding, self.open_drawer, self.cabinet_open = pos, holding, open_drawer, cabinet_open


def expert_from_state(L: Layout, task: TaskSpec, render_state: tuple, history):
    pos, holding, open_drawer, cabinet_open = render_state[0], render_state[1], render_state[2], render_state[3]
    return _expert(L, task, _S(pos, holding, open_drawer, cabinet_open), history)


def expert_action(w: World, history: list[tuple[int, Event]]):
    """Returns (action, relevant_event_indices)."""
    return _expert(w.L, w.task, w, history)


def _expert(L: Layout, T: TaskSpec, w, history):
    B = Belief(history)
    pos = w.pos
    rel = set()

    def goto(dst):
        a = path_action(pos, dst)
        return a

    if T.name in ("find", "adapt"):
        target = T.target_color if T.name == "find" else None
        if w.holding != -2:
            a = goto(L.goal)
            return (a if a is not None else A_PLACE), rel
        if T.name == "find":
            known = [d for d, c in B.drawer_color.items() if c == target]
            if known:
                d = known[0]
                rel.add(B.drawer_src[d])
                dst = L.drawers[d]
                if pos != dst:
                    return goto(dst), rel
                if w.open_drawer != d:
                    return A_INTERACT, rel
                return A_PICK, rel
            # search: nearest unvisited drawer
            unvisited = [d for d in range(N_DRAWERS) if d not in B.visited]
            rel.update(B.visited.values())
            if not unvisited:  # knowledge inconsistent; re-search all
                unvisited = list(range(N_DRAWERS))
            d = unvisited[0]  # canonical order: lowest-index unvisited drawer
            dst = L.drawers[d]
            if pos != dst:
                return goto(dst), rel
            return A_INTERACT, rel
        else:  # adapt
            dst = L.cabinet
            if pos != dst:
                if B.ok is not None:
                    rel.add(B.ok[1])
                else:
                    rel.update(B.failed.values())
                return goto(dst), rel
            if w.cabinet_open:
                return A_PICK, rel
            if B.ok is not None:
                rel.add(B.ok[1])
                return A_STRAT0 + B.ok[0], rel
            rel.update(B.failed.values())
            for k in range(N_STRATS):
                if k not in B.failed:
                    return A_STRAT0 + k, rel
            return A_STRAT0, rel
    if T.name == "count":
        n = B.placed[0] if B.placed else 0
        if B.placed:
            rel.add(B.placed[1])
        if B.cue_k:
            k = B.cue_k[0]
            rel.add(B.cue_k[1])
        else:
            k = n + 1  # cue forgotten: keep adding (consistent guess)
        if n >= k:
            return A_INTERACT, rel  # should not happen (episode ended)
        if w.holding != -2:
            a = goto(L.bin)
            return (a if a is not None else A_PLACE), rel
        a = goto(L.dispenser)
        return (a if a is not None else A_PICK), rel
    if T.name == "sequence":
        # progress = the count carried by the latest press fact (robust to forgotten earlier presses)
        n = max([e_count for e_count, _ in B.pressed_counts], default=0)
        for _, gi in B.pressed[-1:]:
            rel.add(gi)
        if n < 3 and n in B.cue_seq:
            b, gi = B.cue_seq[n]
            rel.add(gi)
        else:
            b = 0  # cue forgotten: consistent guess
        dst = L.buttons[b]
        if pos != dst:
            return goto(dst), rel
        return A_INTERACT, rel
    if T.name == "tidy":
        unvisited = [d for d in range(N_DRAWERS) if d not in B.visited]
        rel.update(B.visited.values())
        if not unvisited:
            return A_INTERACT, rel
        d = unvisited[0]  # canonical order
        dst = L.drawers[d]
        if pos != dst:
            return goto(dst), rel
        return A_INTERACT, rel
    raise ValueError(T.name)
