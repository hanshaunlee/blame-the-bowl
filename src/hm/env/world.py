"""Pantry: a top-down 2D manipulation world with memory-dependent tasks.

Grid of G x G cells. The agent (a gripper) moves between cells and interacts with the
fixture on its cell. Fixtures:
  drawer   : holds one coloured object or nothing; opens on INTERACT, auto-closes when the
             agent leaves (state aliasing: a closed drawer looks like it did at the start).
  goal     : PLACE here delivers the held object.
  bin      : PLACE here drops the held object in; the bin is opaque (count is invisible).
  dispenser: PICK here yields a generic object (infinite supply).
  cue      : shows a cue (a count or a button sequence) for CUE_STEPS steps, then blank.
  button   : INTERACT presses it (lights for one step).
  cabinet  : hidden latch type; STRAT_k opens it iff k == latch, otherwise a 'stuck' flash.
             Holds one coloured object.

Tasks (see tasks.py): find, count, sequence, adapt, tidy.

Every step the world emits zero or more EVENTS (symbolic facts an annotator could read
off the scene). Events are the candidate memory units, together with the frame at that
step. See session.py.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

G = 9                     # grid size
CELL = 7                  # pixels per cell -> 63 px, padded to 64
IMG = 64
CUE_STEPS = 3
N_DRAWERS = 4
N_BUTTONS = 4
N_STRATS = 3
N_COLORS = 6
MAX_STEPS = {"find": 45, "count": 50, "sequence": 40, "adapt": 40, "tidy": 50, "query": 22}

# actions
A_UP, A_DOWN, A_LEFT, A_RIGHT, A_INTERACT, A_PICK, A_PLACE = range(7)
A_STRAT0 = 7
N_ACTIONS = 7 + N_STRATS  # 10
ACTION_NAMES = ["up", "down", "left", "right", "interact", "pick", "place"] + [f"strat{k}" for k in range(N_STRATS)]

# event types
EV_SESSION, EV_CUE_COUNT, EV_CUE_SEQ, EV_DRAWER, EV_PICKED, EV_PLACED_BIN, EV_DELIVERED, \
    EV_PRESSED, EV_STRAT_FAIL, EV_STRAT_OK, EV_DRAWER_EMPTY = range(11)
N_EVENT_TYPES = 11
EVENT_NAMES = ["session", "cue_count", "cue_seq", "drawer", "picked", "placed_bin", "delivered",
               "pressed", "strat_fail", "strat_ok", "drawer_empty"]
N_ARG = 8  # arg vocabulary size per slot

TASKS = ["find", "count", "sequence", "adapt", "tidy"]
TASK_ID = {t: i for i, t in enumerate(TASKS)}

# colours (RGB 0-255): objects use 0..5
PALETTE = np.array([
    [220, 40, 40],    # red
    [40, 170, 60],    # green
    [50, 90, 230],    # blue
    [240, 200, 30],   # yellow
    [200, 60, 200],   # magenta
    [40, 200, 210],   # cyan
], dtype=np.uint8)
COLOR_NAMES = ["red", "green", "blue", "yellow", "magenta", "cyan"]
C_BG = np.array([235, 232, 225], dtype=np.uint8)
C_AGENT = np.array([20, 20, 20], dtype=np.uint8)
C_DRAWER = np.array([150, 110, 70], dtype=np.uint8)
C_DRAWER_OPEN = np.array([90, 60, 30], dtype=np.uint8)
C_GOAL = np.array([120, 220, 120], dtype=np.uint8)
C_BIN = np.array([90, 90, 100], dtype=np.uint8)
C_DISP = np.array([180, 180, 190], dtype=np.uint8)
C_CUE = np.array([250, 250, 250], dtype=np.uint8)
C_BUTTON = np.array([60, 60, 60], dtype=np.uint8)
C_CABINET = np.array([110, 80, 140], dtype=np.uint8)
C_STUCK = np.array([255, 120, 0], dtype=np.uint8)
BUTTON_COLORS = PALETTE[:N_BUTTONS]


@dataclass
class Event:
    type: int
    a1: int = 0
    a2: int = 0

    def vec(self) -> np.ndarray:
        v = np.zeros(N_EVENT_TYPES + 2 * N_ARG, dtype=np.float32)
        v[self.type] = 1.0
        v[N_EVENT_TYPES + self.a1] = 1.0
        v[N_EVENT_TYPES + N_ARG + self.a2] = 1.0
        return v

    def __str__(self):
        return f"{EVENT_NAMES[self.type]}({self.a1},{self.a2})"


@dataclass
class Layout:
    """Static arrangement of fixtures. Drawer and cabinet contents persist across sessions."""
    drawers: list            # list of (r, c)
    drawer_contents: list    # colour index or -1
    goal: tuple
    bin: tuple
    dispenser: tuple
    cue: tuple
    buttons: list            # list of (r, c)
    cabinet: tuple
    cabinet_latch: int
    cabinet_content: int
    start: tuple

    @staticmethod
    def random(rng: np.random.Generator) -> "Layout":
        cells = [(r, c) for r in range(G) for c in range(G)]
        idx = rng.permutation(len(cells))
        picks = [cells[i] for i in idx[: N_DRAWERS + N_BUTTONS + 6]]
        drawers = picks[:N_DRAWERS]
        buttons = picks[N_DRAWERS:N_DRAWERS + N_BUTTONS]
        goal, bn, disp, cue, cab, start = picks[N_DRAWERS + N_BUTTONS:]
        # drawer contents: distinct colours; at most one empty
        cols = list(rng.permutation(N_COLORS)[:N_DRAWERS])
        if rng.random() < 0.3:
            cols[int(rng.integers(N_DRAWERS))] = -1
        cab_col = int(rng.integers(N_COLORS))
        return Layout(drawers, [int(c) for c in cols], goal, bn, disp, cue, buttons, cab,
                      int(rng.integers(N_STRATS)), cab_col, start)


@dataclass
class TaskSpec:
    name: str
    target_color: int = -1      # find
    count_k: int = 0            # count
    seq: tuple = ()             # sequence (button indices)
    max_steps: int = 40

    @property
    def tid(self):
        return TASK_ID[self.name]

    def instr_vec(self) -> np.ndarray:
        v = np.zeros(len(TASKS) + N_COLORS + 1, dtype=np.float32)
        v[self.tid] = 1.0
        if self.target_color >= 0:
            v[len(TASKS) + self.target_color] = 1.0
        else:
            v[len(TASKS) + N_COLORS] = 1.0
        return v


INSTR_DIM = len(TASKS) + N_COLORS + 1
PROPRIO_DIM = 2 + N_COLORS + 2  # xy, holding colour onehot, holding-generic flag, holding-any flag
EVENT_DIM = N_EVENT_TYPES + 2 * N_ARG


class World:
    """One session (episode) of one task on a layout. Call reset() then step()."""

    def __init__(self, layout: Layout, task: TaskSpec, rng: np.random.Generator):
        self.L = layout
        self.task = task
        self.rng = rng
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        L = self.L
        self.t = 0
        self.pos = L.start
        self.holding = -2            # -2 none, -1 generic, >=0 colour
        self.open_drawer = -1        # index of the open drawer (agent is on it)
        self.cabinet_open = False
        self.cabinet_content = L.cabinet_content   # -2 once taken
        self.cabinet_content_at_start = L.cabinet_content
        self.drawer_contents = L.drawer_contents  # shared with the layout: persists across sessions
        self.bin_count = 0
        self.delivered = -2
        self.pressed = []            # ordered presses (button idx)
        self.lit = -1                # button lit this step
        self.stuck = False           # cabinet stuck flash this step
        self.failed_strats = set()
        self.lock = 0                # steps the cabinet stays jammed after a wrong strategy
        self.done = False
        self.success = False
        self.visited_drawers = set()
        self.events = [Event(EV_SESSION, self.task.tid, 0)]
        if self.task.name == "count":
            self.events.append(Event(EV_CUE_COUNT, self.task.count_k, 0))
        if self.task.name == "sequence":
            for pos_, b in enumerate(self.task.seq):
                self.events.append(Event(EV_CUE_SEQ, pos_, b))  # (position in sequence, button)
        return self.obs()

    def fixture_at(self, pos):
        L = self.L
        if pos in L.drawers:
            return "drawer", L.drawers.index(pos)
        if pos == L.goal:
            return "goal", 0
        if pos == L.bin:
            return "bin", 0
        if pos == L.dispenser:
            return "dispenser", 0
        if pos == L.cue:
            return "cue", 0
        if pos in L.buttons:
            return "button", L.buttons.index(pos)
        if pos == L.cabinet:
            return "cabinet", 0
        return "floor", 0

    # ------------------------------------------------------------------ dynamics
    def step(self, a: int):
        assert not self.done
        self.t += 1
        self.lit = -1
        self.stuck = False
        self.events = []
        r, c = self.pos
        if self.lock > 0:            # jammed: the action is lost
            self.lock -= 1
            self.stuck = True
            self._check_done()
            return self.obs(), self.done
        if a in (A_UP, A_DOWN, A_LEFT, A_RIGHT):
            dr, dc = {A_UP: (-1, 0), A_DOWN: (1, 0), A_LEFT: (0, -1), A_RIGHT: (0, 1)}[a]
            nr, nc = min(max(r + dr, 0), G - 1), min(max(c + dc, 0), G - 1)
            if (nr, nc) != self.pos:
                self.pos = (nr, nc)
                self.open_drawer = -1  # auto-close
        else:
            kind, i = self.fixture_at(self.pos)
            if a == A_INTERACT:
                if kind == "drawer" and self.open_drawer != i:
                    self.open_drawer = i
                    self.visited_drawers.add(i)
                    if self.drawer_contents[i] >= 0:
                        self.events.append(Event(EV_DRAWER, i, self.drawer_contents[i]))
                    else:
                        self.events.append(Event(EV_DRAWER_EMPTY, i, 7))
                elif kind == "button":
                    self.lit = i
                    self.pressed.append(i)
                    self.events.append(Event(EV_PRESSED, i, len(self.pressed) if len(self.pressed) < N_ARG else N_ARG - 1))
            elif a == A_PICK and self.holding == -2:
                if kind == "drawer" and self.open_drawer == i and self.drawer_contents[i] >= 0:
                    self.holding = self.drawer_contents[i]
                    self.drawer_contents[i] = -1
                    self.events.append(Event(EV_PICKED, self.holding, 0))
                elif kind == "dispenser":
                    self.holding = -1
                    self.events.append(Event(EV_PICKED, 7, 0))
                elif kind == "cabinet" and self.cabinet_open and self.cabinet_content >= 0:
                    self.holding = self.cabinet_content
                    self.cabinet_content = -2
                    self.L.cabinet_content = -2
                    self.events.append(Event(EV_PICKED, self.holding, 1))
            elif a == A_PLACE and self.holding != -2:
                if kind == "goal":
                    self.delivered = self.holding
                    self.holding = -2
                    self.events.append(Event(EV_DELIVERED, max(self.delivered, 0) if self.delivered >= 0 else 7, 0))
                elif kind == "bin":
                    self.bin_count += 1
                    self.holding = -2
                    self.events.append(Event(EV_PLACED_BIN, min(self.bin_count, N_ARG - 1), 0))
                elif kind == "drawer" and self.open_drawer == i and self.drawer_contents[i] == -1 and self.holding >= 0:
                    self.drawer_contents[i] = self.holding
                    self.holding = -2
            elif A_STRAT0 <= a < A_STRAT0 + N_STRATS:
                k = a - A_STRAT0
                if kind == "cabinet" and not self.cabinet_open:
                    if k == self.L.cabinet_latch:
                        self.cabinet_open = True
                        self.events.append(Event(EV_STRAT_OK, k, 0))
                    else:
                        self.stuck = True
                        self.lock = 2
                        self.failed_strats.add(k)
                        self.events.append(Event(EV_STRAT_FAIL, k, 0))
        self._check_done()
        return self.obs(), self.done

    def _check_done(self):
        T = self.task
        if T.name == "find":
            if self.delivered != -2 and self.delivered == T.target_color:
                self.success = True
            elif self.delivered != -2:
                self.success = False; self.done = True
        elif T.name == "count":
            if self.bin_count == T.count_k and self.holding == -2 and self.pos == self.L.bin and self.t > 0 \
                    and self.events and self.events[-1].type == EV_PLACED_BIN:
                self.success = True
            if self.bin_count > T.count_k:
                self.done = True
        elif T.name == "sequence":
            if len(self.pressed) >= 1 and tuple(self.pressed) != tuple(T.seq[: len(self.pressed)]):
                self.done = True
            if tuple(self.pressed) == tuple(T.seq):
                self.success = True
        elif T.name == "adapt":
            if self.delivered != -2 and self.delivered == self.cabinet_content_at_start:
                self.success = True
            elif self.delivered != -2:
                self.done = True
        elif T.name == "tidy":
            if len(self.visited_drawers) == N_DRAWERS:
                self.success = True
        if self.success or self.t >= T.max_steps:
            self.done = True

    # ------------------------------------------------------------------ observation
    def proprio(self) -> np.ndarray:
        v = np.zeros(PROPRIO_DIM, dtype=np.float32)
        v[0] = self.pos[0] / (G - 1)
        v[1] = self.pos[1] / (G - 1)
        if self.holding >= 0:
            v[2 + self.holding] = 1.0
        if self.holding == -1:
            v[2 + N_COLORS] = 1.0
        if self.holding != -2:
            v[2 + N_COLORS + 1] = 1.0
        return v

    def obs(self):
        return {"img": self.render(), "proprio": self.proprio(), "instr": self.task.instr_vec(),
                "events": list(self.events)}

    def render_state(self) -> tuple:
        return (self.pos, self.holding, self.open_drawer, self.cabinet_open, self.cabinet_content,
                tuple(self.drawer_contents), self.lit, self.stuck, self.t)

    def render(self) -> np.ndarray:
        return render_from(self.L, self.task, self.render_state())


def render_from(L: Layout, task: TaskSpec, st: tuple) -> np.ndarray:
    pos, holding, open_drawer, cabinet_open, cabinet_content, drawer_contents, lit, stuck, t = st
    img = np.empty((IMG, IMG, 3), dtype=np.uint8)
    img[:] = C_BG
    if True:

        def fill(pos, color, inset=0):
            r, c = pos
            y, x = r * CELL + inset, c * CELL + inset
            img[y: y + CELL - 2 * inset, x: x + CELL - 2 * inset] = color

        for i, d in enumerate(L.drawers):
            fill(d, C_DRAWER_OPEN if open_drawer == i else C_DRAWER)
            if open_drawer == i and drawer_contents[i] >= 0:
                fill(d, PALETTE[drawer_contents[i]], inset=2)
        fill(L.goal, C_GOAL)
        fill(L.bin, C_BIN)
        fill(L.dispenser, C_DISP)
        fill(L.cue, C_CUE)
        if t < CUE_STEPS:
            _draw_cue(img, L, task)
        for i, b in enumerate(L.buttons):
            fill(b, C_BUTTON)
            fill(b, BUTTON_COLORS[i] if lit == i else BUTTON_COLORS[i] // 2, inset=2)
        fill(L.cabinet, C_STUCK if stuck else C_CABINET)
        if cabinet_open:
            fill(L.cabinet, C_DRAWER_OPEN, inset=1)
            if cabinet_content >= 0:
                fill(L.cabinet, PALETTE[cabinet_content], inset=2)
        # agent: outline square; held object as centre dot
        r, c = pos
        y, x = r * CELL, c * CELL
        img[y: y + CELL, x] = C_AGENT; img[y: y + CELL, x + CELL - 1] = C_AGENT
        img[y, x: x + CELL] = C_AGENT; img[y + CELL - 1, x: x + CELL] = C_AGENT
        if holding >= 0:
            img[y + 2: y + CELL - 2, x + 2: x + CELL - 2] = PALETTE[holding]
        elif holding == -1:
            img[y + 2: y + CELL - 2, x + 2: x + CELL - 2] = C_DISP // 2
        return img


def _draw_cue(img, L, T):
        r, c = L.cue
        y, x = r * CELL, c * CELL
        if T.name == "count":
            # k dots in a 2x2 arrangement
            spots = [(1, 1), (1, 4), (4, 1), (4, 4)][: T.count_k]
            for dy, dx in spots:
                img[y + dy: y + dy + 2, x + dx: x + dx + 2] = C_AGENT
        elif T.name == "sequence":
            for j, b in enumerate(T.seq):
                img[y + 1: y + 6, x + 1 + 2 * j: x + 2 + 2 * j] = BUTTON_COLORS[b]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
