"""Closed-loop evaluation: the frozen student acts in the world with a given memory system.

Trials are replayed from a reference trial (layouts, session tasks, query task) so that every
memory system is evaluated on identical layouts and task sequences. The student's own
trajectory generates the events that memory systems see.
"""
from __future__ import annotations
import copy, numpy as np, torch
from ..env.world import *
from ..env.session import Trial, StepRec, EventRec, W_FRAMES
from ..train.data import PLACE_DIM, AGE_DIM_, ctx_vec, layout_tokens


class LiveTrial:
    """Incrementally built trial record for one rollout, plus cached embeddings."""

    def __init__(self, ref: Trial, memsys, pol, dev):
        self.ref = ref
        self.layouts = [copy.deepcopy(L) for L in ref.init_layouts]
        self.sessions = []          # (layout_idx, task)
        self.place = ref.place
        self.steps: list[StepRec] = []
        self.events: list[EventRec] = []
        self.successes = []
        self.memsys = memsys
        self.pol, self.dev = pol, dev
        self.ev_base = {}           # ei -> {0: sym base emb, 1: frame base emb}
        self.step_frame_emb = []    # per step: (d,) embedding of that step's frame (with type/pos added later)
        memsys.reset_trial()
        self.w = None
        self.done_all = False

    # -- session management
    def start_session(self, s):
        li, task = self.ref.sessions[s]
        task = copy.deepcopy(task)
        if task.name == "adapt" and s == len(self.ref.sessions) - 1 and self.ref.kind == "cross":
            self.layouts[li].cabinet_content = self.ref.query_restock  # restocked as in the reference trial
        self.sessions.append((li, task))
        if s == len(self.ref.sessions) - 1 and self.ref.kind == "cross":
            self.slots_at_query = list(self.memsys.contents())
        self.w = World(self.layouts[li], task, np.random.default_rng(0))
        self.obs = self.w.obs()
        self.cur = s
        self.first_step = len(self.steps)
        self._ingest_events()

    def _ingest_events(self):
        s = self.cur
        for e in self.obs["events"]:
            ei = len(self.events)
            self.events.append(EventRec(s, self.w.t, len(self.steps), self.sessions[s][0], e))
            self.memsys.on_event(self, ei)

    def record_step(self, action: int):
        st = self.w
        self.steps.append(StepRec(self.cur, st.t, self.sessions[self.cur][0], st.render_state(), self.obs["proprio"],
                                  self.obs["instr"], int(action), len(self.events), ()))

    def env_step(self, action: int):
        self.obs, done = self.w.step(action)
        if done:
            # terminal events are visible to memory (e.g. delivered)
            for e in self.obs["events"]:
                ei = len(self.events)
                self.events.append(EventRec(self.cur, self.w.t, len(self.steps) - 1, self.sessions[self.cur][0], e))
                self.memsys.on_event(self, ei)
            self.successes.append(bool(self.w.success))
            self.memsys.on_session_end(self, self.cur)
        else:
            self._ingest_events()
        return done


def frame_of_state(trial: LiveTrial, li, task, state):
    return render_from(trial.layouts[li], task, state)


@torch.no_grad()
def run_batch(pol, ref_trials: list[Trial], memsys_factory, dev, deterministic=True, teacher_forcing=False, stats=None, beta=0.0):
    """Run every reference trial with a fresh memory system; returns list of LiveTrial."""
    lives = [LiveTrial(r, memsys_factory(r), pol, dev) for r in ref_trials]
    n_sessions = max(len(r.sessions) for r in ref_trials)
    d = pol.d
    for s in range(n_sessions):
        active = [lt for lt in lives if s < len(lt.ref.sessions)]
        for lt in active:
            lt.start_session(s)
        running = list(active)
        while running:
            # ---- build batch
            frames, ctxs, mem_tok, present, maps = [], [], [], [], []
            for lt in running:
                w = lt.w
                # current frame + previous frames from this session (pad by repeat)
                cur = w.render()
                prev = [lt.steps[j].state for j in range(len(lt.steps) - 1, max(lt.first_step - 1, len(lt.steps) - 1 - W_FRAMES), -1)]
                li, task = lt.sessions[lt.cur]
                fr = [cur] + [frame_of_state(lt, li, task, st) for st in prev]
                while len(fr) < W_FRAMES + 1:
                    fr.append(fr[-1])
                frames.append(np.stack(fr))
                ctxs.append(ctx_vec(lt.obs["proprio"], lt.obs["instr"], lt.place[li], lt.layouts[li]))
                maps.append(layout_tokens(lt.layouts[li]))
            frames_t = torch.from_numpy(np.stack(frames)).to(dev).permute(0, 1, 4, 2, 3).float() / 255.0
            ctx_t = torch.from_numpy(np.stack(ctxs)).to(dev)
            map_t = pol.map_tokens(torch.from_numpy(np.stack(maps)).to(dev))
            # ---- memory tokens
            B = max([lt.memsys.B for lt in running] + [getattr(pol, "slots", 8)])  # pad to the trained slot count
            tok = torch.zeros(len(running), B, d, device=dev)
            pres = torch.zeros(len(running), B, dtype=torch.bool, device=dev)
            need_sym, need_frm = [], []
            for bi, lt in enumerate(running):
                for si, (ei, m) in enumerate(lt.memsys.contents()[:B]):
                    if ei not in lt.ev_base:
                        lt.ev_base[ei] = {}
                    if m not in lt.ev_base[ei]:
                        (need_sym if m == 0 else need_frm).append((bi, ei))
            if need_sym:
                vecs = torch.from_numpy(np.stack([running[bi].events[ei].vec() for bi, ei in need_sym])).to(dev)
                emb = pol.unit_base_sym(vecs)
                for (bi, ei), e in zip(need_sym, emb):
                    running[bi].ev_base[ei][0] = e
            if need_frm:
                imgs = []
                for bi, ei in need_frm:
                    lt = running[bi]; ev = lt.events[ei]; st = lt.steps[ev.step] if ev.step < len(lt.steps) else None
                    li2, task2 = lt.sessions[ev.session]
                    state = st.state if st is not None else lt.w.render_state()
                    imgs.append(render_from(lt.layouts[li2], task2, state))
                imgs = torch.from_numpy(np.stack(imgs)).to(dev).permute(0, 3, 1, 2).float() / 255.0
                emb = pol.unit_base_frame(imgs)
                for (bi, ei), e in zip(need_frm, emb):
                    running[bi].ev_base[ei][1] = e
            ages = torch.zeros(len(running), B, AGE_DIM_ + PLACE_DIM, device=dev)
            for bi, lt in enumerate(running):
                cur_step = len(lt.steps)
                for si, (ei, m) in enumerate(lt.memsys.contents()[:B]):
                    ev = lt.events[ei]
                    tok[bi, si] = lt.ev_base[ei][m]
                    pres[bi, si] = True
                    ages[bi, si, 0] = float(ev.session == lt.cur)
                    ages[bi, si, 1] = min(lt.cur - ev.session, 8) / 8.0
                    ages[bi, si, 2] = min(cur_step - ev.step, 120) / 120.0
                    ages[bi, si, 3:] = torch.from_numpy(lt.place[ev.layout]).to(dev)
            tok = pol.unit_finish(tok, ages)
            logits = pol.forward_tokens(pol.frame_tokens(frames_t), pol.ctx_tokens(ctx_t), tok, pres, map_t)
            acts = logits.argmax(-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
            acts = acts.tolist()
            if teacher_forcing or beta > 0:
                from ..env.oracle import expert_from_state
                for bi, lt in enumerate(running):
                    if not teacher_forcing and np.random.random() >= beta:
                        continue
                    li, task = lt.sessions[lt.cur]
                    hist = [(j, e.ev) for j, e in enumerate(lt.events) if e.layout == li]
                    a_exp, _ = expert_from_state(lt.layouts[li], task, lt.w.render_state(), hist)
                    if stats is not None:
                        ok = int(acts[bi] == a_exp)
                        stats.setdefault("agree", []).append(ok)
                        stats.setdefault("by_task", {}).setdefault(task.name, []).append(ok)
                        phase = "query" if (lt.ref.kind == "cross" and lt.cur == len(lt.ref.sessions) - 1) else "other"
                        stats.setdefault("by_phase", {}).setdefault(phase, []).append(ok)
                        if phase == "query":
                            stats.setdefault("by_k", {}).setdefault(lt.ref.k, []).append(ok)
                    acts[bi] = int(a_exp)
            # ---- step
            still = []
            for lt, a in zip(running, acts):
                lt.record_step(a)
                done = lt.env_step(a)
                if not done:
                    still.append(lt)
            running = still
    return lives


def summarize(lives: list[LiveTrial]):
    """Per-task success in single trials; query success by k in cross trials."""
    out = {"single": {}, "query": {}, "distractor": {}}
    acc = {}
    for lt in lives:
        r = lt.ref
        if r.kind == "single":
            acc.setdefault(("single", r.sessions[0][1].name), []).append(lt.successes[0])
        else:
            q = len(r.sessions) - 1
            acc.setdefault(("query", r.k), []).append(lt.successes[q])
            acc.setdefault(("query_task", r.sessions[q][1].name), []).append(lt.successes[q])
            acc.setdefault(("first", r.sessions[0][1].name), []).append(lt.successes[0])
            for s in range(1, q):
                acc.setdefault(("distractor", r.sessions[s][1].name), []).append(lt.successes[s])
    return {f"{k[0]}/{k[1]}": (float(np.mean(v)), len(v)) for k, v in sorted(acc.items(), key=str)}
