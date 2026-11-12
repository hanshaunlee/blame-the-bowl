"""Memory manager wrapped around the released MME-VLA policy (runs inside the policy-learning venv).

Patches the served policy so that (a) `add_buffer` may place frames at explicit step indices,
(b) `infer` may take `memory_system` / `memory_indices` and then feeds the frozen FrameSamp model
exactly those frames (with their true positional embeddings) instead of its even sampling, and
(c) the model's flow-matching loss on a demonstration action chunk can be evaluated for a given
memory, which is the training reward for the learned selector.

Memory systems (all fill the same 32-frame budget; the buffer holds every frame of every session):
  uniform        released policy's default: 32 frames evenly over the whole buffer
  recency        last 32 frames
  exec_only      evenly over the current episode only (no long-term memory)
  lesson_oracle  evenly over [lesson session + current episode] (knows which session matters)
  lesson_late    lesson slots from the last 40% of the lesson video, current-episode slots as default
  two_scale      half budget = dense recent window, half = lesson evenly (MEM-style two-scale)
  retrieval      session whose frames are most similar to the current view (VLA SigLIP features), + current episode
  text_route     label-free, instruction-conditioned: among the sessions whose frames match the current scene (within .05 of the best
                 scene similarity), the one whose frames best match the instruction in a joint image-text space (HF SigLIP); decided once per episode
  learned        FrameSelector over all frames
Controls (tranche 7, see results/robomme/NOTES.md):
  scene_route      text_route's encoder, scene score and one-decision-per-episode commitment, without the instruction (argmax scene)
  text_route_shuf  text_route with the instruction of another task family (fixed per episode)
  retrieval_once   retrieval (policy features), decided once per episode
  alloc<b>[r]      dose-response: exactly b lesson slots (even; r = a random subset fixed per episode), 16 current-episode slots (even),
                   the rest evenly over the distractor sessions; true positions
  comb_dense, comb_stride<s>  the released sampler's lesson frames and slot counts at every step; only the layout of the
                   current-episode slots changes (last m frames / last m frames at stride s)
  <sys>_allshift<N>  same frames, every position moved later by N steps (lesson-to-episode gap preserved)
"""
from __future__ import annotations
import types
import os
import numpy as np, jax, jax.numpy as jnp, torch

from mme_vla_suite.shared.data_utils import even_sampling_indices
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from openpi.shared import nnx_utils
from hmrobo.selector import FrameSelector, pl_sample, pos_features, even_over, MEM
from hmrobo.router import SessionRouter, route

SPATIAL = "4x4"   # FrameSamp-Modul: 16 tokens per frame
OTHER_INSTRUCTIONS = [   # RoboMME task goals (prefix stripped), for the wrong-instruction control
    "move the cube to the target in the same manner as before",
    "use the stick attached to the robot to navigate around the sticks on the table, following the same path",
    "use the stick attached to the robot to retrace the same pattern",
    "grasp the same end of the same peg you've picked before and insert it into the same side of the box",
    "pick up the same block that was previously picked up again, finally put it down and press the button to stop",
    "place the blue cube on the target right before the button was pressed",
    "pick up the container hiding the blue cube",
]


def _feat(f):
    return np.asarray(f["image_emb_2x2"]).astype(np.float32).mean((0, 1))  # (2048,)


class MemoryManager:
    def __init__(self, pol, selector_path: str | None = None, router_path: str | None = None):
        self.pol = pol
        self.sel = None
        self.router = None
        if selector_path:
            self.load_selector(selector_path)
        if router_path:
            ck = torch.load(router_path, map_location="cpu")
            self.router = SessionRouter(**ck.get("cfg", {})).eval()
            self.router.load_state_dict(ck["state"])
        self._loss_fn = None
        self._orig_infer = type(pol).infer
        pol.add_buffer = self.add_buffer
        pol.infer = self.infer
        self.last_indices = None
        self.last_session = None
        self.raw = {}            # step idx -> raw front frame (uint8), for instruction-conditioned routing
        self._prompt = None
        self._text_cache = {}
        self._siglip = None

    # ------------------------------------------------------------ selector
    def load_selector(self, path):
        ck = torch.load(path, map_location="cpu")
        self.sel = FrameSelector(**ck.get("cfg", {})).eval()
        self.sel.load_state_dict(ck["state"])

    # ------------------------------------------------------------ buffer
    def add_buffer(self, obs):
        pol = self.pol
        if pol.mem_buffer is None:
            return
        images, states = obs["images"], obs["state"]
        if obs.get("exec_start_idx", 0) > 0:
            pol.exec_start_idx = int(obs["exec_start_idx"])
        sil = obs.get("step_idx_list")
        sil = [int(i) for i in sil] if sil is not None else list(range(pol.step_idx + 1, pol.step_idx + 1 + len(images)))
        pol.mem_buffer.add_buffer(images, states, sil)
        pol.step_idx = max(pol.step_idx, max(sil))
        if min(sil) <= 1 and self.raw:    # new episode: forget the previous buffer's frames and routing decision
            self.raw = {}; self._text_cache = {}
        for i, im in zip(sil, images):
            im = np.asarray(im)
            while im.ndim > 3:        # frames arrive as (n, views, H, W, 3): keep the front view
                im = im[0]
            if im.dtype != np.uint8:
                im = (im * 255 if im.max() <= 1.0 else im).clip(0, 255).astype(np.uint8)
            self.raw[int(i)] = im[..., :3]

    def features(self, idxs):
        hf = self.pol.mem_buffer._history_feats
        return np.stack([_feat(hf[i]) for i in idxs])

    # ------------------------------------------------------------ selection
    def select(self, system, T, H=None, lesson_bounds=None, session_bounds=None, greedy=True, F=None):
        """Memory indices for the current step T (frames 0..T are in the buffer; current episode = [H, T])."""
        allf = list(range(T + 1))
        if system in (None, "uniform"):
            return list(even_sampling_indices(T, MEM))
        if system == "recency":
            return allf[-MEM:]
        ex = list(range(int(H), T + 1))
        if system == "exec_only":
            return even_over(ex)
        if system == "lesson_oracle":
            a, b = map(int, lesson_bounds)
            self.last_session = (a, b)
            return even_over(list(range(a, b)) + ex)
        if system == "lesson_late":   # lesson slots from the last 40% of the lesson video (the demonstrated grasp), exec slots as default
            a, b = map(int, lesson_bounds)
            ex_d = [i for i in even_sampling_indices(T, MEM) if i >= int(H)]
            m_h = MEM - len(ex_d)
            late = list(range(a + int(0.6 * (b - a)), b)) or list(range(a, b))
            return sorted(even_over(late, m_h) + ex_d)
        if system == "two_scale":     # MEM-style: half the budget is a dense recent window, half is the lesson evenly
            a, b = map(int, lesson_bounds)
            w = ex[-(MEM // 2):]
            return sorted(even_over(list(range(a, b)), MEM - len(w)) + w)
        if system.startswith("alloc"):   # dose-response: b lesson slots, EX current-episode slots, the rest over the distractors
            a, b = map(int, lesson_bounds)
            self.last_session = (a, b)   # alloc<...>_reindex: lesson and current-episode positions compacted; distractor frames keep theirs
            import re
            if system.startswith("allocp"):   # allocp<f>: retrieval's own even comb over lesson + episode; only f% of its lesson slots kept, the freed slots go to the distractors
                ref = even_over(list(range(a, b)) + ex)
                l_ref, e_ref = [i for i in ref if a <= i < b], [i for i in ref if not a <= i < b]
                nb = round(int(system[6:]) / 100 * len(l_ref))
                l_sel = even_over(l_ref, nb) if nb > 0 else []
                others = [i for i in range(int(H)) if not a <= i < b]
                n_o = MEM - len(l_sel) - len(e_ref)
                return sorted(l_sel + e_ref + (even_over(others, n_o) if n_o > 0 else []))
            m_ = re.fullmatch(r"alloc(\d+)(r?)(?:e(\d+))?", system)   # alloc<b>[r][e<EX>]; EX defaults to half the budget
            rand, nb, EX = bool(m_.group(2)), int(m_.group(1)), int(m_.group(3) or MEM // 2)
            les = list(range(a, b))
            if rand:
                import random
                l_sel = sorted(random.Random(7 * a + 13 * b + int(H)).sample(les, min(nb, len(les))))
            else:
                l_sel = even_over(les, nb) if nb > 0 else []
            e_sel = even_over(ex, EX)
            others = [i for i in range(int(H)) if not a <= i < b]
            n_o = MEM - len(l_sel) - len(e_sel)
            return sorted(l_sel + e_sel + (even_over(others, n_o) if n_o > 0 else []))
        if system.startswith("comb_"):   # released sampler's history frames and counts; current-episode slots re-laid out
            default = [int(i) for i in even_sampling_indices(T, MEM)]
            hist = [i for i in default if i < int(H)]
            m_e = len(default) - len(hist)
            stride = 1 if system == "comb_dense" else int(system[len("comb_stride"):])
            e_sel = ex[::-1][::stride][:m_e]
            e_sel += [i for i in ex[::-1] if i not in set(e_sel)][:m_e - len(e_sel)]   # early in the episode: fill with the most recent
            return sorted(hist + e_sel)
        if system in ("retrieval", "retrieval_once", "learned", "learned_route", "text_route", "scene_route", "text_route_shuf") and F is None:
            F = self.features(allf)
        if system in ("text_route", "scene_route", "text_route_shuf"):
            a, b = self.text_route(F, session_bounds, T, mode={"text_route": "text", "scene_route": "scene", "text_route_shuf": "shuf"}[system])
            self.last_session = (a, b)
            return even_over(list(range(a, b)) + ex)
        if system == "learned_route":   # learned session router (trained with labels on the train split), label-free here
            a, b = map(int, session_bounds[route(self.router, F, session_bounds, T)])
            self.last_session = (a, b)
            return even_over(list(range(a, b)) + ex)
        if system in ("retrieval", "retrieval_once"):
            key = ("retrieval_once", tuple((int(a), int(b)) for a, b in session_bounds))
            if system == "retrieval_once" and key in self._text_cache:
                self.last_session = self._text_cache[key]
                return even_over(list(range(*self.last_session)) + ex)
            f = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
            c = f[T]
            best, bs = None, -2.0
            for a, b in session_bounds:
                a, b = int(a), int(b)
                s = float((f[a:b] @ c).max()) if b > a else -2.0
                if s > bs:
                    best, bs = (a, b), s
            self.last_session = best
            if system == "retrieval_once":
                self._text_cache[key] = best
                print(f"retrieval_once -> session {[list(map(int, x)) for x in session_bounds].index(list(best))}", flush=True)
            return even_over(list(range(best[0], best[1])) + ex)
        if system == "learned":
            idxs, _ = self.learned(F, T, int(H), greedy=greedy, G=1)
            return [int(i) for i in idxs[0].tolist()]
        raise ValueError(system)

    # ------------------------------------------------------------ instruction-conditioned routing (label-free)
    def siglip(self):
        if self._siglip is None:
            from transformers import AutoModel, AutoProcessor
            name = "google/siglip-so400m-patch14-384"
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            m = AutoModel.from_pretrained(name, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
            self._siglip = (m, AutoProcessor.from_pretrained(name), dev)
            print("text_route: loaded", name, "on", dev, flush=True)
        return self._siglip

    @staticmethod
    def _vec(e):
        if torch.is_tensor(e):
            return e
        keys = list(e.keys()) if hasattr(e, "keys") else []
        for k in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
            v = e[k] if k in keys else getattr(e, k, None)
            if torch.is_tensor(v):
                return v.mean(1) if v.dim() == 3 else v
        return [v for v in e.values() if torch.is_tensor(v)][0]

    @torch.no_grad()
    def text_route(self, F, session_bounds, T, max_frames=32, mode="text"):
        """Scene gate, then instruction match, both in SigLIP's joint space (as the offline study): a session is a scene
        candidate if the best cosine between one of its frames and the current view is within .05 of the best session's;
        among candidates pick the session whose frames best match the instruction. Cached per episode (one decision).
        v0 of this function gated on the policy's own features, whose max-over-frames cosine spans <.003 across sessions,
        so the gate admitted every session; see NOTES (text_route v0)."""
        import re
        from PIL import Image
        key = (mode, tuple((int(a), int(b)) for a, b in session_bounds), self._prompt)
        if key in self._text_cache:
            return self._text_cache[key]
        m, proc, dev = self.siglip()

        def embed(idx):
            ims = [Image.fromarray(self.raw[i]) for i in idx]
            px = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev, dtype=m.dtype)
            return torch.nn.functional.normalize(self._vec(m.get_image_features(pixel_values=px)).float(), dim=-1)

        instr = re.sub(r"^\s*watch the video carefully,?\s*then\s*", "", str(self._prompt or ""), flags=re.I).strip()
        if mode == "shuf":   # control: the instruction of another task family, fixed per episode
            import random
            wrong = [w for w in OTHER_INSTRUCTIONS if w != instr]
            instr = random.Random(sum(int(b) for _, b in session_bounds)).choice(wrong)
        tok = proc(text=[instr], padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(dev)
        t = torch.nn.functional.normalize(self._vec(m.get_text_features(**tok)).float(), dim=-1)[0]
        cur = [i for i in self.raw if i <= T]
        q = embed([max(cur)])[0] if cur else None
        scene = np.full(len(session_bounds), -2.0); text = np.full(len(session_bounds), -9.0)
        for j, (a, b) in enumerate(session_bounds):
            a, b = int(a), int(b)
            idx = [i for i in np.linspace(a, b - 1, min(max_frames, max(b - a, 1))).astype(int) if i in self.raw] if b > a else []
            if not idx:
                continue
            e = embed(idx)
            text[j] = float((e @ t).max())
            scene[j] = float((e @ q).max()) if q is not None else 0.0
        margin = float(os.environ.get("TEXT_GATE_MARGIN", "0.03"))   # chosen on the train split (offline sweep: .02-.03 best; .05 admits other-task sessions on sparse scenes)
        gate = scene >= scene.max() - margin
        cand = np.where(gate, text, -9.0)
        best = int(np.argmax(cand)) if cand.max() > -9 else int(np.argmax(scene))
        if mode == "scene":   # control: same encoder and commitment, no instruction
            best = int(np.argmax(scene))
        print(f"text_route[{mode}](m={margin}): '{instr[:50]}' scene {np.round(scene, 3).tolist()} gate {gate.astype(int).tolist()} text {np.round(text, 3).tolist()} -> session {best}", flush=True)
        self._text_cache[key] = (int(session_bounds[best][0]), int(session_bounds[best][1]))
        return self._text_cache[key]

    def learned(self, F, T, H, greedy=False, G=1):
        """Structured selection: the current episode keeps exactly the frames the released policy's even
        sampling would give it (working memory, identical to the baseline); the remaining slots are filled
        from the history [0,H) by the learned scorer (Plackett-Luce; greedy -> argmax). Returns (idxs, logp)."""
        n = T + 1
        default = list(even_sampling_indices(T, MEM))
        ex = [i for i in default if i >= H]
        m_h = MEM - len(ex)
        if H <= 0 or m_h <= 0:
            return [torch.tensor(default, dtype=torch.long) for _ in range(G)], torch.zeros(G)
        emb = torch.from_numpy(np.ascontiguousarray(F[:n])).float().unsqueeze(0).expand(G, -1, -1)
        pos = torch.from_numpy(pos_features(n, H)).unsqueeze(0).expand(G, -1, -1)
        cur = emb[:, T]
        mask = torch.zeros(G, n, dtype=torch.bool); mask[:, :H] = True
        scores = self.sel(emb, pos, cur, mask)
        idx_h, logp = pl_sample(scores, min(m_h, H), greedy=greedy)
        ex_t = torch.tensor(ex, dtype=torch.long)
        return [torch.sort(torch.cat([i, ex_t])).values for i in idx_h], logp

    # ------------------------------------------------------------ inference with a chosen memory
    def _with_memory(self, mi, fn, remap: dict | None = None):
        """Run fn() with the memory frames `mi`; optionally with the selected frames' time positions
        remapped (remap: frame idx -> new time idx), restoring the buffer afterwards."""
        mb = self.pol.mem_buffer
        mi = [int(i) for i in mi] or [self.pol.step_idx]
        orig = mb.get_frame_sampling_indices
        mb.get_frame_sampling_indices = lambda step_idx, tb, tpi: [int(i) for i in mi]
        saved = {}
        if remap:
            pe = mb.pos_emb_dict[SPATIAL]
            for i in mi:
                if i in remap and i in mb._history_feats:
                    saved[i] = mb._history_feats[i][f"pos_emb_{SPATIAL}"]
                    mb._history_feats[i][f"pos_emb_{SPATIAL}"] = pe[remap[i]:remap[i] + 1]
        try:
            return fn()
        finally:
            mb.get_frame_sampling_indices = orig
            for i, v in saved.items():
                mb._history_feats[i][f"pos_emb_{SPATIAL}"] = v

    @staticmethod
    def compact_remap(session, H, mi):
        """Time positions as if the chosen session were immediately followed by the current episode
        (the layout the policy was trained on): session frames -> 0.., current-episode frames -> D + (i - H)."""
        a, b = map(int, session)
        D = b - a
        return {int(i): (int(i) - a if a <= int(i) < b else D + int(i) - int(H)) for i in mi if a <= int(i) < b or int(i) >= int(H)}

    def infer(self, obs):
        obs = dict(obs)
        mi = obs.pop("memory_indices", None)
        system = obs.pop("memory_system", None)
        H = obs.pop("phase_start", None); lb = obs.pop("lesson_bounds", None); sb = obs.pop("session_bounds", None)
        self._prompt = obs.get("prompt", self._prompt)
        remap = None
        if mi is None and system is not None:
            shift = allshift = 0
            if "_allshift" in system:   # e.g. uniform_allshift480: same frames, every position moved later by N (gap preserved)
                system, allshift = system.split("_allshift")[0], int(system.split("_allshift")[1])
            if "_shift" in system:   # e.g. uniform_shift224: same frames, current-episode positions moved later by N steps
                system, shift = system.split("_shift")[0], int(system.split("_shift")[1])
            base = system[:-len("_reindex")] if system.endswith("_reindex") else system
            self.last_session = None
            mi = self.select(base, self.pol.step_idx, H, lb, sb)
            if base != system and self.last_session is not None:
                remap = self.compact_remap(self.last_session, H, mi)
            if shift:
                remap = {int(i): int(i) + shift for i in mi if int(i) >= int(H)}
            if allshift:
                remap = {int(i): int(i) + allshift for i in mi}
        out = self._with_memory(mi, lambda: self._orig_infer(self.pol, obs), remap)
        if mi is not None:
            out["memory_indices"] = np.asarray(mi, dtype=np.int32)
        self.last_indices = mi
        return out

    # ------------------------------------------------------------ reward: the VLA's own loss
    def flow_loss(self, obs, actions, mi, seed: int, n_mc: int = 4) -> float:
        """Flow-matching loss of the frozen model on `actions` (ah x 8, absolute joints+gripper) at `obs`
        with memory `mi`; common random numbers via `seed` so candidates are compared on the same noise."""
        pol = self.pol
        inputs = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
        inputs["actions"] = np.array(actions, dtype=np.float32, copy=True)  # DeltaActions edits in place
        x = self._with_memory(mi, lambda: pol._input_transform(pol._prepare_history(inputs)))
        acts = x.pop("actions")
        batch = jax.tree.map(lambda v: jnp.repeat(jnp.asarray(v)[None], n_mc, axis=0), x)
        observation = HistAugObservation.from_dict(batch)
        acts = jnp.repeat(jnp.asarray(acts)[None], n_mc, axis=0)
        if self._loss_fn is None:
            self._loss_fn = nnx_utils.module_jit(pol._model.compute_loss)
        l, _ = self._loss_fn(jax.random.key(seed), observation, acts)
        return float(jnp.mean(l))


# ---------------------------------------------------------------- offline buffer construction (training)
def encode_frames(pol, front: np.ndarray, states: np.ndarray, chunk: int = 64) -> list[dict]:
    """Encode frames with the policy's own vision encoder (fixed chunk shape -> one jit); returns per-frame dicts."""
    mb = pol.mem_buffer
    out = []
    for s in range(0, len(front), chunk):
        imgs, sts = front[s:s + chunk], states[s:s + chunk]
        n = len(imgs)
        if n < chunk:
            imgs = np.concatenate([imgs, np.repeat(imgs[-1:], chunk - n, 0)]); sts = np.concatenate([sts, np.repeat(sts[-1:], chunk - n, 0)])
        mb.clear()
        mb.add_buffer(imgs.astype(np.uint8)[:, None], sts.astype(np.float32), list(range(chunk)))
        for i in range(n):
            f = mb._history_feats[i]
            out.append({"image_emb_4x4": f["image_emb_4x4"], "image_emb_2x2": f["image_emb_2x2"], "state": np.asarray(sts[i], np.float32)})
    mb.clear()
    return out


def set_buffer(pol, frames: list[dict], exec_start: int):
    """Place encoded frames at positions 0..n-1 (true positional embeddings), current episode starting at exec_start."""
    mb = pol.mem_buffer
    mb.clear()
    pe = mb.pos_emb_dict[SPATIAL]
    for step, f in enumerate(frames):
        mb._history_feats[step] = {"image_emb_4x4": f["image_emb_4x4"], "image_emb_2x2": f["image_emb_2x2"],
                                   f"pos_emb_{SPATIAL}": pe[step:step + 1], "state_emb": f["state"]}
    pol.step_idx = len(frames) - 1
    pol.exec_start_idx = int(exec_start)
