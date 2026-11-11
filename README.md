# Blame the Bowl, Not the Goldfish: Robot Memory Interference Is a Budget-Allocation Problem

Han Lee, Stanford University. Paper: [`paper/blame_the_bowl.pdf`](paper/blame_the_bowl.pdf).

Released memory-augmented π0.5 policies (RoboMME) lose most of their success when the demonstration they need is
buried under *k* unrelated sessions. This repository shows that the policy has not forgotten how to act: its 32-frame
memory budget is filled by even sampling over a growing buffer, so the lesson's share of the budget shrinks with *k*.
Filling the same 32 slots from the lesson session, on the same frozen policy, restores the no-interference success on
every task tested. The code here is the memory manager that does this on the policy server, the evaluator (including
the hard condition with same-scene distractors), the routers, the fine-tuning study, every per-episode result file, and
the scripts that produce every figure, table and statistic in the paper.

## Layout

| path | contents |
|---|---|
| `hmrobo/mm_policy.py` | the memory manager: all memory systems (`uniform`, `recency`, `two_scale`, `lesson_oracle[_reindex]`, `retrieval[_reindex]`, `learned_route_reindex`, `text_route_reindex`, ...) and positional re-indexing |
| `hmrobo/serve_mm.py` | patched policy server that lets the manager choose the 32 frames |
| `hmrobo/eval_cross.py` | closed-loop evaluation on RoboMME-Interference; `--same_scene M` is the hard condition |
| `hmrobo/gen_same_scene.py` | records the same-scene distractor sessions (replays of other-task motions in the query scene) |
| `hmrobo/router.py`, `hmrobo/train_router.py` | label-trained session router and its training (in-task, cross-task, leave-one-task-out) |
| `hmrobo/siglip_feats.py`, `scripts/textroute_offline.py` | SigLIP features; offline study and gate-margin sweep of the label-free instruction router |
| `hmrobo/vlm_verify.py` | Qwen2.5-VL verifier (appendix) |
| `hmrobo/build_ft_data.py`, `hmrobo/ft_dataset.py`, `hmrobo/train_ft.py` | fine-tuning study: data build, the three training distributions (`control`, `shift`, `interf`), training entry point |
| `hmrobo/selector.py`, `hmrobo/train_mm.py` | the demonstration-loss GRPO selector (negative result, Sec. 5.5) |
| `modal_robomme.py` | Modal app that runs all of the above (deploy, then spawn functions); `modal_app.py` is the Pantry app |
| `src/hm/` | Pantry grid world, student, expert, memory writers (Sec. 4) |
| `results/robomme/*.json` | one row per evaluated episode: `task, split, system, k, same_scene, episode, success, steps, lesson_frac, same_frac, ...` |
| `results/robomme/ft/` | fine-tuning study evaluations |
| `results/pantry/` | Pantry sweeps, distortion tables and report |
| `paper/` | LaTeX source, figures, compiled PDF |

## Reproducing the paper from the result files (no GPU)

```bash
pip install numpy scipy matplotlib
python scripts/paired_tests.py                         # every paired McNemar / Fisher p-value quoted in the paper
python scripts/plot_headline.py  paper/fig/headline.png   # Figure 1
python scripts/plot_schematic.py paper/fig/schematic.png  # Figure 2
python scripts/plot_mechanism.py paper/fig/mechanism.png  # Figure 3
python scripts/plot_hard_routing.py results/robomme/hard_routing.png \
  "MoveCube|results/robomme/MoveCube_ss_v1.json|-|results/robomme/MoveCube_ss_v1_loto3b.json|results/robomme/MoveCube_ss_v1_text03.json" \
  "RouteStick|results/robomme/RouteStick_ss_v1.json|results/robomme/RouteStick_ss_v1_inrouter.json|results/robomme/RouteStick_ss_v1_loto.json|results/robomme/RouteStick_ss_v1_text03.json" \
  "PatternLock|results/robomme/PatternLock_ss_v1.json+results/robomme/PatternLock_v1.json|-|-|results/robomme/PatternLock_ss_v1_text03.json"
python scripts/plot_robomme.py                          # success-vs-k curves (appendix)
```

Result files behind each table: Table 1 (fine-tuning) `results/robomme/ft/{control,shift,interf}4k_*.json` and
`*_MoveCube_val_k15.json`; appendix k-sweep table `results/robomme/{MoveCube,PatternLock,RouteStick,MoveCube_expert}_v1.json`;
development tasks `VideoRepick_v3.json`, `VideoRepick_reindex.json`, `VideoPlaceButton_v3.json`, `VideoPlaceButton_same.json`;
hard condition `*_ss_v1*.json`, `*_ss_val.json` (val split), `*_text03.json` (instruction router at gate margin .03);
standard-protocol instruction router `*_text_k7.json`; Pantry `results/pantry/sweep_final.json`, `distortion_final.json`.

## Running the closed-loop evaluations

The evaluations need the RoboMME benchmark and its released policy-learning suite and checkpoints
(`perceptual-framesamp-modul/79999`, `perceptual-framesamp-expert`), which are distributed by the RoboMME authors, plus a
GPU. `modal_robomme.py` builds the image, mounts a volume with the datasets and checkpoints, and exposes the functions;
after `modal deploy modal_robomme.py`:

```python
import modal
ev = modal.Function.from_name("robomme", "eval_cross")
# standard protocol, five systems, k = 0/7/15, 50 episodes
ev.spawn("--task MoveCube --k 0 7 15 --systems uniform retrieval retrieval_reindex lesson_oracle_reindex text_route_reindex "
         "--episodes 50 --out /vol/eval_cross/MoveCube.json", "perceptual-framesamp-modul", "", "", "TEXT_GATE_MARGIN=0.03")
# hard condition: 3 of the 7 distractors recorded in the query scene (caches from gen_same_scene)
ev.spawn("--task RouteStick --k 7 --same_scene 3 --systems uniform retrieval_reindex lesson_oracle_reindex text_route_reindex "
         "--episodes 50 --out /vol/eval_cross/RouteStick_ss.json", "perceptual-framesamp-modul", "", "", "TEXT_GATE_MARGIN=0.03")
```

Fine-tuning (Sec. 5.7): `build_ft_data` builds per-step samples and SigLIP features from the benchmark datasets,
`pack_ftdata` packs them into tar shards, and `train_ft(exp, mode)` with `mode` in `control | shift | interf` runs the
released training recipe through `hmrobo/train_ft.py` on one H100 (about 55 minutes for 4000 steps). A fine-tuned
checkpoint is evaluated by passing `ft-<mode>-v1/3999` as the checkpoint argument of `eval_cross`.

## Pantry (Sec. 4)

```bash
pip install -e .            # installs the `hm` package
bash scripts/pipeline_v6.sh # student, expert, writers, sweeps (see the script for stages)
python scripts/plot_sweep.py; python scripts/plot_distortion.py
```

## Citation

```
@article{lee2026bowl,
  title  = {Blame the Bowl, Not the Goldfish: Robot Memory Interference Is a Budget-Allocation Problem},
  author = {Lee, Han},
  year   = {2026}
}
```

MIT license.
