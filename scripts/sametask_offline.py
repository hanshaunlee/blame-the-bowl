"""Offline routing against same-task, same-scene competitors (hmrobo/gen_same_scene.py --pool same; siglip_feats.py).
Sessions: the lesson video and three replays of *other recorded episodes of the same task* in the query scene.
    python scripts/sametask_offline.py <dir with <task>_test.npz>"""
import sys, glob, os, numpy as np
f32 = lambda x: np.asarray(x, dtype=np.float32)
for p in sorted(glob.glob(f"{sys.argv[1]}/*_test.npz")):
    eps = np.load(p, allow_pickle=True)["eps"].item(); R = []
    for ep, e in sorted(eps.items()):
        S = [f32(e["lesson"])] + [f32(x) for x in e["sessions"]]; q, t = f32(e["exec"]), f32(e["text_stripped"])
        scene = np.array([(F @ q).max() for F in S]); text = np.array([(F @ t).max() for F in S]); tmean = np.array([(F @ t).mean() for F in S])
        gate = scene >= scene.max() - .03
        R.append((np.argmax(scene) == 0, np.argmax(text) == 0, np.argmax(np.where(gate, text, -9)) == 0, np.argmax(tmean) == 0, gate.sum(), text[0] - text[1:].max()))
    R = np.array(R, dtype=float)
    print(f"{os.path.basename(p):24s} n={len(R)} scene-only {R[:,0].mean():.2f}  instr-only {R[:,1].mean():.2f}  gate+instr {R[:,2].mean():.2f}  instr-mean {R[:,3].mean():.2f}  "
          f"gate size {R[:,4].mean():.1f}  lesson-minus-best-competitor instr score {R[:,5].mean():+.3f} (chance .25)")
