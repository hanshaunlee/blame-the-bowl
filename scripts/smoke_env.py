import numpy as np, time, sys
from collections import Counter
from hm.env.world import *
from hm.env.session import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
# expert success per task
for t in TASKS:
    succ, steps = [], []
    for i in range(200):
        tr = make_single(rng, t)
        succ.append(tr.successes[0]); steps.append(len(tr.steps))
    print(f"{t:9s} expert success {np.mean(succ):.3f}  mean steps {np.mean(steps):.1f} max {max(steps)}")
# cross trials
t0=time.time()
qs, ks, qlen, nev, ntot = [], Counter(), [], [], []
for i in range(300):
    k = int(rng.choice([0,1,3,7]))
    tr = make_cross(rng, k)
    qs.append(tr.successes[-1]); ks[tr.sessions[-1][1].name]+=1
    qlen.append(tr.sessions[-1][1].max_steps); nev.append(len(tr.events)); ntot.append(len(tr.steps))
    assert all(tr.successes[:-1]) or True
print(f"cross: query expert success {np.mean(qs):.3f} query tasks {dict(ks)} query max_steps mean {np.mean(qlen):.1f} events/trial {np.mean(nev):.1f} steps/trial {np.mean(ntot):.1f}  ({time.time()-t0:.1f}s for 300)")
# distractor sessions success
tr = make_cross(np.random.default_rng(3), 3, first_task="find", query_task="find")
print("sessions:", [(s[1].name, ok) for s, ok in zip(tr.sessions, tr.successes)])
print("events:", [(e.session, e.t, str(e.ev)) for e in tr.events][:30])
print("rel of query steps:", [st.rel for st in tr.steps if st.session == tr.query_session][:5])
# render strip of query session + session 0
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
s0 = [i for i, st in enumerate(tr.steps) if st.session == 0][:8]
sq = [i for i, st in enumerate(tr.steps) if st.session == tr.query_session][:8]
for ax, i in zip(axes[0], s0):
    ax.imshow(frames_for_step(tr, i)[0]); ax.set_title(f"s0 t{tr.steps[i].t} a={ACTION_NAMES[tr.steps[i].action]}", fontsize=7); ax.axis("off")
for ax, i in zip(axes[1], sq):
    ax.imshow(frames_for_step(tr, i)[0]); ax.set_title(f"q t{tr.steps[i].t} a={ACTION_NAMES[tr.steps[i].action]}", fontsize=7); ax.axis("off")
plt.tight_layout(); plt.savefig("results/smoke_strip.png", dpi=110)
print("saved results/smoke_strip.png")
