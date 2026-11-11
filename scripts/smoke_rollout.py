import time, torch, numpy as np
from hm.train.data import load
from hm.train.bc import make_policy, device
from hm.eval.rollout import run_batch, summarize
from hm.eval.memory import *
if __name__ == "__main__":
    val = load("data/trials_val.pkl")[:40]
    dev = device(); pol = make_policy().to(dev).eval()
    for fac in [lambda r: NoMemory(8), lambda r: RecencySym(8), lambda r: RecencyFrame(8), lambda r: ScriptedMEM(8), lambda r: OracleMemory(8, query_task=r.sessions[-1][1] if r.kind=="cross" else None)]:
        t0=time.time(); lives = run_batch(pol, val, fac, dev); print(fac(val[0]).name, f"{time.time()-t0:.1f}s", summarize(lives))
    lt = lives[0]; print("events seen:", len(lt.events), "slots:", lt.memsys.contents(), "steps:", len(lt.steps))
