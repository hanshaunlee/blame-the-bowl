"""Modal entrypoints: data generation, behaviour cloning, writer RL, and evaluation sweeps.

    modal run modal_app.py::bc --n-train 6000 --epochs 6
    modal run modal_app.py::cmd --args "-m hm.train.writer_rl --mode consolidator"
Results persist on the volume `hindsight-memory` under /vol/{data,ckpt,results}.
"""
import modal, os, subprocess, sys, shlex

app = modal.App("hindsight-memory")
vol = modal.Volume.from_name("hindsight-memory", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy", "torch==2.5.1", "tqdm", "matplotlib")
         .add_local_dir("src", "/root/proj/src", copy=True)
         .add_local_dir("scripts", "/root/proj/scripts", copy=True)
         .add_local_file("pyproject.toml", "/root/proj/pyproject.toml", copy=True)
         .run_commands("cd /root/proj && pip install -e ."))

GPU = os.environ.get("HM_GPU", "A10G")


def _run(args: str, shell: bool = False):
    os.chdir("/root/proj")
    for d in ("data", "ckpt", "results", "logs"):
        if not os.path.islink(d):
            os.makedirs(f"/vol/{d}", exist_ok=True)
            if os.path.isdir(d):
                os.rmdir(d)
            os.symlink(f"/vol/{d}", d)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    p = subprocess.run(["bash", "-c", args] if shell else [sys.executable] + shlex.split(args), env=env)
    vol.commit()
    return p.returncode


@app.function(image=image, gpu=GPU, cpu=16, memory=32768, timeout=6 * 3600, volumes={"/vol": vol})
def cmd(args: str):
    return _run(args)


@app.function(image=image, cpu=16, memory=16384, timeout=3600, volumes={"/vol": vol})
def cmd_cpu(args: str):
    return _run(args)


@app.local_entrypoint()
def bc(n_train: int = 6000, epochs: int = 6, workers: int = 14, budget: int = 8, out: str = "ckpt/policy.pt", extra: str = ""):
    cmd.remote(f"-m hm.train.bc --n_train {n_train} --epochs {epochs} --workers {workers} --B {budget} --out {out} {extra}")


@app.function(image=image, gpu=GPU, cpu=16, memory=32768, timeout=8 * 3600, volumes={"/vol": vol})
def sh(script: str):
    return _run(script, shell=True)


@app.local_entrypoint()
def pipeline(script: str):
    sh.remote(open(script).read())


@app.local_entrypoint()
def run(args: str, gpu: bool = True):
    (cmd if gpu else cmd_cpu).remote(args)
