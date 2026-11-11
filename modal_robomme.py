"""RoboMME on Modal: benchmark simulator (ManiSkill/SAPIEN) + MME-VLA policy learning (openpi/JAX).

    modal run --detach modal_robomme.py::env_smoke          # simulator only, CPU render, one scripted rollout
    modal run --detach modal_robomme.py::policy_smoke       # download a released checkpoint, serve one request
Everything persistent lives on the volume `robomme` under /vol.
"""
import modal, os, subprocess, sys, shlex

app = modal.App("robomme")
vol = modal.Volume.from_name("robomme", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_HUB_ENABLE_HF_TRANSFER": "0"})

APT = ["build-essential", "ca-certificates", "curl", "ffmpeg", "git", "git-lfs", "libegl1", "libgl1", "libglib2.0-0",
       "libvulkan1", "vulkan-tools", "libxext6", "libxrender1", "libosmesa6", "tmux", "lsof", "mesa-vulkan-drivers", "clang", "cmake", "ninja-build", "pkg-config", "libx11-dev", "libxi-dev", "libxrandr-dev", "libxinerama-dev", "libxcursor-dev"]

image = (modal.Image.from_registry("nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04", add_python="3.11")
         .apt_install(*APT)
         .run_commands(
             "curl -LsSf https://astral.sh/uv/0.8.17/install.sh | sh && cp /root/.local/bin/uv /usr/local/bin/uv && cp /root/.local/bin/uvx /usr/local/bin/uvx",
             "git lfs install",
             # simulator
             "git clone --depth 1 https://github.com/RoboMME/robomme_benchmark /app/robomme_benchmark",
             "cd /app/robomme_benchmark && uv sync --frozen --no-dev --group server --python 3.11 && uv pip install -e .",
             # policy learning (openpi/JAX)
             "git clone --depth 1 https://github.com/RoboMME/robomme_policy_learning /app/robomme_policy_learning",
             "cd /app/robomme_policy_learning && GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11 && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .",
             "pip install huggingface_hub",
         )
         .add_local_dir("hmrobo", "/app/ours/hmrobo", copy=True)
         .env({"UV_LINK_MODE": "copy", "NVIDIA_DRIVER_CAPABILITIES": "compute,graphics,utility,video",
               "XDG_RUNTIME_DIR": "/tmp/runtime-root", "OPENPI_DATA_HOME": "/vol/openpi_data", "HF_HOME": "/vol/hf"}))


def _sh(cmd, cwd=None, env=None):
    print("+", cmd, flush=True)
    e = dict(os.environ); e.update(env or {})
    return subprocess.run(["bash", "-lc", cmd], cwd=cwd, env=e).returncode


@app.function(image=image, cpu=8, memory=16384, timeout=3600, volumes={"/vol": vol})
def env_smoke():
    os.makedirs("/tmp/runtime-root", exist_ok=True)
    env = {"SAPIEN_RENDER_DEVICE": "cpu", "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}
    rc = _sh("uv run python -c \"import mani_skill, sapien; print('maniskill', mani_skill.__version__, 'sapien', sapien.__version__)\"",
             cwd="/app/robomme_benchmark", env=env)
    _sh("ls scripts src; sed -n 1,80p scripts/run_example.py", cwd="/app/robomme_benchmark")
    rc2 = _sh("timeout 1500 uv run scripts/run_example.py 2>&1 | tail -40", cwd="/app/robomme_benchmark", env=env)
    _sh("ls -la runs 2>/dev/null | head; find . -name '*.mp4' -newer pyproject.toml | head", cwd="/app/robomme_benchmark")
    vol.commit()
    return rc, rc2


@app.function(image=image, gpu="A10G", cpu=8, memory=32768, timeout=3600, volumes={"/vol": vol})
def env_smoke_gpu():
    os.makedirs("/tmp/runtime-root", exist_ok=True)
    _sh("nvidia-smi | head -15; vulkaninfo --summary 2>&1 | head -30")
    rc = _sh("timeout 1500 uv run scripts/run_example.py 2>&1 | tail -40", cwd="/app/robomme_benchmark",
             env={"SAPIEN_RENDER_DEVICE": "cuda"})
    vol.commit()
    return rc


@app.function(image=image, gpu="A10G", cpu=8, memory=32768, timeout=3 * 3600, volumes={"/vol": vol})
def policy_smoke(variant: str = "perceptual-framesamp-modul", task: str = "VideoRepick", budget_s: int = 1500):
    """Download the released checkpoint + pi05 assets, serve the policy, run the official eval client on one task."""
    os.makedirs("/tmp/runtime-root", exist_ok=True)
    PL = "/app/robomme_policy_learning"; BM = "/app/robomme_benchmark"
    os.makedirs("/vol/openpi_data", exist_ok=True); os.makedirs(f"{PL}/runs/ckpts", exist_ok=True); os.makedirs(f"{PL}/runs/assets/mme_vla_suite", exist_ok=True)
    # 1. assets
    _sh(f"python -c \"from huggingface_hub import snapshot_download as s; s('Yinpei/mme_vla_suite', local_dir='/vol/ckpts/mme_vla_suite', allow_patterns=['{variant}/*','*.md']); s('Yinpei/pi05_vision_encoder', local_dir='/vol/openpi_data/pi05_vision_encoder')\"")
    _sh("du -sh /vol/ckpts/mme_vla_suite/* /vol/openpi_data/* 2>/dev/null; find /vol/ckpts/mme_vla_suite -maxdepth 3 | head -30")
    if not os.path.exists("/vol/openpi_data/openpi-assets/checkpoints/pi05_base"):
        _sh("uv run scripts/download_pi05_base.py 2>&1 | tail -5", cwd=PL)
    _sh("find /vol/openpi_data -maxdepth 3 | head -30; du -sh /vol/openpi_data/*")
    # 2. unzip checkpoint archives if needed
    _sh("ln -sfn /vol/ckpts/mme_vla_suite runs/ckpts/mme_vla_suite; uv run scripts/unzip_ckpt.py /vol/ckpts 2>&1 | tail -5; find /vol/ckpts/mme_vla_suite/ -maxdepth 3 | head -30", cwd=PL)
    _sh("cp -r assets/norm_stats.json runs/assets/mme_vla_suite/ 2>/dev/null; ls runs/assets/mme_vla_suite", cwd=PL)
    # 3. client deps in the simulator venv
    _sh("grep -vE '^(torch|torchvision|mani|sapien)' /app/robomme_policy_learning/examples/robomme/requirements.txt > /tmp/req.txt; cat /tmp/req.txt; uv pip install -r /tmp/req.txt 2>&1 | tail -2; uv pip install -e /app/robomme_policy_learning/packages/openpi-client 2>&1 | tail -1; uv pip install 'huggingface-hub>=0.34,<1.0' 2>&1 | tail -1; .venv/bin/python -c 'import transformers, google.genai; print(\"client deps ok\", transformers.__version__)'", cwd=BM)
    # 4. serve + eval
    ck = subprocess.run(["bash", "-lc", f"ls -d /vol/ckpts/mme_vla_suite/{variant}/*/ | grep -v cache | sort | tail -1"], cwd=PL, capture_output=True, text=True).stdout.strip().rstrip("/")
    _sh(f"find {ck} -maxdepth 2 | head -20")
    print("checkpoint dir:", ck, flush=True)
    srv = subprocess.Popen(["bash", "-lc", f"uv run scripts/serve_policy.py --seed=7 --port=8011 policy:checkpoint --policy.dir={ck} --policy.config=mme_vla_suite > /vol/serve.log 2>&1"],
                           cwd=PL, env=dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION="0.8", CUDA_VISIBLE_DEVICES="0"))
    import time; t0 = time.time()
    while time.time() - t0 < 900:
        time.sleep(15)
        if subprocess.run(["bash", "-lc", "lsof -i :8011 | grep -q LISTEN"]).returncode == 0:
            break
        if srv.poll() is not None:
            break
    _sh("tail -30 /vol/serve.log")
    rc = _sh(f"timeout {budget_s} .venv/bin/python /app/robomme_policy_learning/examples/robomme/eval.py --args.model_seed=7 --args.port=8011 --args.policy_name={variant} --args.model_ckpt_id=smoke --args.save_dir=/vol/eval_smoke --args.only_tasks {task} 2>&1 | tee /vol/eval_smoke.log | tail -60",
             cwd=BM, env={"SAPIEN_RENDER_DEVICE": "cuda", "PYTHONPATH": f"/app/robomme_policy_learning/examples/robomme:/app/robomme_policy_learning/src:{BM}/src"})
    _sh("find /vol/eval_smoke -maxdepth 4 | head; cat $(find /vol/eval_smoke -name progress.json | head -1) 2>/dev/null; tail -20 /vol/serve.log; tail -40 /vol/eval_smoke.log")
    srv.terminate()
    vol.commit()
    return rc


@app.function(image=image, cpu=8, memory=32768, timeout=3 * 3600, volumes={"/vol": vol})
def fetch_data(tasks: str = "VideoRepick,VideoUnmask,MoveCube", dest: str = "/vol/data"):
    """Download + decompress RoboMME h5 archives for the given tasks into dest (default /vol/data = the eval distractor
    pool; use another dir, e.g. /vol/data_extra, for router-training data that must not change that pool)."""
    vol.reload()
    os.makedirs(dest, exist_ok=True)
    pats = [f"record_dataset_{t}.h5.tar.xz" for t in tasks.split(",")] + ["tarxz_h5.py", "README.md"]
    _sh(f"python -c \"from huggingface_hub import snapshot_download as s; s('Yinpei/robomme_data_h5', repo_type='dataset', local_dir='{dest}', allow_patterns={pats!r})\"")
    _sh(f"ls -la {dest}")
    for t in tasks.split(","):
        if not os.path.exists(f"{dest}/record_dataset_{t}.h5"):
            _sh(f"cd {dest} && tar -xJf record_dataset_{t}.h5.tar.xz && rm -f record_dataset_{t}.h5.tar.xz && ls -la")
    if dest != "/vol/data":
        _sh(f"du -sh {dest}/*"); vol.commit(); return 0
    _sh("du -sh /vol/data/*; uv run python -c \"import h5py,sys; f=h5py.File('/vol/data/record_dataset_VideoRepick.h5'); eps=list(f.keys()); print(len(eps), eps[:3]); e=f[eps[0]]; print(list(e.keys())[:5], dict(e['setup'].attrs) if hasattr(e['setup'],'attrs') else ''); ts=[k for k in e.keys() if k.startswith('timestep')]; print('timesteps', len(ts)); t=e[ts[0]]; print({k: (t['obs'][k].shape, t['obs'][k].dtype) for k in t['obs']}); print({k: t['info'][k][()] for k in t['info']})\"", cwd="/app/robomme_benchmark")
    vol.commit()
    return 0


@app.function(image=image, cpu=8, memory=32768, timeout=3 * 3600, volumes={"/vol": vol})
def fetch_ckpt(variant: str = "perceptual-framesamp-expert"):
    """Download + unzip one released MME-VLA checkpoint into /vol/ckpts/mme_vla_suite/<variant>/ (no GPU)."""
    PL = "/app/robomme_policy_learning"
    _sh(f"python -c \"from huggingface_hub import snapshot_download as s; s('Yinpei/mme_vla_suite', local_dir='/vol/ckpts/mme_vla_suite', allow_patterns=['{variant}/*'])\"")
    _sh(f"ls -la /vol/ckpts/mme_vla_suite/{variant}; cat /vol/ckpts/mme_vla_suite/{variant}/history_config.txt")
    _sh(f"uv run scripts/unzip_ckpt.py /vol/ckpts/mme_vla_suite/{variant} 2>&1 | tail -3", cwd=PL)
    _sh(f"rm -f /vol/ckpts/mme_vla_suite/{variant}/*.zip; find /vol/ckpts/mme_vla_suite/{variant} -maxdepth 2 | head; du -sh /vol/ckpts/mme_vla_suite/{variant}")
    vol.commit()
    return _ckpt(variant)


def _ckpt(variant: str) -> str:
    if "/" in variant:   # explicit "<exp>/<step>" (fine-tuned checkpoints)
        return f"/vol/ckpts/mme_vla_suite/{variant}"
    return subprocess.run(["bash", "-lc", f"ls -d /vol/ckpts/mme_vla_suite/{variant}/*/ | grep -v cache | sort -V | tail -1"], capture_output=True, text=True).stdout.strip().rstrip("/")


def _kill_server():
    """Stop a policy server left over from a previous call in this (warm) container. `srv.terminate()` only kills the
    bash wrapper; the python child keeps /vol/serve.log open, and vol.reload() then refuses to run."""
    import time
    subprocess.run(["bash", "-lc", "pkill -f 'hmrobo/[s]erve_mm.py'"])
    for _ in range(30):
        if subprocess.run(["bash", "-lc", "pgrep -f 'hmrobo/[s]erve_mm.py' >/dev/null"]).returncode != 0:
            return
        time.sleep(2)
    subprocess.run(["bash", "-lc", "pkill -9 -f 'hmrobo/[s]erve_mm.py'"]); time.sleep(2)


def _serve(variant: str, PL: str, selector: str = "", router: str = "", env: str = ""):
    """Start the memory-managed policy server (hmrobo/serve_mm.py); returns the Popen once it listens (<= 15 min)."""
    import time
    ck = _ckpt(variant)
    os.makedirs(f"{PL}/runs/assets/mme_vla_suite", exist_ok=True)
    _sh("uv pip install --python .venv/bin/python transformers sentencepiece protobuf pillow 2>&1 | tail -1", cwd=PL)   # HF SigLIP for text_route
    srv = subprocess.Popen(["bash", "-lc", f".venv/bin/python /app/ours/hmrobo/serve_mm.py --dir {ck} --config mme_vla_suite --port 8011 --seed 7 --selector '{selector}' --router '{router}' > /tmp/serve.log 2>&1"],
                           cwd=PL, env=dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION="0.6", CUDA_VISIBLE_DEVICES="0", PYTHONPATH="/app/ours", **dict(kv.split("=", 1) for kv in env.split(",") if "=" in kv)))
    t0 = time.time()
    while time.time() - t0 < 900 and srv.poll() is None:
        if subprocess.run(["bash", "-lc", "lsof -i :8011 | grep -q LISTEN"]).returncode == 0:
            return srv
        time.sleep(10)
    _sh("tail -30 /tmp/serve.log")
    return srv


@app.function(image=image, gpu="A10G", cpu=8, memory=32768, timeout=8 * 3600 + 60, volumes={"/vol": vol})
def eval_cross(args: str = "--task VideoRepick --k 0 3 7 --systems uniform retrieval --episodes 10", variant: str = "perceptual-framesamp-modul", selector: str = "", router: str = "", env: str = ""):
    _kill_server()
    try:
        vol.reload()   # a warm container may hold a stale view of files committed by another function
    except Exception as e:   # the open-files check spans containers; a parallel cell's server used to hold /vol/serve.log
        print("vol.reload skipped:", str(e)[:200], flush=True)
    os.makedirs("/tmp/runtime-root", exist_ok=True)
    PL = "/app/robomme_policy_learning"; BM = "/app/robomme_benchmark"
    _sh("uv pip install -e /app/robomme_policy_learning/packages/openpi-client h5py 2>&1 | tail -1", cwd=BM)
    srv = _serve(variant, PL, selector, router, env)
    import time
    cmd = f".venv/bin/python /app/ours/hmrobo/eval_cross.py {args} --port 8011 2>&1 | tee -a /vol/eval_cross.log | grep -vE 'Warning|warn'"
    print("+", cmd, flush=True)
    proc = subprocess.Popen(["bash", "-lc", cmd], cwd=BM, env=dict(os.environ, SAPIEN_RENDER_DEVICE="cuda", PYTHONPATH=f"/app/ours:/app/robomme_policy_learning/examples/robomme:{BM}/src"))
    while proc.poll() is None:   # persist per-episode results every 5 min (the JSON is resumable)
        time.sleep(300)
        vol.commit()
    rc = proc.returncode
    _sh("tail -5 /tmp/serve.log")
    srv.terminate(); _kill_server()
    import re
    m = re.search(r"--task\s+(\S+)", args); tag = m.group(1) if m else "run"
    os.makedirs("/vol/serve_logs", exist_ok=True)
    _sh(f"cp /tmp/serve.log /vol/serve_logs/{tag}_{int(time.time())}.log; cp /tmp/serve.log /vol/serve.log")
    vol.commit()
    return rc


@app.function(image=image, gpu="A10G", cpu=8, memory=32768, timeout=6 * 3600, volumes={"/vol": vol})
def gen_same_scene(args: str = "--task MoveCube --split train --episodes 50 --m 4"):
    """Simulator-only: cache same-scene sessions + lesson videos for a split (hard-routing condition / router training)."""
    vol.reload()   # a warm container may hold a stale view of files committed by another function
    os.makedirs("/tmp/runtime-root", exist_ok=True)
    BM = "/app/robomme_benchmark"
    _sh("uv pip install -e /app/robomme_policy_learning/packages/openpi-client h5py 2>&1 | tail -1", cwd=BM)
    import time
    cmd = f"set -o pipefail; .venv/bin/python /app/ours/hmrobo/gen_same_scene.py {args} 2>&1 | tee -a /vol/gen_same_scene.log | grep -vE 'Warning|warn'; test ${{PIPESTATUS[0]}} -eq 0"
    proc = subprocess.Popen(["bash", "-lc", cmd], cwd=BM, env=dict(os.environ, SAPIEN_RENDER_DEVICE="cuda", PYTHONPATH=f"/app/ours:/app/robomme_policy_learning/examples/robomme:{BM}/src"))
    while proc.poll() is None:
        time.sleep(300); vol.commit()
    vol.commit()
    return proc.returncode


@app.function(image=image, gpu="A10G", cpu=8, memory=32768, timeout=3 * 3600, volumes={"/vol": vol})
def siglip_feats(args: str = "--tasks MoveCube RouteStick --split train"):
    """HF SigLIP embeddings of the same-scene data + distractor units (hmrobo/siglip_feats.py), for instruction-conditioned routing."""
    vol.reload()
    BM = "/app/robomme_benchmark"
    _sh("uv pip install transformers sentencepiece protobuf pillow 2>&1 | tail -1", cwd=BM)
    cmd = f"set -o pipefail; .venv/bin/python /app/ours/hmrobo/siglip_feats.py {args} 2>&1 | tee -a /vol/siglip_feats.log | grep -vE 'Warning|warn'; test ${{PIPESTATUS[0]}} -eq 0"
    rc = subprocess.call(["bash", "-lc", cmd], cwd=BM, env=dict(os.environ, PYTHONPATH="/app/ours"))
    vol.commit()
    return rc


@app.function(image=image, gpu="A10G", cpu=8, memory=32768, timeout=3 * 3600, volumes={"/vol": vol})
def vlm_verify(args: str = "--tasks MoveCube RouteStick"):
    """Zero-shot VLM verifier routing pilot on the cached same-scene data (hmrobo/vlm_verify.py)."""
    vol.reload()
    BM = "/app/robomme_benchmark"
    _sh("uv pip install transformers accelerate pillow 2>&1 | tail -1", cwd=BM)
    cmd = f"set -o pipefail; .venv/bin/python /app/ours/hmrobo/vlm_verify.py {args} 2>&1 | tee -a /vol/vlm_verify.log | grep -vE 'Warning|warn'; test ${{PIPESTATUS[0]}} -eq 0"
    rc = subprocess.call(["bash", "-lc", cmd], cwd=BM, env=dict(os.environ, PYTHONPATH="/app/ours"))
    vol.commit()
    return rc


@app.function(image=image, gpu="A10G", cpu=8, memory=65536, timeout=8 * 3600, volumes={"/vol": vol})
def train_mm(args: str = "--task VideoRepick --updates 100 --out /vol/mm_VideoRepick.pt", variant: str = "perceptual-framesamp-modul"):
    """In-process GRPO training (or --probe) of the memory manager against the frozen VLA."""
    vol.reload()   # a warm container may hold a stale view of files committed by another function
    PL = "/app/robomme_policy_learning"
    os.makedirs(f"{PL}/runs/assets/mme_vla_suite", exist_ok=True)
    _sh("uv pip install --python .venv/bin/python h5py 2>&1 | tail -1", cwd=PL)
    import time
    cmd = f".venv/bin/python /app/ours/hmrobo/train_mm.py --ckpt {_ckpt(variant)} {args} 2>&1 | tee -a /vol/train_mm_$(echo '{args}' | grep -o 'task [A-Za-z]*' | cut -d' ' -f2).log | grep -vE 'Warning|warn'"
    print("+", cmd, flush=True)
    proc = subprocess.Popen(["bash", "-lc", cmd], cwd=PL, env=dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION="0.75", CUDA_VISIBLE_DEVICES="0", PYTHONPATH="/app/ours"))
    while proc.poll() is None:   # persist checkpoints/logs every 5 min so a cancellation or preemption loses little
        time.sleep(300)
        vol.commit()
    vol.commit()
    return proc.returncode


@app.function(image=image, gpu="A10G", cpu=8, memory=32768, timeout=3 * 3600, volumes={"/vol": vol})
def train_router(args: str = "--task VideoPlaceButton --out /vol/router_VideoPlaceButton.pt", variant: str = "perceptual-framesamp-modul"):
    """Encode train-split features with the frozen VLA and train the session router (hmrobo/train_router.py)."""
    vol.reload()   # a warm container may hold a stale view of files committed by another function
    PL = "/app/robomme_policy_learning"
    os.makedirs(f"{PL}/runs/assets/mme_vla_suite", exist_ok=True)
    _sh("uv pip install --python .venv/bin/python h5py 2>&1 | tail -1", cwd=PL)
    import time
    cmd = f"set -o pipefail; .venv/bin/python /app/ours/hmrobo/train_router.py --ckpt {_ckpt(variant)} {args} 2>&1 | tee -a /vol/train_router.log | grep -vE 'Warning|warn'; test ${{PIPESTATUS[0]}} -eq 0"
    print("+", cmd, flush=True)
    proc = subprocess.Popen(["bash", "-lc", cmd], cwd=PL, env=dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION="0.6", CUDA_VISIBLE_DEVICES="0", PYTHONPATH="/app/ours", **dict(kv.split("=", 1) for kv in env.split(",") if "=" in kv)))
    while proc.poll() is None:
        time.sleep(120)
        vol.commit()
    vol.commit()
    return proc.returncode


@app.function(image=image, gpu="A10G", cpu=8, memory=32768, timeout=6 * 3600, volumes={"/vol": vol})
def gen_demos(args: str = "--task VideoRepick --split val --out /vol/demos/val"):
    """Held-out oracle demonstrations with the benchmark's motion planner (simulator venv)."""
    os.makedirs("/tmp/runtime-root", exist_ok=True)
    BM = "/app/robomme_benchmark"
    rc = 0
    for task in args.split("--task")[1].split()[0].split(",") if "--task" in args else ["VideoRepick"]:
        rest = " ".join(t for t in args.replace(f"--task {args.split('--task')[1].split()[0]}", "").split())
        rc |= _sh(f".venv/bin/python /app/ours/hmrobo/gen_demos.py --task {task} {rest} 2>&1 | tee -a /vol/gen_demos.log | grep -vE 'Warning|warn'",
                  cwd=BM, env={"SAPIEN_RENDER_DEVICE": "cuda", "PYTHONPATH": f"/app/ours:/app/robomme_policy_learning/examples/robomme:{BM}/src"})
        vol.commit()
    return rc


@app.function(image=image, cpu=1, memory=2048, timeout=6 * 3600, volumes={"/vol": vol})
def wait_then_eval(marker: str, args: str, selector: str = "", max_wait_s: int = 4 * 3600):
    """Unattended chaining: wait until the training log on the volume contains `marker` (the final
    checkpoint message), then run eval_cross with that selector. Runs on Modal, independent of any laptop."""
    import time
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        vol.reload()
        try:
            import glob
            if any(marker in open(f).read() for f in glob.glob("/vol/train_mm*.log")) and (not selector or os.path.exists(selector)):
                break
        except FileNotFoundError:
            pass
        time.sleep(120)
    else:
        print("timed out waiting for", marker); return 1
    print("marker found after", int(time.time() - t0), "s; launching eval", flush=True)
    return eval_cross.remote(args, selector=selector)


@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={"/vol": vol})
def probe_h5(task: str = "VideoRepick"):
    code = r'''
import h5py, numpy as np
f = h5py.File("/vol/data/record_dataset_%s.h5", "r"); ep = f["episode_0"]
def walk(g, pre="", depth=0):
    for k in g:
        o = g[k]
        if isinstance(o, h5py.Dataset): print(pre + k, o.shape, o.dtype)
        elif depth < 2: print(pre + k + "/"); walk(o, pre + "  ", depth + 1)
print("setup:"); walk(ep["setup"], "  ")
print("timestep_0:"); walk(ep["timestep_0"], "  ")
ts = sorted([k for k in ep if k.startswith("timestep")], key=lambda s: int(s.split("_")[1]))
demo = [bool(ep[t]["info"]["is_video_demo"][()]) for t in ts]
print("T", len(ts), "demo frames", sum(demo), "first exec idx", demo.index(False) if False in demo else None)
print("goal", ep["setup"]["task_goal"][()])
''' % task
    import sys
    sys.path.insert(0, "/app/ours")
    rc = _sh(f"uv run python -c '{code}'", cwd="/app/robomme_benchmark")
    rc2 = _sh("uv run python -c \"import sys; sys.path.insert(0,'/app/ours'); from hmrobo.buffer import *; ep=load_episode('/vol/data/record_dataset_%s.h5','episode_0','%s', stride=8, max_T=40); print('load ok', ep.front.shape, ep.joint_action.shape, ep.is_demo.sum(), ep.goal); u=distractor_unit('/vol/data/record_dataset_MoveCube.h5','episode_0','MoveCube'); print('unit', u['front'].shape, u['state'].shape)\"" % (task, task), cwd="/app/robomme_benchmark")
    return rc, rc2


@app.function(image=image, gpu="A10G", cpu=8, memory=65536, timeout=6 * 3600, volumes={"/vol": vol})
def build_ft_data(args: str = "--h5 /vol/data/record_dataset_MoveCube.h5 --out /vol/ftdata_smoke --max_episodes 2"):
    """Per-episode SigLIP features + per-step samples for fine-tuning (hmrobo/build_ft_data.py)."""
    try: vol.reload()
    except Exception as e: print("vol.reload skipped:", str(e)[:120], flush=True)
    PL = "/app/robomme_policy_learning"
    os.makedirs(f"{PL}/runs/assets/mme_vla_suite", exist_ok=True)
    _sh("uv pip install --python .venv/bin/python h5py ml_dtypes 2>&1 | tail -1", cwd=PL)
    import time
    tag = args.split("--out")[1].split()[0].strip("/").replace("/", "_") if "--out" in args else "ftdata"
    cmd = f"set -o pipefail; .venv/bin/python /app/ours/hmrobo/build_ft_data.py {args} 2>&1 | grep -vE 'Warning|warn' | tee /tmp/build.log; test ${{PIPESTATUS[0]}} -eq 0"
    print("+", cmd, flush=True)
    proc = subprocess.Popen(["bash", "-lc", cmd], cwd=PL, env=dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION="0.6", CUDA_VISIBLE_DEVICES="0", PYTHONPATH="/app/ours"))
    while proc.poll() is None:
        time.sleep(300); vol.commit()
    _sh(f"mkdir -p /vol/ft_logs && cp /tmp/build.log /vol/ft_logs/build_{tag}.log"); vol.commit()
    return proc.returncode


@app.function(image=image, cpu=32, memory=65536, timeout=3 * 3600, volumes={"/vol": vol}, ephemeral_disk=524288)
def pack_ftdata(data: str = "/vol/ftdata", shards: int = 32):
    """Pack <data>/data (100k+ small pkl files) into <data>/tars/data_NN.tar so training containers can fetch it as a few
    large files; per-file reads from the volume run at ~5 files/s, which stalled the first pilot launch for hours."""
    try: vol.reload()
    except Exception as e: print("vol.reload skipped:", str(e)[:120], flush=True)
    import time
    t0 = time.time()
    names = sorted(os.listdir(f"{data}/data")); print("files", len(names), flush=True)
    os.makedirs("/root/lists", exist_ok=True); os.makedirs(f"{data}/tars", exist_ok=True)
    for i in range(shards):
        open(f"/root/lists/{i:02d}", "w").write("\n".join(names[i::shards]) + "\n")
    rc = _sh(f"ls /root/lists | xargs -P {shards} -I{{}} sh -c 'tar -cf /root/data_{{}}.tar -C {data}/data -T /root/lists/{{}} && echo done {{}}'")
    print(f"tarred in {time.time()-t0:.0f}s rc={rc}", flush=True)
    rc2 = _sh(f"cp /root/data_*.tar {data}/tars/ && ls -la {data}/tars | tail -3 && du -sh {data}/tars")
    vol.commit(); print(f"total {time.time()-t0:.0f}s", flush=True)
    return rc or rc2


@app.function(image=image, gpu="H100", cpu=16, memory=131072, timeout=12 * 3600, volumes={"/vol": vol}, ephemeral_disk=524288)
def train_ft(exp: str, mode: str = "interf", args: str = "--steps 2000 --save 500 --bs 32 --lr 1e-5", data: str = "/vol/ftdata",
             init: str = "/vol/ckpts/mme_vla_suite/perceptual-framesamp-modul/79999/params", env: str = ""):
    """Fine-tune a released checkpoint on local copies of the data; sync checkpoints to /vol/ckpts/mme_vla_suite/<exp>/<step>
    and the log to /vol/ft_logs/<exp>.log every minute (hmrobo/train_ft.py). env: extra "K=V,K=V" (FT_KS, FT_P_K0, ...)."""
    try: vol.reload()
    except Exception as e: print("vol.reload skipped:", str(e)[:120], flush=True)
    import time, shutil, glob
    PL = "/app/robomme_policy_learning"
    _sh("uv pip install --python .venv/bin/python h5py ml_dtypes 2>&1 | tail -1", cwd=PL)
    # norm stats of the released checkpoint -> training assets (asset_id "robomme")
    rel = os.path.dirname(init.rstrip("/"))
    _sh(f"mkdir -p {PL}/runs/assets/mme_vla_suite/robomme && cp {rel}/assets/robomme/norm_stats.json {PL}/runs/assets/mme_vla_suite/robomme/")
    # local copy of the data (the volume is slow for many small reads)
    t0 = time.time(); _sh(f"mkdir -p /root/ftdata/data && cp -r {data}/meta {data}/features /root/ftdata/")
    if os.path.isdir(f"{data}/tars"):   # packed by pack_ftdata: a few large files, extracted in parallel
        rc = _sh(f"ls {data}/tars/*.tar | xargs -P 8 -I{{}} tar -xf {{}} -C /root/ftdata/data")
    else:
        rc = _sh(f"cp -r {data}/data /root/ftdata/")
    _sh("ls /root/ftdata/data | wc -l; du -sh /root/ftdata/*"); print(f"data copied in {time.time()-t0:.0f}s rc={rc}", flush=True)
    if rc: return rc
    dst = f"/vol/ckpts/mme_vla_suite/{exp}"; os.makedirs(dst, exist_ok=True); os.makedirs("/vol/ft_logs", exist_ok=True)
    cmd = f"set -o pipefail; .venv/bin/python /app/ours/hmrobo/train_ft.py --exp {exp} --init {init} --data /root/ftdata --ckpt_base /root/ckpts {args} 2>&1 | grep -vE 'Warning|warn|UserWarning' | tee /tmp/train.log; test ${{PIPESTATUS[0]}} -eq 0"
    print("+", cmd, flush=True)
    e = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION="0.92", CUDA_VISIBLE_DEVICES="0", PYTHONPATH="/app/ours", WANDB_MODE="disabled", FT_MODE=mode,
             PYTHONUNBUFFERED="1",   # train.py print()s the "Step N: loss=" lines; piped stdout is block-buffered otherwise (no loss lines until ~step 1400)
             **dict(kv.split("=", 1) for kv in env.split(",") if "=" in kv))
    proc = subprocess.Popen(["bash", "-lc", cmd], cwd=PL, env=e)
    synced = set()
    def sync():
        src = f"/root/ckpts/mme_vla_suite/{exp}"
        if os.path.exists(f"{src}/history_config.txt"):
            shutil.copy(f"{src}/history_config.txt", f"{dst}/history_config.txt")
        for d in sorted(glob.glob(f"{src}/[0-9]*")):
            step = os.path.basename(d)
            if step.isdigit() and step not in synced and os.path.exists(f"{d}/params") and not glob.glob(f"{src}/*tmp*"):
                _sh(f"cp -r {d} {dst}/{step}.tmp && mv {dst}/{step}.tmp {dst}/{step}"); synced.add(step); print("synced ckpt", step, flush=True)
        shutil.copy("/tmp/train.log", f"/vol/ft_logs/{exp}.log"); vol.commit()
    while proc.poll() is None:
        time.sleep(60)
        try: sync()
        except Exception as ex: print("sync error:", ex, flush=True)
    time.sleep(5); sync()
    print("train_ft exit", proc.returncode, flush=True)
    return proc.returncode


@app.local_entrypoint()
def main(which: str = "env", selector: str = ""):
    if which == "env":
        print(env_smoke.remote())
    elif which == "env_gpu":
        print(env_smoke_gpu.remote())
    elif which == "data":
        print(fetch_data.remote())
    elif which.startswith("cross:"):
        print(eval_cross.remote(which.split(":", 1)[1], selector=selector))
    elif which.startswith("gen_ss:"):
        print(gen_same_scene.remote(which.split(":", 1)[1]))
    elif which.startswith("ckpt:"):
        print(fetch_ckpt.remote(which.split(":", 1)[1]))
    elif which == "probe":
        print(probe_h5.remote())
    elif which == "pack":
        print(pack_ftdata.remote())
    elif which.startswith("train:"):
        print(train_mm.remote(which.split(":", 1)[1]))
    elif which.startswith("router:"):
        print(train_router.remote(which.split(":", 1)[1]))
    elif which.startswith("gen:"):
        print(gen_demos.remote(which.split(":", 1)[1]))
    elif which.startswith("ftdata:"):
        print(build_ft_data.remote(which.split(":", 1)[1]))
    elif which.startswith("chain:"):   # chain:<marker>|<eval args>
        marker, eargs = which.split(":", 1)[1].split("|", 1)
        print(wait_then_eval.remote(marker, eargs, selector))
    else:
        print(policy_smoke.remote())
