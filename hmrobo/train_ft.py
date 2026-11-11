"""Fine-tune a released MME-VLA checkpoint with the InterferenceDataset (policy-learning venv, one GPU).

    FT_MODE=interf .venv/bin/python /app/ours/hmrobo/train_ft.py --exp ft-interf-v1 --init /vol/ckpts/mme_vla_suite/perceptual-framesamp-modul/79999/params \
        --data /root/ftdata --steps 2000 --save 500 --bs 32 --lr 1e-5 --ckpt_base /root/ckpts

Builds the `mme_vla_suite` TrainConfig in Python (no tyro overrides of nested fields), swaps the dataset class, runs
scripts/train.py:main once. Saved params are the EMA params; checkpoints land in <ckpt_base>/mme_vla_suite/<exp>/<step>.
"""
from __future__ import annotations
import argparse, dataclasses, os, sys
sys.path.insert(0, "/app/ours"); sys.path.insert(0, "/app/robomme_policy_learning/scripts")
import mme_vla_suite.training.dataloader as _dl  # noqa: E402
from hmrobo.ft_dataset import InterferenceDataset  # noqa: E402
_dl.RoboMMEDataset = InterferenceDataset
from mme_vla_suite.training import config as _config  # noqa: E402
import openpi.training.weight_loaders as _wl  # noqa: E402
import openpi.training.optimizer as _opt  # noqa: E402
import train as _train  # noqa: E402  (scripts/train.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True); ap.add_argument("--init", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--save", type=int, default=500)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--ema", type=float, default=0.99); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_base", default="/root/ckpts"); ap.add_argument("--assets_base", default="runs/assets")
    ap.add_argument("--history", default="perceptual-framesamp-modul.yaml"); ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--resume_step", type=int, default=None)
    a = ap.parse_args()
    # positional table check: numpy copy vs the suite's JAX PosEmb3D (the training positions must match evaluation)
    import numpy as np, jax.numpy as jnp
    from mme_vla_suite.shared.posemb_3d import PosEmb3D
    from hmrobo.ft_dataset import pos_table
    ref = np.asarray(PosEmb3D(768)(jnp.arange(300), 4)); ours = pos_table(300)
    err = float(np.abs(ref - ours).max()); print(f"pos_table max abs err vs PosEmb3D: {err:.2e}", flush=True); assert err < 1e-3, err
    cfg = _config.get_config("mme_vla_suite")
    cfg = dataclasses.replace(
        cfg, exp_name=a.exp, batch_size=a.bs, num_workers=a.workers, fsdp_devices=1, seed=a.seed,
        dataset_path=a.data, checkpoint_base_dir=a.ckpt_base, assets_base_dir=a.assets_base,
        model=dataclasses.replace(cfg.model, use_history=True, history_config=a.history),
        weight_loader=_wl.CheckpointWeightLoader(a.init),
        lr_schedule=_opt.CosineDecaySchedule(warmup_steps=a.warmup, peak_lr=a.lr, decay_steps=a.steps, decay_lr=a.lr / 10),
        num_train_steps=a.steps, save_interval=a.save, keep_period=a.save, log_interval=a.log_interval,
        ema_decay=a.ema, wandb_enabled=False, overwrite=not a.resume, resume=a.resume, resum_ckpt_id=a.resume_step)
    print("TRAIN", a.exp, "mode", os.environ.get("FT_MODE"), "steps", a.steps, "bs", a.bs, "lr", a.lr, flush=True)
    _train.main(cfg)


if __name__ == "__main__":
    main()
