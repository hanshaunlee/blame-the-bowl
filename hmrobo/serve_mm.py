"""Serve the released MME-VLA checkpoint with the memory manager patched in (policy-learning venv).

    .venv/bin/python /app/ours/hmrobo/serve_mm.py --dir <ckpt> --port 8011 --seed 7 [--selector /vol/sel.pt]
"""
import dataclasses, logging, os, sys
from pathlib import Path
import tyro
sys.path.insert(0, "/app/ours")
from mme_vla_suite.policies import policy_config as _policy_config  # noqa: E402
from mme_vla_suite.serving import websocket_policy_server  # noqa: E402
from mme_vla_suite.training import config as _config  # noqa: E402
from hmrobo.mm_policy import MemoryManager  # noqa: E402


@dataclasses.dataclass
class Args:
    dir: str
    config: str = "mme_vla_suite"
    port: int = 8011
    seed: int = 7
    selector: str = ""
    router: str = ""


def main(a: Args):
    pol = _policy_config.create_trained_policy(_config.get_config(a.config), Path(a.dir), seed=a.seed)
    MemoryManager(pol, a.selector or None, a.router or None)
    print("history config:", pol.config, flush=True)
    websocket_policy_server.WebsocketPolicyServer(policy=pol, host="0.0.0.0", port=a.port, metadata=pol.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
