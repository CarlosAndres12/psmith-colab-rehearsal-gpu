"""GPU rehearsal unit: one CUDA matmul, seconds. Needs torch (preinstalled on Colab GPU runtimes)."""
from __future__ import annotations

import json
import platform
from pathlib import Path


def _cuda_probe() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this runtime")
    x = torch.randn(2048, 2048, device="cuda")
    y = (x @ x).sum().item()
    return {
        "cuda": True,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "matmul_ok": y == y,
    }


def main(**kwargs) -> None:
    probe = _cuda_probe()
    Path("run-proof.txt").write_text("ran on " + probe["gpu"] + "\n", encoding="utf-8")
    print("run: ok")


def smoke(**kwargs) -> None:
    evidence = {"backend": "colab", "python": platform.python_version(), **_cuda_probe()}
    Path("shard.json").write_text(json.dumps({"evidence": evidence}), encoding="utf-8")
    print("smoke: ok")
