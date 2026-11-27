from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import torch


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def training_data_identity(manifests_dir: Path = Path("data/processed/manifests")) -> str:
    h = hashlib.sha256()
    for name in ["train.csv", "validation.csv"]:
        p = manifests_dir / name
        if p.exists():
            h.update(name.encode("utf-8"))
            h.update(sha256_file(p).encode("utf-8"))
    return h.hexdigest()[:16]


def _git_state() -> dict[str, object]:
    commit = "unknown"
    dirty = False
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = res.stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"

    try:
        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = bool(status_res.stdout.strip())
    except Exception:
        dirty = False

    return {"commit": commit, "dirty": dirty}


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def build_run_manifest(params: dict, device: str, mlflow_run_id: str) -> dict:
    """Build experiment provenance metadata."""
    data_cfg = params.get("data", {})
    train_cfg = params.get("train", {})
    aug_cfg = params.get("augmentation", {})
    cuda_ver = torch.version.cuda if torch.version.cuda is not None else "none"

    return {
        "git": _git_state(),
        "data": {
            "training_data_identity": training_data_identity(),
        },
        "execution": {
            "mlflow_run_id": mlflow_run_id,
            "device": str(device),
            "random_state": train_cfg.get("random_state"),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": _pkg_version("torch"),
            "torchvision": _pkg_version("torchvision"),
            "dvc": _pkg_version("dvc"),
            "mlflow": _pkg_version("mlflow"),
            "cuda_runtime": str(cuda_ver),
        },
        "configuration": {
            "image_size": data_cfg.get("image_size"),
            "batch_size": train_cfg.get("batch_size"),
            "epochs": train_cfg.get("epochs"),
            "learning_rate": train_cfg.get("learning_rate"),
            "dropout": train_cfg.get("dropout"),
            "augmentation_enabled": aug_cfg.get("enabled"),
            "horizontal_flip": aug_cfg.get("horizontal_flip"),
            "rotation_degrees": aug_cfg.get("rotation_degrees"),
        },
    }


def write_run_manifest(
    path: Path,
    params: dict,
    run_id: str,
    device: str,
    manifests_dir: Path = Path("data/processed/manifests"),
) -> dict:
    """Create and write reports/run_manifest.json before training writes outputs."""
    manifest = build_run_manifest(params=params, device=device, mlflow_run_id=run_id)
    manifest["data"]["training_data_identity"] = training_data_identity(manifests_dir)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
