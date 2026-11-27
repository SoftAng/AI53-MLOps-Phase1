from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.pytorch
import pandas as pd
import torch
import yaml
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import Optimizer

from src.data import IMAGENET_MEAN, IMAGENET_STD, build_training_loaders
from src.model import BeanCNN
from src.provenance import write_run_manifest
from src.reproducibility import set_reproducibility


def resolve_device(requested: str) -> torch.device:
    """Resolve the execution device requested in params.yaml."""
    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("compute.device=cuda but CUDA is not available")
        return torch.device("cuda")
    if requested != "auto":
        raise ValueError("compute.device must be auto, cpu or cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: Optimizer | None,
    device: torch.device,
    training: bool,
) -> dict[str, float]:
    """Run one training or validation epoch."""
    if training and optimizer is None:
        raise ValueError("optimizer must be provided when training=True")

    model.train(training)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            y_true.extend(targets.detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def save_training_curves(history: dict[str, list[float]], output_path: Path) -> None:
    """Save loss and validation metric curves for MLflow and DVC."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="train_loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="val_loss")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_accuracy"], marker="o", label="val_accuracy")
    axes[1].plot(epochs, history["val_macro_f1"], marker="o", label="val_macro_f1")
    axes[1].set_title("Validation Performance")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))

    train_manifest_path = Path("data/processed/manifests/train.csv")
    validation_manifest_path = Path("data/processed/manifests/validation.csv")

    missing_manifests = [
        str(path)
        for path in (train_manifest_path, validation_manifest_path)
        if not path.exists()
    ]
    if missing_manifests:
        raise FileNotFoundError(
            "Required prepared manifest(s) are missing: "
            + ", ".join(missing_manifests)
            + ". Run PREPARE first."
        )

    seed = int(params["train"]["random_state"])
    set_reproducibility(seed)

    device = resolve_device(str(params["compute"]["device"]))
    train_loader, val_loader, class_names = build_training_loaders(
        Path("data/processed"), params
    )

    model = BeanCNN(
        num_classes=len(class_names),
        dropout=float(params["train"]["dropout"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(params["train"]["learning_rate"])
    )
    epochs = int(params["train"]["epochs"])
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    tracking = params["tracking"]
    mlflow.set_tracking_uri(tracking["uri"])
    mlflow.set_experiment(tracking["experiment"])

    run_name = (
        f"cnn_aug-{params['augmentation']['enabled']}"
        f"_dropout-{params['train']['dropout']}"
        f"_lr-{params['train']['learning_rate']}"
    )

    with mlflow.start_run(run_name=run_name) as run:
        # TODO 1 - CAPTURE PROVENANCE FIRST
        run_manifest_path = Path("reports/run_manifest.json")
        run_manifest = write_run_manifest(
            path=run_manifest_path,
            params=params,
            run_id=run.info.run_id,
            device=str(device),
        )
        training_identity = run_manifest["data"]["training_data_identity"]

        # TODO 2 - LOG RUN PARAMETERS
        mlflow.log_params(
            {
                "image_size": int(params["data"]["image_size"]),
                "batch_size": int(params["train"]["batch_size"]),
                "epochs": epochs,
                "learning_rate": float(params["train"]["learning_rate"]),
                "dropout": float(params["train"]["dropout"]),
                "random_state": seed,
                "augmentation_enabled": bool(params["augmentation"]["enabled"]),
                "horizontal_flip": float(params["augmentation"]["horizontal_flip"]),
                "rotation_degrees": int(params["augmentation"]["rotation_degrees"]),
                "device": str(device),
                "training_data_identity": training_identity,
            }
        )

        # TODO 3 - LOG TRAINING DATASET INPUT
        train_manifest_df = pd.read_csv(train_manifest_path)
        if "label" not in train_manifest_df.columns:
            raise ValueError("Training manifest must contain a 'label' column")

        training_dataset = mlflow.data.from_pandas(
            train_manifest_df,
            source=str(train_manifest_path),
            name="bean-training",
            targets="label",
        )
        mlflow.log_input(training_dataset, context="training")

        start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, criterion, optimizer, device, True
            )
            val_metrics = run_epoch(
                model, val_loader, criterion, None, device, False
            )

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])

            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )

            # TODO 4 - Log metrics per epoch
            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                },
                step=epoch,
            )

        training_seconds = time.perf_counter() - start
        # TODO 5 - Log training_seconds
        mlflow.log_metric("training_seconds", float(training_seconds))

        models_dir = Path("models")
        reports_dir = Path("reports")
        models_dir.mkdir(exist_ok=True)
        reports_dir.mkdir(exist_ok=True)

        image_size = int(params["data"]["image_size"])
        model_path = models_dir / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": list(class_names),
                "image_size": image_size,
                "dropout": float(params["train"]["dropout"]),
            },
            model_path,
        )

        training_curves_path = reports_dir / "training_curves.png"
        save_training_curves(history, training_curves_path)

        summary = {
            "mlflow_run_id": run.info.run_id,
            "run_name": run_name,
            "class_names": list(class_names),
            "epochs": epochs,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "final_val_accuracy": history["val_accuracy"][-1],
            "final_val_macro_f1": history["val_macro_f1"][-1],
            "best_val_accuracy": max(history["val_accuracy"]),
            "best_val_macro_f1": max(history["val_macro_f1"]),
            "training_seconds": round(training_seconds, 3),
            "training_data_identity": training_identity,
        }
        summary_path = reports_dir / "train_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

               # TODO 6 - LOG ARTIFACTS TO MLFLOW
        mlflow.log_artifact(str(training_curves_path))
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(run_manifest_path))

        # TODO 7 - LOG PYTORCH MODEL WITH SIGNATURE
        cpu_model = model.to("cpu")
        cpu_model.eval()
        raw_example = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        normalized_example = (raw_example - mean) / std
        input_example = normalized_example.numpy()

        with torch.no_grad():
            output_example = cpu_model(normalized_example).numpy()

        signature = infer_signature(input_example, output_example)

        mlflow.pytorch.log_model(
            pytorch_model=cpu_model,
            name="bean-cnn",
            signature=signature,
            input_example=input_example,
            code_paths=["src/model.py"],
            metadata={
                "class_names": list(class_names),
                "image_size": image_size,
            },
        )

        print(f"Saved MLflow Run: {run.info.run_id}")


if __name__ == "__main__":
    main()
