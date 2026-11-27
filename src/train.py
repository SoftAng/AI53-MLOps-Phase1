from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.pytorch
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
from torch import nn
import yaml

from src.data import CLASSES, IMAGENET_MEAN, IMAGENET_STD, create_dataloaders
from src.model import BeanCNN
from src.provenance import write_run_manifest
from src.reproducibility import set_reproducibility


def load_params() -> dict:
    with Path("params.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def save_training_curves(history: list[dict], out_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train_loss")
    axes.plot(epochs, [row["val_loss"] for row in history], marker="s", label="val_loss")
    axes.set_title("Loss")
    axes.set_xlabel("Epoch")
    axes.grid(True, alpha=0.3)
    axes.legend()

    axes.plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="val_accuracy")
    axes.plot(epochs, [row["val_macro_f1"] for row in history], marker="s", label="val_macro_f1")
    axes.set_title("Validation Metrics")
    axes.set_xlabel("Epoch")
    axes.set_ylim(0, 1.05)
    axes.grid(True, alpha=0.3)
    axes.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_epoch(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    avg_loss = total_loss / max(total_examples, 1)
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_targets)))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return avg_loss, accuracy, macro_f1


def main() -> None:
    params = load_params()
    train_cfg = params["train"]
    aug_cfg = params["augmentation"]
    data_cfg = params["data"]
    tracking_cfg = params["tracking"]

    seed = int(train_cfg["random_state"])
    image_size = int(data_cfg["image_size"])
    batch_size = int(train_cfg["batch_size"])
    epochs = int(train_cfg["epochs"])
    lr = float(train_cfg["learning_rate"])
    dropout = float(train_cfg["dropout"])

    set_reproducibility(seed)
    device = resolve_device(params["compute"]["device"])

    mlflow.set_tracking_uri(tracking_cfg["tracking_uri"])
    mlflow.set_experiment(tracking_cfg["experiment_name"])

    train_loader, val_loader, _ = create_dataloaders(
        manifests_dir=Path("data/processed/manifests"),
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        aug_cfg=aug_cfg,
    )

    model = BeanCNN(num_classes=len(CLASSES), dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -1.0
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = []

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    run_name = f"cnn_aug-{bool(aug_cfg['enabled'])}_dropout-{dropout}_lr-{lr}"

    with mlflow.start_run(run_name=run_name) as run:
        manifest_path = Path("reports/run_manifest.json")
        manifest = write_run_manifest(
            path=manifest_path,
            params=params,
            run_id=run.info.run_id,
            device=str(device),
            manifests_dir=Path("data/processed/manifests"),
        )
        data_identity = manifest["data"]["training_data_identity"]

        mlflow.log_params({
            "image_size": image_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "dropout": dropout,
            "random_state": seed,
            "augmentation_enabled": bool(aug_cfg["enabled"]),
            "horizontal_flip": bool(aug_cfg["horizontal_flip"]),
            "rotation_degrees": aug_cfg["rotation_degrees"],
            "device": str(device),
            "training_data_identity": data_identity,
        })

        train_csv_path = Path("data/processed/manifests/train.csv")
        if train_csv_path.exists():
            train_df = pd.read_csv(train_csv_path)
            train_dataset = mlflow.data.from_pandas(
                train_df,
                source=str(train_csv_path),
                name="bean-training",
                targets="label",
            )
            mlflow.log_input(train_dataset, context="training")

        start_time = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0

            for images, targets in train_loader:
                images = images.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                bs = targets.size(0)
                running_loss += loss.item() * bs
                seen += bs

            train_loss = running_loss / max(seen, 1)
            val_loss, val_acc, val_macro_f1 = evaluate_epoch(model, val_loader, criterion, device)

            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_acc, 6),
                "val_macro_f1": round(val_macro_f1, 6),
            }
            history.append(row)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_macro_f1,
                },
                step=epoch,
            )

            if val_macro_f1 >= best_val_f1:
                best_val_f1 = val_macro_f1
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

            print(
                f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_macro_f1={val_macro_f1:.4f}"
            )

        training_seconds = round(time.perf_counter() - start_time, 3)
        mlflow.log_metric("training_seconds", training_seconds)

        model.load_state_dict(best_state)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": list(CLASSES),
                "image_size": image_size,
                "dropout": dropout,
            },
            "models/model.pt",
        )

        best_row = history[best_epoch - 1]
        summary = {
            "mlflow_run_id": run.info.run_id,
            "best_epoch": best_epoch,
            "best_val_loss": best_row["val_loss"],
            "best_val_accuracy": best_row["val_accuracy"],
            "best_val_macro_f1": best_row["val_macro_f1"],
            "final_train_loss": history[-1]["train_loss"],
            "training_seconds": training_seconds,
            "history": history,
        }
        summary_path = Path("reports/train_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        curves_path = Path("reports/training_curves.png")
        save_training_curves(history, curves_path)

        mlflow.log_artifact(str(curves_path))
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(manifest_path))

        cpu_model = model.to("cpu")
        cpu_model.eval()
        example_tensor = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        normalized_example = (example_tensor - mean) / std
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
                "class_names": list(CLASSES),
                "image_size": image_size,
            },
        )


if __name__ == "__main__":
    main()
