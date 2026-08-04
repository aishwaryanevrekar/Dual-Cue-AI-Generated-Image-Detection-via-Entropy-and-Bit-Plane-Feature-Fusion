#!/usr/bin/env python3
"""
Model Training & Evaluation Script: LOTA Steganalysis AI-Generated Image Classifier
Trains LOTAClassifier exclusively using least-significant bit (LSB) noise maps and 4-directional MGPS Top-K quadrant patches.
Compatible with dataset10000 or custom dataset directories.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure root directory is on python path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.data.dataset import SharedImageDataset
from src.data.dataloader import create_dataloader
from src.models.classifier import LOTAClassifier
from src.utils.logger import get_logger

logger = get_logger("lota_train")


def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute binary classification accuracy, precision, recall, F1, and ROC-AUC.
    """
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    try:
        auc = float(roc_auc_score(y_true, y_pred_prob))
    except Exception:
        auc = 0.5

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
    }


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Execute single training epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        images, labels = batch[0].to(device), batch[1].to(device, dtype=torch.float32)

        optimizer.zero_grad()
        logits = model(images).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = labels.cpu().numpy()

        all_probs.extend(probs)
        all_targets.extend(targets)

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_targets), np.array(all_probs))
    metrics["loss"] = epoch_loss
    metrics["time_sec"] = time.time() - start_time
    return epoch_loss, metrics


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate model on validation or test split.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    start_time = time.time()
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device, dtype=torch.float32)

            logits = model(images).squeeze(-1)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = labels.cpu().numpy()

            all_probs.extend(probs)
            all_targets.extend(targets)

    eval_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_targets), np.array(all_probs))
    metrics["loss"] = eval_loss
    metrics["time_sec"] = time.time() - start_time
    return eval_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train LOTA Steganalysis AI Image Detection Model")
    parser.add_argument("--data_dir", type=str, default="dataset10000", help="Path to benchmark dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--output_dir", type=str, default="outputs/train", help="Directory to save model & logs")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")

    args = parser.parse_args()

    # Determine execution device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using execution device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Datasets
    logger.info(f"Loading dataset splits from {args.data_dir}...")
    train_dataset = SharedImageDataset(root_dir=args.data_dir, split="train")
    val_dataset = SharedImageDataset(root_dir=args.data_dir, split="validation")
    test_dataset = SharedImageDataset(root_dir=args.data_dir, split="test")

    logger.info(f"Dataset Loaded | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # 2. Create DataLoaders
    train_loader = create_dataloader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, balanced_sampling=True
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )
    test_loader = create_dataloader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    # 3. Instantiate LOTA Classifier Model
    model = LOTAClassifier(k_patches=4, patch_size=32, grid_size=8).to(device)
    logger.info(f"Initialized LOTAClassifier Model ({sum(p.numel() for p in model.parameters())} parameters)")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auc = 0.0
    history = {"train": [], "val": [], "test": None}

    # 4. Training Loop
    logger.info(f"Starting LOTA Model Training for {args.epochs} Epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        logger.info(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_metrics['accuracy']:.4f} AUC: {train_metrics['roc_auc']:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']:.4f} AUC: {val_metrics['roc_auc']:.4f}"
        )

        # Save Best Model Checkpoint based on Validation ROC-AUC
        if val_metrics["roc_auc"] >= best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            checkpoint_path = output_dir / "best_lota_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_val_auc,
                "val_metrics": val_metrics,
            }, checkpoint_path)
            logger.info(f"--> Saved best model checkpoint to {checkpoint_path} (Val AUC: {best_val_auc:.4f})")

    # 5. Final Evaluation on Test Set
    logger.info("Evaluating Best LOTA Model on Test Split...")
    best_checkpoint = torch.load(output_dir / "best_lota_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    history["test"] = test_metrics

    logger.info("================================================================================")
    logger.info("TEST EVALUATION RESULTS:")
    logger.info(f"Test Loss     : {test_metrics['loss']:.4f}")
    logger.info(f"Test Accuracy : {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall   : {test_metrics['recall']:.4f}")
    logger.info(f"Test F1-Score : {test_metrics['f1_score']:.4f}")
    logger.info(f"Test ROC-AUC  : {test_metrics['roc_auc']:.4f}")
    logger.info("================================================================================")

    # Save History JSON Report
    history_file = output_dir / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    logger.info(f"Saved full training history report to {history_file}")


if __name__ == "__main__":
    main()
