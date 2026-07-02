"""
train.py
Training loop for the Real-Time Voice Deepfake Detection System.

Checkpoints the model whenever validation EER improves (not just val loss --
EER is the metric that actually matters for this task and is what we report
in the final results). Uses early stopping with a configurable patience.

Usage (staged experiments -- see config.py / dataset.py for the reasoning):

    # Experiment 1: pipeline sanity check (1,000 train clips)
    python src/train.py \
        --train_protocol data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
        --train_audio_dir data/raw/ASVspoof2019_LA_train/flac \
        --dev_protocol data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
        --dev_audio_dir data/raw/ASVspoof2019_LA_dev/flac \
        --train_max_bonafide 500 --train_max_spoof 500 \
        --dev_max_bonafide 150 --dev_max_spoof 150 \
        --epochs 5 --checkpoint_path checkpoints/exp1_best.pt

    # Experiment 2: 4,000 train clips, 10 epochs
    ... --train_max_bonafide 2000 --train_max_spoof 2000 --epochs 10 ...

    # Experiment 3: 6,000 train clips, early stopping decides epoch count
    ... --train_max_bonafide 3000 --train_max_spoof 3000 --epochs 30 --patience 3 ...
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import build_file_list, cache_features, SpoofDataset
from model import SpoofCNN, count_parameters
from metrics import full_report


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch_train(model, loader, optimizer, loss_fn, device, max_grad_norm=1.0):
    """
    max_grad_norm: gradient clipping threshold. Without this, occasional large
    gradients (common on small/noisy batches) can push weights into a bad
    region and cause the loss spikes seen in early experiments (e.g. val_loss
    jumping from 0.05 to 9.96 between epochs). Clipping caps the gradient's
    L2 norm before the optimizer step, which smooths training without
    changing what the model is capable of learning.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)  # (batch,1) to match logits shape

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def run_epoch_eval(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)

            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits)
            all_scores.extend(probs.cpu().numpy().flatten().tolist())
            all_labels.extend(y.cpu().numpy().flatten().tolist())

    avg_loss = total_loss / max(n_batches, 1)
    report = full_report(all_labels, all_scores)
    return avg_loss, report


def train(args):
    device = get_device()
    print(f"Using device: {device}")

    # --- Build file lists (staged subsets) ---
    print("\n=== Building train file list ===")
    train_files = build_file_list(
        args.train_protocol, args.train_audio_dir,
        max_bonafide=args.train_max_bonafide, max_spoof=args.train_max_spoof,
    )

    print("\n=== Building dev file list ===")
    dev_files = build_file_list(
        args.dev_protocol, args.dev_audio_dir,
        max_bonafide=args.dev_max_bonafide, max_spoof=args.dev_max_spoof,
    )

    # --- Cache features (no-op for anything already cached) ---
    print("\n=== Caching train features ===")
    cache_features(train_files, args.train_cache_dir)
    print("\n=== Caching dev features ===")
    cache_features(dev_files, args.dev_cache_dir)

    # --- Datasets / loaders ---
    train_ds = SpoofDataset(train_files, args.train_cache_dir)
    dev_ds = SpoofDataset(dev_files, args.dev_cache_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- Model / optimizer / loss ---
    model = SpoofCNN().to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # --- Training loop with early stopping on val EER ---
    best_eer = float("inf")
    epochs_without_improvement = 0
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Training: up to {args.epochs} epochs, patience={args.patience} ===\n")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = run_epoch_train(model, train_loader, optimizer, loss_fn, device,
                                      max_grad_norm=args.max_grad_norm)
        val_loss, val_report = run_epoch_eval(model, dev_loader, loss_fn, device)

        elapsed = time.time() - t0
        val_eer = val_report["eer"]

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_acc={val_report['accuracy']:.4f} | val_f1={val_report['f1']:.4f} | "
              f"val_EER={val_eer*100:.2f}% | {elapsed:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_accuracy": val_report["accuracy"], "val_precision": val_report["precision"],
            "val_recall": val_report["recall"], "val_f1": val_report["f1"], "val_eer": val_eer,
        })

        if val_eer < best_eer:
            best_eer = val_eer
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_eer": val_eer,
                "val_report": val_report,
                "config": {
                    "n_mels": config.N_MELS,
                    "n_frames": config.N_FRAMES,
                    "sample_rate": config.SAMPLE_RATE,
                },
            }, checkpoint_path)
            print(f"  -> New best val EER ({val_eer*100:.2f}%). Saved to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement ({epochs_without_improvement}/{args.patience})")
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    print(f"\n=== Training complete. Best val EER: {best_eer*100:.2f}% ===")
    print(f"Best checkpoint saved at: {checkpoint_path}")
    return history, best_eer


def build_arg_parser():
    p = argparse.ArgumentParser(description="Train the voice deepfake detection CNN")

    p.add_argument("--train_protocol", required=True)
    p.add_argument("--train_audio_dir", required=True)
    p.add_argument("--dev_protocol", required=True)
    p.add_argument("--dev_audio_dir", required=True)

    p.add_argument("--train_cache_dir", default="data/features/train")
    p.add_argument("--dev_cache_dir", default="data/features/dev")

    p.add_argument("--train_max_bonafide", type=int, default=None)
    p.add_argument("--train_max_spoof", type=int, default=None)
    p.add_argument("--dev_max_bonafide", type=int, default=None)
    p.add_argument("--dev_max_spoof", type=int, default=None)

    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                    help="Gradient clipping threshold; stabilizes training against loss spikes")
    p.add_argument("--checkpoint_path", default=config.BEST_CHECKPOINT)

    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)