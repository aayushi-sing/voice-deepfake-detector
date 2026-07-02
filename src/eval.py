"""
eval.py
Standalone evaluation for the Real-Time Voice Deepfake Detection System.

Loads a saved checkpoint (from train.py) and runs it against any protocol +
audio_dir you point it at, producing the final Accuracy/Precision/Recall/F1/
EER report -- independent of the training loop. Use this to:
  - Re-report Experiment 3's dev-set numbers cleanly for your README
  - Evaluate on a *different* split later (e.g. if you download the eval
    partition for a generalization check)

Usage:
    python src/eval.py \
        --checkpoint checkpoints/exp3_best.pt \
        --protocol data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
        --audio_dir data/raw/ASVspoof2019_LA_dev/flac \
        --max_bonafide 750 --max_spoof 750 \
        --output_report outputs/metrics_report.txt
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from dataset import build_file_list, cache_features, SpoofDataset
from model import SpoofCNN
from metrics import full_report


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SpoofCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
          f"(val_eer at save time: {checkpoint.get('val_eer', float('nan')) * 100:.2f}%)")
    return model, checkpoint


def evaluate(model, loader, device):
    """Run the model over a DataLoader, return (y_true, y_scores) lists."""
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = model.predict_proba(x)
            all_scores.extend(probs.cpu().numpy().flatten().tolist())
            all_labels.extend(y.numpy().flatten().tolist())

    return all_labels, all_scores


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, checkpoint = load_model(args.checkpoint, device)

    print("\n=== Building evaluation file list ===")
    files = build_file_list(
        args.protocol, args.audio_dir,
        max_bonafide=args.max_bonafide, max_spoof=args.max_spoof,
    )

    print("\n=== Caching features (skips anything already cached) ===")
    cache_features(files, args.cache_dir)

    ds = SpoofDataset(files, args.cache_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\n=== Running inference on {len(ds)} clips ===")
    y_true, y_scores = evaluate(model, loader, device)

    report = full_report(y_true, y_scores)
    print("\n" + report["text"])

    if args.output_report:
        out_path = Path(args.output_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Evaluated on: {args.protocol}\n")
            f.write(f"Num clips: {len(ds)} "
                    f"(bonafide={args.max_bonafide}, spoof={args.max_spoof})\n\n")
            f.write(report["text"])
        print(f"\nReport saved to {out_path}")

    return report


def build_arg_parser():
    p = argparse.ArgumentParser(description="Evaluate a saved checkpoint")

    p.add_argument("--checkpoint", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--cache_dir", default="data/features/eval")

    p.add_argument("--max_bonafide", type=int, default=None)
    p.add_argument("--max_spoof", type=int, default=None)

    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--output_report", default="outputs/metrics_report.txt")

    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)
