"""
predict.py
Single-file inference for the Real-Time Voice Deepfake Detection System.

Takes one audio file (wav, flac, or anything librosa can load), runs it
through the same preprocessing used in training, and prints a Real/Fake
prediction with confidence. This is the script app.py (Streamlit) wraps,
and is also usable standalone from the command line.

Usage:
    python src/predict.py --checkpoint checkpoints/exp3_best.pt --audio path/to/clip.wav
"""

import argparse

import torch

import config
from dataset import extract_mel_spectrogram
from model import SpoofCNN


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SpoofCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_file(model, audio_path, device):
    """
    Run inference on a single audio file.

    Returns:
        dict with keys: label (str, "REAL" or "FAKE"), label_id (int),
        confidence (float, 0-1 -- probability of the predicted class),
        fake_probability (float, 0-1 -- raw model output, P(fake))
    """
    mel = extract_mel_spectrogram(audio_path)          # (N_MELS, N_FRAMES)
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(device)  # (1,1,N_MELS,N_FRAMES)

    with torch.no_grad():
        fake_prob = model.predict_proba(x).item()       # P(fake), since label 1 = FAKE

    if fake_prob >= 0.5:
        label_id = config.LABEL_FAKE
        confidence = fake_prob
    else:
        label_id = config.LABEL_REAL
        confidence = 1.0 - fake_prob

    return {
        "label": config.LABEL_NAMES[label_id],
        "label_id": label_id,
        "confidence": confidence,
        "fake_probability": fake_prob,
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    result = predict_file(model, args.audio, device)

    print(f"\nFile:       {args.audio}")
    print(f"Prediction: {result['label']}  (confidence: {result['confidence']*100:.1f}%)")
    print(f"P(fake):    {result['fake_probability']*100:.1f}%")

    return result


def build_arg_parser():
    p = argparse.ArgumentParser(description="Predict Real/Fake for a single audio file")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--audio", required=True, help="Path to a .wav/.flac file")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)
