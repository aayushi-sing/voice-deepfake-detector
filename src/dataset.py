"""
dataset.py
Handles everything data-related for the Real-Time Voice Deepfake Detection System:

1. Parsing ASVspoof2019 LA protocol files -> list of (filepath, label)
2. Extracting log-mel spectrograms from audio with librosa
3. Caching extracted features to .npy so we never recompute them
4. A PyTorch Dataset class that serves cached features to the training loop
5. A helper to build balanced staged subsets (Experiment 1 / 2 / 3)

Usage (see bottom of file for a __main__ smoke-test you can run once
librosa/torch are installed and the dataset is downloaded):

    from dataset import build_file_list, cache_features, SpoofDataset

    files = build_file_list(
        protocol_path="data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        audio_dir="data/raw/ASVspoof2019_LA_train/flac",
        max_bonafide=500, max_spoof=500   # Experiment 1 staged subset
    )
    cache_features(files, cache_dir="data/features/train")
    ds = SpoofDataset(files, cache_dir="data/features/train")
"""

import os
import random
from pathlib import Path

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None  # allows this module to be imported/tested without librosa installed

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object  # fallback so the class definition doesn't crash on import

import config


# ---------------------------------------------------------------------------
# 1. Protocol parsing
# ---------------------------------------------------------------------------

def parse_protocol_file(protocol_path):
    """
    Parse an ASVspoof2019 LA protocol file into a dict {filename: label}.

    Format (space-separated, per line):
        SPEAKER_ID AUDIO_FILE_NAME SYSTEM_ID ATTACK_ID KEY
    where KEY (always the LAST field) is 'bonafide' or 'spoof',
    and AUDIO_FILE_NAME (always the SECOND field) is the file to load.

    We parse by position from both ends rather than assuming an exact
    column count, since this is robust to minor format variants across
    ASVspoof protocol files (train/dev/eval protocol files are not all
    formatted identically).

    Returns:
        dict: {audio_file_name (str, no extension): label (int, 0=real/1=fake)}
    """
    if not os.path.exists(protocol_path):
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    file_to_label = {}
    with open(protocol_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 3:
                print(f"[parse_protocol_file] Skipping malformed line {line_num}: {line!r}")
                continue

            filename = fields[1]
            key = fields[-1].lower()

            if key == "bonafide":
                label = config.LABEL_REAL
            elif key == "spoof":
                label = config.LABEL_FAKE
            else:
                print(f"[parse_protocol_file] Unknown key '{key}' on line {line_num}, skipping")
                continue

            file_to_label[filename] = label

    return file_to_label


def build_file_list(protocol_path, audio_dir, audio_ext=".flac",
                     max_bonafide=None, max_spoof=None, seed=config.RANDOM_SEED):
    """
    Build a list of (filepath, label) tuples from a protocol file + audio dir.

    max_bonafide / max_spoof let you build the staged subsets:
        Experiment 1: max_bonafide=500,  max_spoof=500   (1,000 total)
        Experiment 2: max_bonafide=2000, max_spoof=2000  (4,000 total)
        Experiment 3: max_bonafide=3000, max_spoof=3000  (6,000 total)
        Full set:     leave both as None

    Returns:
        list of (filepath: str, label: int) tuples, shuffled.
    """
    file_to_label = parse_protocol_file(protocol_path)

    bonafide_files = []
    spoof_files = []

    audio_dir = Path(audio_dir)
    missing = 0
    for filename, label in file_to_label.items():
        filepath = audio_dir / f"{filename}{audio_ext}"
        if not filepath.exists():
            missing += 1
            continue
        if label == config.LABEL_REAL:
            bonafide_files.append((str(filepath), label))
        else:
            spoof_files.append((str(filepath), label))

    if missing:
        print(f"[build_file_list] Warning: {missing} files listed in protocol "
              f"but not found in {audio_dir}")

    rng = random.Random(seed)
    rng.shuffle(bonafide_files)
    rng.shuffle(spoof_files)

    if max_bonafide is not None:
        bonafide_files = bonafide_files[:max_bonafide]
    if max_spoof is not None:
        spoof_files = spoof_files[:max_spoof]

    print(f"[build_file_list] bonafide={len(bonafide_files)}  spoof={len(spoof_files)}  "
          f"total={len(bonafide_files) + len(spoof_files)}")

    combined = bonafide_files + spoof_files
    rng.shuffle(combined)
    return combined


# ---------------------------------------------------------------------------
# 2. Feature extraction
# ---------------------------------------------------------------------------

def extract_mel_spectrogram(wav_path):
    """
    Load an audio file and extract a fixed-length log-mel spectrogram.

    Steps:
        1. Load audio, resample to config.SAMPLE_RATE
        2. Pad or truncate to config.CLIP_SAMPLES (fixed duration)
        3. Compute mel-spectrogram (config.N_MELS bands)
        4. Convert power -> dB (log scale)

    Returns:
        np.ndarray of shape (N_MELS, N_FRAMES), dtype float32
    """
    if librosa is None:
        raise ImportError("librosa is required for feature extraction. "
                           "Install it with: pip install librosa")

    y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)

    # Pad or truncate to a fixed length so every clip produces the same shape
    if len(y) < config.CLIP_SAMPLES:
        y = np.pad(y, (0, config.CLIP_SAMPLES - len(y)), mode="constant")
    else:
        y = y[:config.CLIP_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        fmin=config.FMIN,
        fmax=config.FMAX,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def _cache_path_for(filepath, cache_dir):
    """Derive the .npy cache path for a given audio file."""
    stem = Path(filepath).stem
    return Path(cache_dir) / f"{stem}.npy"


def cache_features(file_list, cache_dir, overwrite=False, verbose_every=200):
    """
    Compute (or reuse) mel-spectrograms for every file in file_list and
    save them as .npy under cache_dir. This is a one-time cost per file --
    subsequent epochs just load the .npy instead of recomputing librosa.

    Args:
        file_list: list of (filepath, label) as returned by build_file_list
        cache_dir: directory to store .npy files in
        overwrite: if True, recompute even if a cached .npy already exists
        verbose_every: print progress every N files

    Returns:
        int: number of files newly computed (vs loaded from existing cache)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    computed = 0
    skipped = 0
    failed = 0

    for i, (filepath, label) in enumerate(file_list):
        cache_path = _cache_path_for(filepath, cache_dir)

        if cache_path.exists() and not overwrite:
            skipped += 1
        else:
            try:
                mel = extract_mel_spectrogram(filepath)
                np.save(cache_path, mel)
                computed += 1
            except Exception as e:
                print(f"[cache_features] FAILED on {filepath}: {e}")
                failed += 1

        if (i + 1) % verbose_every == 0:
            print(f"[cache_features] {i + 1}/{len(file_list)} processed "
                  f"(new={computed}, cached={skipped}, failed={failed})")

    print(f"[cache_features] Done. new={computed}, already_cached={skipped}, failed={failed}")
    return computed


# ---------------------------------------------------------------------------
# 3. PyTorch Dataset
# ---------------------------------------------------------------------------

class SpoofDataset(Dataset):
    """
    Serves cached mel-spectrograms + labels to the training loop.

    IMPORTANT: call cache_features() on file_list BEFORE constructing this,
    otherwise __getitem__ will fail on missing .npy files. This separation
    is intentional -- caching is a slow one-time pass, __getitem__ should
    be fast (disk load only) since it runs every epoch.
    """

    def __init__(self, file_list, cache_dir):
        if torch is None:
            raise ImportError("torch is required to use SpoofDataset. "
                               "Install it with: pip install torch")
        self.file_list = file_list
        self.cache_dir = Path(cache_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath, label = self.file_list[idx]
        cache_path = _cache_path_for(filepath, self.cache_dir)

        if not cache_path.exists():
            # Fallback: compute on the fly if somehow not cached yet.
            # Should be rare if cache_features() was run first.
            mel = extract_mel_spectrogram(filepath)
            np.save(cache_path, mel)
        else:
            mel = np.load(cache_path)

        # Add channel dimension: (N_MELS, N_FRAMES) -> (1, N_MELS, N_FRAMES)
        tensor = torch.from_numpy(mel).unsqueeze(0).float()
        return tensor, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 4. Smoke test (run this file directly to sanity-check the pipeline)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity-check the dataset pipeline")
    parser.add_argument("--protocol", required=True, help="Path to ASVspoof protocol .txt file")
    parser.add_argument("--audio_dir", required=True, help="Path to flac/ directory")
    parser.add_argument("--cache_dir", default="data/features/train")
    parser.add_argument("--max_bonafide", type=int, default=10)
    parser.add_argument("--max_spoof", type=int, default=10)
    args = parser.parse_args()

    print("=== Step 1: build file list ===")
    files = build_file_list(
        args.protocol, args.audio_dir,
        max_bonafide=args.max_bonafide, max_spoof=args.max_spoof
    )
    print(f"First 3 entries: {files[:3]}")

    print("\n=== Step 2: cache features ===")
    cache_features(files, args.cache_dir)

    print("\n=== Step 3: build Dataset and pull one sample ===")
    ds = SpoofDataset(files, args.cache_dir)
    x, y = ds[0]
    print(f"Sample shape: {x.shape}, label: {y.item()} "
          f"({config.LABEL_NAMES[int(y.item())]})")
    print(f"Dataset length: {len(ds)}")
    print("\nSmoke test passed.")