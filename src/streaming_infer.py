"""
streaming_infer.py
Real-time streaming inference and latency benchmarking for the Voice Deepfake
Detection System. This is the Tier 2 / "real-time" differentiator: instead of
requiring a full audio file, it slides a short window (e.g. 1.5s) across the
signal and scores each chunk as it "arrives" -- simulating a live call/stream.

Two things this script produces, both worth reporting in your README:
  1. A per-chunk prediction timeline for a given audio file (streaming demo)
  2. A latency benchmark: ms/chunk and RTF (Real-Time Factor) on CPU --
     RTF < 1.0 means the model classifies audio FASTER than it plays back,
     i.e. it can genuinely run live rather than falling behind.

Usage:
    # Streaming demo on one file
    python src/streaming_infer.py --checkpoint checkpoints/exp3_best.pt \
        --audio path/to/clip.flac --mode stream

    # Latency benchmark (CPU, averaged over many chunks)
    python src/streaming_infer.py --checkpoint checkpoints/exp3_best.pt \
        --mode benchmark --num_trials 100

    # Latency benchmark with dynamic quantization applied (optimization pass)
    python src/streaming_infer.py --checkpoint checkpoints/exp3_best.pt \
        --mode benchmark --num_trials 100 --quantize
"""

import argparse
import time

import numpy as np
import librosa
import torch

import config
from model import SpoofCNN


# ---------------------------------------------------------------------------
# Chunk-level feature extraction
# ---------------------------------------------------------------------------

def extract_chunk_mel(y_chunk, sr=config.SAMPLE_RATE):
    """
    Extract a log-mel spectrogram for a single audio chunk (not padded/
    truncated to the full CLIP_DURATION like training clips -- chunks can be
    shorter). This works because SpoofCNN uses global average pooling, so it
    accepts variable-length spectrograms.
    """
    mel = librosa.feature.melspectrogram(
        y=y_chunk, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS, fmin=config.FMIN, fmax=config.FMAX,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def load_model(checkpoint_path, device, quantize=False):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SpoofCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if quantize:
        # Dynamic quantization: converts Linear layer weights to int8 at
        # inference time. Conv2d isn't supported by dynamic quantization in
        # PyTorch (only Linear/LSTM are), so this optimizes the classifier
        # head. Must run on CPU -- quantized ops aren't CUDA-backed.
        model = model.to("cpu")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("Applied dynamic quantization (Linear layers -> int8).")

    return model


# ---------------------------------------------------------------------------
# Mode 1: streaming demo on a real audio file
# ---------------------------------------------------------------------------

def stream_file(model, audio_path, device, chunk_duration=config.CHUNK_DURATION,
                 overlap=config.CHUNK_OVERLAP):
    """
    Slide a window across audio_path and classify each chunk, printing a
    per-chunk timeline with latency. Returns the aggregated (mean-pooled)
    final prediction across all chunks.
    """
    y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
    audio_duration = len(y) / sr

    chunk_samples = int(chunk_duration * sr)
    stride_samples = int((chunk_duration - overlap) * sr)
    if stride_samples <= 0:
        raise ValueError("overlap must be smaller than chunk_duration")

    print(f"\nStreaming '{audio_path}' "
          f"({audio_duration:.2f}s audio, {chunk_duration}s chunks, {overlap}s overlap)\n")
    print(f"{'Chunk':>6} {'Start(s)':>9} {'End(s)':>8} {'P(fake)':>9} {'Latency(ms)':>13}")

    chunk_scores = []
    chunk_latencies = []
    start = 0
    chunk_idx = 0

    while start < len(y):
        end = min(start + chunk_samples, len(y))
        y_chunk = y[start:end]

        # Pad the final (short) chunk so the mel extraction has a sane minimum length
        if len(y_chunk) < sr * 0.25:  # skip trailing slivers < 0.25s
            break
        if len(y_chunk) < chunk_samples:
            y_chunk = np.pad(y_chunk, (0, chunk_samples - len(y_chunk)))

        t0 = time.perf_counter()
        mel = extract_chunk_mel(y_chunk, sr)
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            prob_fake = torch.sigmoid(model(x)).item()
        latency_ms = (time.perf_counter() - t0) * 1000

        chunk_scores.append(prob_fake)
        chunk_latencies.append(latency_ms)

        print(f"{chunk_idx:>6} {start/sr:>9.2f} {end/sr:>8.2f} {prob_fake:>9.3f} {latency_ms:>13.2f}")

        start += stride_samples
        chunk_idx += 1

    mean_prob = float(np.mean(chunk_scores))
    final_label = config.LABEL_NAMES[config.LABEL_FAKE if mean_prob >= 0.5 else config.LABEL_REAL]

    total_latency = sum(chunk_latencies)
    rtf = total_latency / 1000 / audio_duration

    print(f"\nAggregated prediction: {final_label}  (mean P(fake)={mean_prob:.3f})")
    print(f"Total processing time: {total_latency:.1f} ms  |  Audio duration: {audio_duration:.2f}s")
    print(f"Real-Time Factor (RTF): {rtf:.4f}  "
          f"({'FASTER than real-time' if rtf < 1.0 else 'SLOWER than real-time'})")

    return {
        "chunk_scores": chunk_scores,
        "chunk_latencies_ms": chunk_latencies,
        "final_label": final_label,
        "mean_prob_fake": mean_prob,
        "rtf": rtf,
    }


# ---------------------------------------------------------------------------
# Mode 2: latency benchmark (no real audio needed -- synthetic chunks are
# fine here since we're measuring compute time, not accuracy)
# ---------------------------------------------------------------------------

def benchmark_latency(model, device, chunk_duration=config.CHUNK_DURATION,
                       num_trials=100, warmup=10):
    """
    Measure average per-chunk latency and RTF over many trials on synthetic
    audio. Synthetic input is valid here because we're benchmarking compute
    time (feature extraction + forward pass), which doesn't depend on
    whether the audio content is real or fake.
    """
    sr = config.SAMPLE_RATE
    chunk_samples = int(chunk_duration * sr)

    rng = np.random.default_rng(42)

    def run_one():
        y_chunk = rng.uniform(-1, 1, size=chunk_samples).astype(np.float32)
        t0 = time.perf_counter()
        mel = extract_chunk_mel(y_chunk, sr)
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            _ = torch.sigmoid(model(x)).item()
        return (time.perf_counter() - t0) * 1000

    print(f"Warming up ({warmup} runs)...")
    for _ in range(warmup):
        run_one()

    print(f"Benchmarking ({num_trials} runs, {chunk_duration}s chunks, device={device})...")
    latencies = [run_one() for _ in range(num_trials)]

    latencies = np.array(latencies)
    mean_ms = latencies.mean()
    p50_ms = np.percentile(latencies, 50)
    p95_ms = np.percentile(latencies, 95)
    rtf = (mean_ms / 1000) / chunk_duration

    print(f"\n=== Latency Benchmark Results ===")
    print(f"Device:          {device}")
    print(f"Chunk duration:  {chunk_duration}s")
    print(f"Mean latency:    {mean_ms:.2f} ms")
    print(f"P50 latency:     {p50_ms:.2f} ms")
    print(f"P95 latency:     {p95_ms:.2f} ms")
    print(f"RTF:             {rtf:.4f}  "
          f"({'FASTER than real-time -- can run live' if rtf < 1.0 else 'SLOWER than real-time'})")
    print(f"\nInterpretation: at {mean_ms:.1f} ms per {chunk_duration}s chunk, this model "
          f"could comfortably flag a spoofed clip within roughly {mean_ms/1000:.2f}s of "
          f"receiving it -- well within the latency budget for a live call-monitoring "
          f"or fraud-detection use case.")

    return {"mean_ms": mean_ms, "p50_ms": p50_ms, "p95_ms": p95_ms, "rtf": rtf}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(description="Streaming inference / latency benchmark")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", choices=["stream", "benchmark"], default="benchmark")
    p.add_argument("--audio", help="Required for --mode stream")
    p.add_argument("--chunk_duration", type=float, default=config.CHUNK_DURATION)
    p.add_argument("--overlap", type=float, default=config.CHUNK_OVERLAP)
    p.add_argument("--num_trials", type=int, default=100, help="For --mode benchmark")
    p.add_argument("--quantize", action="store_true",
                    help="Apply dynamic quantization before benchmarking (forces CPU)")
    p.add_argument("--force_cpu", action="store_true",
                    help="Benchmark on CPU even if a GPU is available (realistic for edge deployment)")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    if args.quantize or args.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, device, quantize=args.quantize)

    if args.mode == "stream":
        if not args.audio:
            raise ValueError("--audio is required for --mode stream")
        stream_file(model, args.audio, device, args.chunk_duration, args.overlap)
    else:
        benchmark_latency(model, device, args.chunk_duration, args.num_trials)
