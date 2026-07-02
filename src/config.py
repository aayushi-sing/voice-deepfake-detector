"""
Central configuration for the Real-Time Voice Deepfake Detection System.
Keep every hyperparameter here so train/eval/predict/streaming all agree.
"""

# --- Audio ---
SAMPLE_RATE = 16000          # Hz. ASVspoof-style setups commonly use 16kHz.
CLIP_DURATION = 4.0          # seconds. Fixed-length window used for training/eval.
N_MELS = 128                 # mel bands
N_FFT = 1024
HOP_LENGTH = 256
FMIN = 20
FMAX = SAMPLE_RATE // 2

# Derived: number of time frames a CLIP_DURATION-long clip produces
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)
N_FRAMES = 1 + CLIP_SAMPLES // HOP_LENGTH

# --- Streaming / real-time ---
CHUNK_DURATION = 1.5         # seconds per streamed chunk
CHUNK_OVERLAP = 0.5          # seconds of overlap between consecutive chunks

# --- Labels ---
# 0 = real / bonafide, 1 = fake / spoof  (matches ASVspoof convention)
LABEL_REAL = 0
LABEL_FAKE = 1
LABEL_NAMES = {LABEL_REAL: "REAL", LABEL_FAKE: "FAKE"}

# --- Training ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
VAL_SPLIT = 0.15
RANDOM_SEED = 42

# --- Paths ---
CHECKPOINT_DIR = "checkpoints"
BEST_CHECKPOINT = "checkpoints/best_model.pt"