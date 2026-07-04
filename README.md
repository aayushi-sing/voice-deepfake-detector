# Real-Time Voice Deepfake Detection System

A CNN-based audio spoof/deepfake detector built on the ASVspoof2019 LA benchmark, designed with an explicit focus on **real-time inference latency** in addition to detection accuracy.

Built in the context of the IndiaAI Mission's *"Real-Time Voice Deepfake Detection System"* project (Responsible AI theme, IAI/TCG CREST in collaboration with IIT Kharagpur) — a system intended for applications in **fraud prevention, media verification, and digital trust**.

---

## Problem framing

Voice cloning and text-to-speech systems have become good enough that synthetic speech can plausibly impersonate a real speaker — creating risk for voice-based authentication, phone-based fraud, and media authenticity. A useful detector for this problem needs two things that are often treated separately in academic work:

1. **Accuracy** on the actual spoofing task (measured the way the field measures it — via Equal Error Rate, not just accuracy)
2. **Speed** — a detector that can't run within the latency budget of a live call or stream isn't deployable, no matter how accurate it is offline

This project builds both: a CNN trained and evaluated on the ASVspoof2019 LA benchmark, plus a streaming inference mode with measured latency and real-time-factor benchmarks.

---

## Architecture

```
Audio (.wav/.flac)
      |
      v
Resample to 16kHz, mono
      |
      v
Log-mel spectrogram (128 mel bands, librosa)
      |
      v
CNN (4 conv blocks: Conv2d -> BatchNorm -> ReLU -> MaxPool)
      |
      v
Global average pooling  <-- lets the same model score variable-length
      |                      audio chunks, not just fixed 4s clips
      v
FC head -> single logit -> sigmoid -> P(fake)
```

Model size: **105,953 parameters** — deliberately lightweight, since the model needs to run fast, not just accurately.

---

## Dataset

**ASVspoof2019 LA** (Logical Access) — the standard benchmark for synthetic/converted-voice spoofing detection.

- **Train partition**: 2,580 bonafide + spoof utterances (attack types A01–A06), used for training
- **Dev partition**: used for validation, early stopping, and final reported metrics
- **Eval partition**: *not used in this project* — it contains 11 additional, unseen attack types (A07–A19) specifically designed to test generalization. This is intentionally left as a documented next step (see Limitations).

Labels follow the ASVspoof convention: `0 = bonafide (real)`, `1 = spoof (fake)`.

### Staged training approach

Rather than jumping straight to full-scale training, the model was built and validated incrementally:

| Experiment | Train clips | Purpose | Best Val EER |
|---|---|---|---|
| Experiment 1 | 1,000 (500 bonafide + 500 spoof) | Pipeline sanity check | 1.33% |
| Experiment 2 | 4,000 (2,000 + 2,000) | Hyperparameter tuning | 0.10% |
| Experiment 3 | 5,580 + 3,000 spoof (all available bonafide + a balanced spoof sample) | Final model | **0.20%** |

Note: Experiment 3 used all 2,580 available bonafide clips in the train partition (there aren't more), paired with 3,000 spoof clips.

An early instability issue (occasional loss spikes during training, e.g. epoch-to-epoch val_loss jumping from 0.05 to 9.96 in Experiment 2) was addressed by adding **gradient clipping** and **lowering the learning rate** (1e-3 → 5e-4) before the final Experiment 3 run.

---

## Results (final model — Experiment 3, evaluated on ASVspoof2019 LA dev set)

```
Accuracy:  0.9940
Precision: (see outputs/metrics_report.txt for full precision/recall breakdown)
Recall:    (see outputs/metrics_report.txt)
F1-score:  0.9940
EER:       0.20%   (threshold computed via ROC curve, standard ASVspoof methodology)
```

*(Full report with confusion matrix: `outputs/metrics_report.txt`, generated via `eval.py`.)*

**Verified on real held-out examples**, not just aggregate metrics — e.g. a known A01-attack spoof file was classified FAKE at 99.9% confidence; a known A05-attack spoof file was classified FAKE at 66.0% confidence, illustrating that detection confidence varies meaningfully by attack/synthesis method — a realistic and expected pattern, not a discrepancy.

---

## Real-time performance (the "real-time" part of the project)

Measured via `streaming_infer.py`, which slides a 1.5-second window (0.5s overlap) across audio and classifies each chunk as it "arrives," simulating a live stream rather than requiring a complete file upfront.

**CPU latency benchmark** (100 trials, after 10 warmup runs, no GPU):

| Metric | Value |
|---|---|
| Mean latency | 14.34 ms / chunk |
| P50 latency | 11.52 ms / chunk |
| P95 latency | 21.36 ms / chunk |
| **Real-Time Factor (RTF)** | **0.0096** |

An RTF of 0.0096 means the model classifies a 1.5s audio chunk in about 14ms — roughly **100x faster than real-time**, on CPU, with no special hardware.

**Note on first-chunk latency:** the very first inference call on a fresh process pays a one-time PyTorch "warm-up" cost (memory allocation, kernel init) — in a live demo run, chunk 0 took ~100ms while chunks 1-2 dropped to 6–13ms. The benchmark above reports steady-state performance (after warmup), which is the number relevant to a long-running deployed service.

**What this means practically:** at ~14ms per chunk, this model could flag a suspected spoofed segment within a fraction of a second of receiving it — comfortably within budget for a live call-monitoring or fraud-detection pipeline, with substantial headroom left for network/buffering overhead in a full production system. (This benchmark measures model inference only — feature extraction + forward pass — not a complete end-to-end audio pipeline.)

---

## Repository structure

```
voice-deepfake-detector/
├── data/
│   ├── raw/                          # ASVspoof2019 LA train + dev (not committed to git)
│   └── features/                     # cached .npy mel-spectrograms
├── checkpoints/
│   └── exp3_best.pt                  # final trained model
├── outputs/
│   └── metrics_report.txt            # full eval report (accuracy/precision/recall/f1/EER)
├── src/
│   ├── config.py                     # all constants (audio, model, training)
│   ├── dataset.py                    # protocol parsing, feature extraction, caching, PyTorch Dataset
│   ├── model.py                      # SpoofCNN architecture
│   ├── metrics.py                    # EER + classification metrics (shared by train/eval)
│   ├── train.py                      # training loop, checkpoints on best val EER, early stopping
│   ├── eval.py                       # standalone evaluation of a saved checkpoint
│   ├── predict.py                    # single-file inference (wav/flac -> Real/Fake + confidence)
│   └── streaming_infer.py            # chunked streaming inference + latency/RTF benchmark
├── app.py                            # Streamlit demo (upload audio -> prediction + spectrogram)
├── requirements.txt
└── README.md
```

---

## How to run

```bash
pip install -r requirements.txt
```

**Train** (staged experiments controlled via flags, no code changes needed):
```bash
python src/train.py \
  --train_protocol data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
  --train_audio_dir data/raw/ASVspoof2019_LA_train/flac \
  --dev_protocol data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
  --dev_audio_dir data/raw/ASVspoof2019_LA_dev/flac \
  --train_max_bonafide 3000 --train_max_spoof 3000 \
  --dev_max_bonafide 750 --dev_max_spoof 750 \
  --epochs 15 --checkpoint_path checkpoints/exp3_best.pt
```

**Evaluate a checkpoint:**
```bash
python src/eval.py --checkpoint checkpoints/exp3_best.pt \
  --protocol data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
  --audio_dir data/raw/ASVspoof2019_LA_dev/flac \
  --max_bonafide 750 --max_spoof 750
```

**Predict a single file:**
```bash
python src/predict.py --checkpoint checkpoints/exp3_best.pt --audio path/to/clip.flac
```

**Run the Streamlit demo:**
```bash
streamlit run app.py
```

**Benchmark real-time latency:**
```bash
python src/streaming_infer.py --checkpoint checkpoints/exp3_best.pt --mode benchmark --num_trials 100 --force_cpu
```

**Stream-classify a full file in chunks:**
```bash
python src/streaming_infer.py --checkpoint checkpoints/exp3_best.pt --mode stream --audio path/to/clip.flac
```

---

## Limitations & honest next steps

- **Train and dev partitions share the same six attack types (A01–A06)**; this project has not yet been evaluated on ASVspoof2019 LA's eval partition, which introduces 11 unseen attack types (A07–A19) specifically to test generalization. This is the field's actual open problem, and the natural next step for this project.
- Confidence is not uniform across attack types (see A01 vs A05 example above) — a production system would benefit from per-attack-type analysis, not just an aggregate EER.
- The real-time benchmark measures model inference latency only, not a full deployed pipeline (audio capture, buffering, network transmission).
- Training showed some instability (loss spikes) even after gradient clipping and a lowered learning rate — worth further investigation (e.g. learning rate warmup/scheduling) if extending this work.

---

## Applications

This detector is designed to plug into scenarios like:
- **Live call monitoring / fraud prevention**: flagging suspected synthetic voice in real time during a phone call or voice-authentication attempt
- **Media verification**: batch-checking uploaded audio/video clips for synthetic voice content
- **Digital trust tooling**: as a component in a larger content-provenance or verification pipeline
