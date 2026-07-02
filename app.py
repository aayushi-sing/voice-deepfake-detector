"""
app.py
Streamlit demo for the Real-Time Voice Deepfake Detection System.

Upload a .wav or .flac file, see the model's Real/Fake prediction, confidence,
and the mel-spectrogram the model actually looked at. This is a thin UI layer
over predict.py's predict_file() -- no logic is duplicated here.


"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import librosa
import librosa.display
import streamlit as st
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

import config
from predict import load_model, predict_file
from dataset import extract_mel_spectrogram

CHECKPOINT_PATH = "checkpoints/exp3_best.pt"


@st.cache_resource
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT_PATH, device)
    return model, device


def plot_mel_spectrogram(mel):
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(
        mel, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
        x_axis="time", y_axis="mel", ax=ax, cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Log-Mel Spectrogram (what the model sees)")
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Voice Deepfake Detector", page_icon="🎙️", layout="centered")

    st.title("🎙️ Real-Time Voice Deepfake Detection")
    st.caption(
        "Upload an audio clip to check whether it's real (bonafide) speech "
        "or AI-generated/spoofed audio. Trained on ASVspoof2019 LA."
    )

    try:
        model, device = get_model()
    except FileNotFoundError:
        st.error(
            f"Checkpoint not found at `{CHECKPOINT_PATH}`. "
            f"Train a model first (see train.py) or place a checkpoint at this path."
        )
        return

    uploaded_file = st.file_uploader("Upload a .wav or .flac file", type=["wav", "flac"])

    if uploaded_file is not None:
        # Save to a temp file since librosa/soundfile need a path or file-like object
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.audio(uploaded_file)

        with st.spinner("Analyzing..."):
            result = predict_file(model, tmp_path, device)
            mel = extract_mel_spectrogram(tmp_path)

        # --- Prediction result ---
        st.subheader("Result")
        if result["label"] == "FAKE":
            st.error(f"🚨 **FAKE** — confidence: {result['confidence']*100:.1f}%")
        else:
            st.success(f"✅ **REAL** — confidence: {result['confidence']*100:.1f}%")

        st.progress(result["fake_probability"])
        st.caption(f"P(fake) = {result['fake_probability']*100:.1f}%  "
                   f"(0% = certainly real, 100% = certainly fake)")

        # --- Spectrogram visualization ---
        st.subheader("Mel-Spectrogram")
        fig = plot_mel_spectrogram(mel)
        st.pyplot(fig)

        Path(tmp_path).unlink(missing_ok=True)  # clean up temp file

    st.divider()
    st.caption(
        
    )


if __name__ == "__main__":
    main()
