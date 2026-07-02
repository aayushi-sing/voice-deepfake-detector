"""
model.py
CNN architecture for the Real-Time Voice Deepfake Detection System.

Input:  (batch, 1, N_MELS, N_FRAMES)  -- log-mel spectrogram, 1 channel
Output: (batch, 1)                    -- raw logit (apply sigmoid for probability)

We output a raw logit rather than a sigmoid probability so training can use
BCEWithLogitsLoss, which is numerically more stable than BCELoss(sigmoid(x)).
Use model.predict_proba(x) at inference time to get an actual probability.

Architecture: 4 conv blocks (Conv -> BatchNorm -> ReLU -> MaxPool), then
global average pooling (so the model isn't locked to one exact input length --
useful later for streaming chunks of a different duration than training clips),
then a small FC head.
"""

import torch
import torch.nn as nn

import config


class SpoofCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        self.conv_block1 = self._conv_block(in_channels=1, out_channels=16)
        self.conv_block2 = self._conv_block(in_channels=16, out_channels=32)
        self.conv_block3 = self._conv_block(in_channels=32, out_channels=64)
        self.conv_block4 = self._conv_block(in_channels=64, out_channels=128)

        # Global average pool collapses (freq, time) -> 1x1, regardless of
        # input size. This means the same model can later score audio chunks
        # of a different length than what it was trained on (relevant for
        # streaming inference in Tier 2), without architecture changes.
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # raw logit
        )

    @staticmethod
    def _conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch, 1, n_mels, n_frames)
        Returns:
            logits: tensor of shape (batch, 1)
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits

    def predict_proba(self, x):
        """Convenience method: forward pass + sigmoid -> probability of FAKE."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Smoke test: build the model and run a forward pass on a dummy batch shaped
# exactly like real data (batch=4, 1, N_MELS, N_FRAMES from config.py).
# Run this on your machine/Kaggle once torch is installed:
#     python src/model.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = SpoofCNN()
    print(f"SpoofCNN parameters: {count_parameters(model):,}")

    dummy_batch = torch.randn(4, 1, config.N_MELS, config.N_FRAMES)
    logits = model(dummy_batch)
    probs = model.predict_proba(dummy_batch)

    print(f"Input shape:  {tuple(dummy_batch.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Probs shape:  {tuple(probs.shape)}")
    print(f"Sample probs: {probs.squeeze().tolist()}")

    assert logits.shape == (4, 1), "Unexpected output shape"
    assert torch.all((probs >= 0) & (probs <= 1)), "Probabilities out of range"
    print("\nSmoke test passed.")