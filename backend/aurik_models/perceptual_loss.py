"""
Perceptual Loss für AurikEnhancerNet
====================================

Kombiniert klassische und psychoakustische Metriken:
- VGGish-Feature-Loss (Perceptual)
- SI-SDR (Signal-to-Distortion)
- PESQ (Speech Quality)
- Spectral Loss (STFT/L1)

Author: Aurik KI-Team 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Optional: VGGish-Feature-Extractor (Platzhalter)
class DummyVGGish(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(100, 10)

    def forward(self, x):
        # x: (B, 1, T) → Dummy-Feature
        return self.dummy(x[..., :100].mean(-1))


def stft_loss(x, y):
    # Einfache STFT-L1-Loss
    X = torch.stft(x.squeeze(1), n_fft=1024, return_complex=True)
    Y = torch.stft(y.squeeze(1), n_fft=1024, return_complex=True)
    return F.l1_loss(torch.abs(X), torch.abs(Y))


def perceptual_loss(x, y, vggish=None, alpha=0.5):
    # Kombiniert Feature-Loss und STFT-Loss
    loss = stft_loss(x, y)
    if vggish is not None:
        feat_x = vggish(x)
        feat_y = vggish(y)
        loss += alpha * F.mse_loss(feat_x, feat_y)
    return loss


# SI-SDR, PESQ etc. können als weitere Komponenten ergänzt werden
