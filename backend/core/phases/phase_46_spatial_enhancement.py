"""
Phase 46: Spatial Enhancement v2.0 — Haas-Effekt + Allpass-Diffusion
=====================================================================

Vollständige DSP-Implementierung ohne aurik_ml.
Ersetzt den kaputten ML-Stub.

ALGORITHMUS:
  1. Mono-Check: Mono-Signale passieren unverändert (kein Stereo-Bild vorhanden).

  2. Haas-Effekt (Inter-Aural Time Difference / IATD):
     - L-Kanal erhält ein leichtes Delay (0.5 ms)
     - Erzeugt wahrgenommene Breite durch ITD ohne Spektralfärbung
       (unter Hochman-Grenze von ~35 ms → kein Kammer-Echo)

  3. Mid/Side + Allpass-Diffusion:
     - M/S Enkodierung
     - S-Kanal durch Schroeder-Allpass (25 ms, g=0.5)
     - Erweitert Stereo-Bild ohne Phasenlöschungen bei Mono-Sum

  4. Normalisierungs-Pass: Pegel-Erhalt nach Haas + Allpass

HINWEIS:
  - Mono wird unverändert zurückgegeben
  - Keineaurik_ml-Imports

Author: Aurik Development Team
Version: 2.0.0
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.signal as sig

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

_HAAS_DELAY_MS = 0.5  # ITD-Delay auf L-Kanal  (< Hochman 35 ms → kein Echo)
_ALLPASS_DELAY_MS = 25.0  # Schroeder Allpass auf Side
_ALLPASS_GAIN = 0.50


def _haas_delay(L: np.ndarray, R: np.ndarray, sr: int, delay_ms: float):
    """Verzögert L um delay_ms; R unverändert."""
    delay_s = max(1, int(delay_ms / 1000.0 * sr))
    L_delayed = np.concatenate([np.zeros(delay_s), L[:-delay_s]])
    return L_delayed, R


def _allpass_filter(signal: np.ndarray, sr: int, delay_ms: float, g: float) -> np.ndarray:
    """Schroeder-Allpass-Filter: H(z) = (-g + z^{-D}) / (1 - g*z^{-D})."""
    D = max(1, int(delay_ms / 1000.0 * sr))
    b = np.zeros(D + 1)
    b[0] = -g
    b[-1] = 1.0
    a = np.zeros(D + 1)
    a[0] = 1.0
    a[-1] = -g
    return sig.lfilter(b, a, signal)


class SpatialEnhancementPhase(PhaseInterface):
    """Haas-Effekt + M/S-Allpass-Diffusion für erweitertes Stereobild."""

    phase_id = "phase_46_spatial_enhancement"
    name = "Spatial Enhancement (Haas + M/S Allpass)"
    description = (
        "Stereobild-Erweiterung via Haas-Effekt (0.5 ms ITD) und "
        "M/S-Allpass-Diffusion auf dem Side-Kanal (25 ms, g=0.5). "
        "Mono: Passthrough. Kein aurik_ml."
    )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self.phase_id,
            name=self.name,
            category=PhaseCategory.STEREO,
            priority=3,
            version="2.0.0",
            dependencies=[],
            estimated_time_factor=0.04,
            memory_requirement_mb=60,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.85,
            description=self.description,
        )

    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> PhaseResult:
        """
        Räumliche Erweiterung des Stereobildes.

        Args:
            audio:        Mono oder Stereo (float32/float64)
            sample_rate:  Hz
            **kwargs:     haas_ms     (float, default 0.5)  — ITD Delay L-Kanal ms
                          allpass_ms  (float, default 25.0) — Allpass Delay Side ms
                          allpass_g   (float, default 0.50) — Allpass Gain (0-<1)
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        self.validate_input(audio)
        t0 = time.time()

        haas_ms: float = float(kwargs.get("haas_ms", _HAAS_DELAY_MS))
        allpass_ms: float = float(kwargs.get("allpass_ms", _ALLPASS_DELAY_MS))
        allpass_g: float = float(kwargs.get("allpass_g", _ALLPASS_GAIN))

        if audio.ndim == 1:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio,
                execution_time_seconds=time.time() - t0,
                metadata={"skipped": "mono_input"},
                metrics={},
            )

        x = audio.astype(np.float64)
        L = x[:, 0]
        R = x[:, 1]

        # 1. Haas-Effekt: L leicht verzögern
        L_h, R_h = _haas_delay(L, R, sample_rate, haas_ms)

        # 2. M/S Enkодierung
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        M = (L_h + R_h) * inv_sqrt2
        S = (L_h - R_h) * inv_sqrt2

        # 3. Allpass-Diffusion auf Side
        S_diff = _allpass_filter(S, sample_rate, allpass_ms, allpass_g)

        # 4. Rück-Dekodierung
        L_out = (M + S_diff) * inv_sqrt2
        R_out = (M - S_diff) * inv_sqrt2
        processed = np.column_stack([L_out, R_out])

        # 5. Normalisierung: Pegel-Erhalt
        peak_in = float(np.max(np.abs(audio)))
        peak_out = float(np.max(np.abs(processed)))
        if peak_out > 1e-8 and peak_in > 1e-8:
            processed = processed * (peak_in / peak_out)

        processed = np.clip(processed, -1.0, 1.0).astype(audio.dtype)

        logger.info(
            "Phase 46 SpatialEnhancement: haas=%.1fms, allpass=%.1fms/g=%.2f",
            haas_ms,
            allpass_ms,
            allpass_g,
        )

        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.clip(processed, -1.0, 1.0)
        return PhaseResult(
            success=True,
            audio=processed,
            execution_time_seconds=time.time() - t0,
            metadata={"haas_ms": haas_ms, "allpass_ms": allpass_ms, "allpass_g": allpass_g},
            metrics={"haas_ms": haas_ms, "allpass_g": allpass_g},
        )
