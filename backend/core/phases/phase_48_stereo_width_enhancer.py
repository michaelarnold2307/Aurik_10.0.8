"""
Phase 48: Stereo Width Enhancer v2.0 — M/S DSP
===============================================

Vollständige DSP-Implementierung des Stereo-Breitensteuerers ohne ML.
Ersetzt den kaputten aurik_ml.mastering-Stub.

ALGORITHMUS — Mid/Side (M/S) Processing:
  Mid   M = (L + R) / sqrt(2)   → Mono-kompatibles Zentrum
  Side  S = (L - R) / sqrt(2)   → Stereo-Information, Breite

  Stereobreite skalieren:
    S' = S × width_factor   (> 1.0 = breiter, < 1.0 = schmaler)

  Rück-Dekodierung:
    L' = (M + S') / sqrt(2)
    R' = (M - S') / sqrt(2)

  PLUS: Schroeder Allpass-Kette für Diffusion der S-Komponente →
  reduziert Kammfiltereffekte bei breitem Mix.

ANWENDUNG:
  - Mono-Signal: Passthrough (kein Stereo vorhanden)
  - Stereo: M/S + Allpass-Diffusion + width-Skalierung
  - width=1.0: unveränderter Klang

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

_DEFAULT_WIDTH = 1.25  # Leichte Verbreiterung: 25 % mehr Side-Energie
_ALLPASS_DELAYS_MS = [17.1, 19.7, 23.3]  # Schroeder Allpass-Delays (Primzahl-Verhältnis)
_ALLPASS_GAIN = 0.60


class StereoWidthEnhancerPhase(PhaseInterface):
    """M/S-basierter Stereobreiten-Enhancer mit Allpass-Diffusion."""

    phase_id = "phase_48_stereo_width_enhancer"
    name = "Stereo Width Enhancer (M/S)"
    description = (
        "M/S-Stereobreitensteuerer mit Allpass-Diffusion: "
        "Side-Kanal wird skaliert und durch drei Schroeder-Allpass-Filter "
        "diffundiert, um Kammfiltereffekte bei breitem Mix zu vermeiden."
    )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self.phase_id,
            name=self.name,
            category=PhaseCategory.STEREO,
            priority=3,
            version="2.0.0",
            dependencies=[],
            estimated_time_factor=0.02,
            memory_requirement_mb=30,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.80,
            description=self.description,
        )

    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> PhaseResult:
        """
        Erweitert Stereobild via M/S.

        Args:
            audio:       Mono oder Stereo
            sample_rate: Abtastrate Hz
            **kwargs:    width  (float, default 1.25; 1.0 = unverändert)
                         diffuse (bool, default True: Allpass-Diffusion anwenden)
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        self.validate_input(audio)
        t0 = time.time()

        width: float = float(kwargs.get("width", _DEFAULT_WIDTH))
        diffuse: bool = bool(kwargs.get("diffuse", True))

        if audio.ndim == 1:
            # Mono: kein Stereo-Processing möglich
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio,
                execution_time_seconds=time.time() - t0,
                metadata={"skipped": "mono_input", "width": width},
                metrics={},
            )

        L = audio[:, 0].astype(np.float64)
        R = audio[:, 1].astype(np.float64)

        # M/S Enkodierung
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        M = (L + R) * inv_sqrt2
        S = (L - R) * inv_sqrt2

        # Breite skalieren
        S_wide = S * width

        # Optionale Allpass-Diffusion auf S (reduziert Kammfiltereffekte)
        if diffuse and width > 1.05:
            S_wide = self._allpass_chain(S_wide, sample_rate)

        # M/S Rück-Dekodierung
        L_out = (M + S_wide) * inv_sqrt2
        R_out = (M - S_wide) * inv_sqrt2
        processed = np.column_stack([L_out, R_out])

        # Normalisierung: Pegel-Erhalt
        peak_in = float(np.max(np.abs(audio)))
        peak_out = float(np.max(np.abs(processed)))
        if peak_out > 1e-8 and peak_in > 1e-8:
            processed = processed * (peak_in / peak_out)

        processed = np.clip(processed, -1.0, 1.0).astype(audio.dtype)

        logger.info("Phase 48 Stereo-Width: width=%.2f, diffuse=%s", width, diffuse)

        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.clip(processed, -1.0, 1.0)
        return PhaseResult(
            success=True,
            audio=processed,
            execution_time_seconds=time.time() - t0,
            metadata={"width": width, "diffuse": diffuse},
            metrics={"width": width},
        )

    def _allpass_chain(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Kette von Schroeder-Allpass-Filtern zur Diffusion.

        Jeder Allpass erhält eine leicht geänderte Verzögerung
        (Primzahl-Verhältnis nach Schroeder 1962), was Kammfiltereffekte
        im Breit-Stereo minimiert.
        """
        out = signal.copy()
        for delay_ms in _ALLPASS_DELAYS_MS:
            delay_s = max(1, int(delay_ms / 1000.0 * sample_rate))
            g = _ALLPASS_GAIN
            # Schroeder Allpass: H(z) = (-g + z^{-D}) / (1 - g * z^{-D})
            b = np.zeros(delay_s + 1)
            b[0] = -g
            b[-1] = 1.0
            a = np.zeros(delay_s + 1)
            a[0] = 1.0
            a[-1] = -g
            out = sig.lfilter(b, a, out)
        return out
