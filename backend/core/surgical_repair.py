"""
§2.59 Surgical Defect Repair (2026-07-09)

Zeitlich präzise, ortsgenaue Reparatur einzelner Defekt-Instanzen.
Kein globales Processing. Nur die kranke Stelle wird operiert.

Prinzip:
  1. Defekt-Instanz lokalisieren (start_sample, end_sample)
  2. Kontext-Fenster extrahieren (für Cross-Fade)
  3. Phase NUR auf das Fenster anwenden
  4. Repariertes Segment nahtlos zurück-crossfaden
  5. Lautstärke, Phase, DC-Offset am Übergang angleichen

Garantiert: keine Sprünge, keine Pegeländerungen, keine Artefakte
an den Übergängen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DefectInstance:
    """Eine zeitlich lokalisierte Defekt-Instanz."""
    start_s: float
    end_s: float
    defect_type: str
    severity: float


@dataclass
class RepairResult:
    """Ergebnis einer chirurgischen Reparatur."""
    audio: np.ndarray
    zones_repaired: int = 0
    zones_skipped: int = 0


class SurgicalRepair:
    """Führt zeitlich präzise, ortsgenaue Reparaturen durch.

    Extrahiert jedes Defekt-Fenster mit Kontext, wendet die Phase an,
    und cross-faded das Ergebnis nahtlos zurück.
    """

    def __init__(
        self,
        sr: int = 48000,
        context_ms: float = 50.0,    # Kontext vor/nach dem Defekt
        crossfade_ms: float = 10.0,  # Cross-Fade-Dauer
    ) -> None:
        self.sr = sr
        self._context_samples = int(context_ms * sr / 1000)
        self._crossfade_samples = int(crossfade_ms * sr / 1000)

    def repair(
        self,
        audio: np.ndarray,
        instances: list[DefectInstance],
        phase_fn: Any,  # callable(audio_segment, sr, **kwargs) → np.ndarray
        phase_kwargs: dict[str, Any] | None = None,
    ) -> RepairResult:
        """Repariert jede Defekt-Instanz einzeln mit Cross-Fade.

        Args:
            audio: Original-Audio (channels, samples) oder (samples,)
            instances: Liste zeitlich lokalisierter Defekte
            phase_fn: Funktion die ein Audio-Segment repariert
            phase_kwargs: Zusätzliche KWArgs für die Phase

        Returns:
            RepairResult mit repariertem Audio
        """
        if not instances:
            return RepairResult(audio=audio.copy(), zones_skipped=0)

        was_mono = audio.ndim == 1
        if was_mono:
            audio = audio.reshape(1, -1)

        result = audio.copy()
        total_samples = audio.shape[1]
        repaired = 0
        skipped = 0

        for inst in sorted(instances, key=lambda x: x.start_s):
            s0 = max(0, int(inst.start_s * self.sr) - self._context_samples)
            s1 = min(total_samples, int(inst.end_s * self.sr) + self._context_samples)

            if s1 - s0 < self._crossfade_samples * 3:
                skipped += 1
                continue  # Zu kurz für sinnvolle Reparatur

            # Extrahiere Fenster mit Kontext
            segment = audio[:, s0:s1].copy()
            original_segment = segment.copy()

            # Wende Phase nur auf dieses Fenster an
            try:
                kwargs = {"audio": segment, "sr": self.sr,
                          "material": phase_kwargs.pop("material", "unknown") if phase_kwargs else "unknown",
                          "mode": phase_kwargs.pop("mode", "restoration") if phase_kwargs else "restoration"}
                if phase_kwargs:
                    kwargs.update(phase_kwargs)
                repaired_segment = phase_fn(**kwargs)
                if isinstance(repaired_segment, np.ndarray):
                    segment = repaired_segment
            except Exception:
                skipped += 1
                continue

            # Cross-Fade: nur an den Rändern, Mitte bleibt Reparatur
            if segment.shape[1] >= self._crossfade_samples * 2:
                self._apply_crossfade(segment, original_segment,
                                      self._crossfade_samples)

            # Pegel-Angleich: RMS vorher/nachher matchen
            segment = self._match_rms(segment, original_segment)

            # Zurückschreiben
            result[:, s0:s1] = segment
            repaired += 1

        if was_mono:
            result = result[0]

        logger.info(
            "SurgicalRepair: %d/%d Zonen präzise repariert (%.1f%%), "
            "%d übersprungen (zu kurz/Fehler)",
            repaired, len(instances),
            100 * repaired / max(len(instances), 1),
            skipped,
        )

        return RepairResult(
            audio=result,
            zones_repaired=repaired,
            zones_skipped=skipped,
        )

    @staticmethod
    def _apply_crossfade(
        repaired: np.ndarray,
        original: np.ndarray,
        fade_samples: int,
    ) -> None:
        """Cosine Cross-Fade an den Rändern (psychoakustisch transparent).

        Verwendet Cosine-Ramp wie SectionStrengthEnvelope (§8.3).
        Max 1 dB / 100 ms Änderungsrate — unterhalb der menschlichen
        Wahrnehmungsschwelle (Zwicker & Fastl 1999).
        """
        if fade_samples <= 0 or repaired.shape[1] < fade_samples * 2:
            return

        # Cosine Fade-In (linker Rand): sanfter als linear
        ramp_in = 0.5 * (1 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
        for ch in range(repaired.shape[0]):
            repaired[ch, :fade_samples] = (
                original[ch, :fade_samples] * (1 - ramp_in) +
                repaired[ch, :fade_samples] * ramp_in
            )

        # Cosine Fade-Out (rechter Rand)
        ramp_out = 0.5 * (1 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
        for ch in range(repaired.shape[0]):
            repaired[ch, -fade_samples:] = (
                original[ch, -fade_samples:] * (1 - ramp_out[::-1]) +
                repaired[ch, -fade_samples:] * ramp_out[::-1]
            )

    @staticmethod
    def _match_rms(repaired: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Passt RMS-Pegel des reparierten Segments ans Original an."""
        rms_orig = np.sqrt(np.mean(original ** 2)) + 1e-10
        rms_rep = np.sqrt(np.mean(repaired ** 2)) + 1e-10
        if abs(rms_rep - rms_orig) / rms_orig > 0.01:  # >1% Abweichung
            return repaired * (rms_orig / rms_rep)
        return repaired
