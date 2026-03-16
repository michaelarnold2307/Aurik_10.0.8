"""
TontraegerDenker — Domäne: Tonträger-/Material-Erkennung
=========================================================

Kapselt `forensics.medium_detector.MediumDetector` und liefert
einen strukturierten `TontraegerInfo`-Bericht über das erkannte
Quellmaterial (§6.1 / §2.14).

Singleton-Pattern nach §3.2 (Double-Checked Locking).
NaN/Inf-Schutz nach §3.1.
Type-Annotations nach §3.7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import threading
from typing import Any, List

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class TontraegerInfo:
    """Strukturierter Bericht über das erkannte Quellmaterial."""

    material_type: str
    """Erkannter MaterialType-Name (z. B. 'vinyl', 'tape', 'unknown')."""

    confidence: float
    """Erkennungs-Konfidenz ∈ [0, 1]."""

    medium_details: dict[str, Any] = field(default_factory=dict)
    """Rohe Zusatz-Informationen aus MediumDetector (z. B. rolloff_hz, snr_db)."""

    chain_detected: bool = False
    """True, wenn eine Mehrstufen-Tonträgerkette erkannt wurde."""

    transfer_chain: list = field(default_factory=list)
    """Liste der erkannten Trägerstufen (z. B. ['tape', 'mp3_low'])."""

    reasoning: str = ""
    """Menschenlesbare Begründung der Erkennung (Deutsch)."""

    def as_dict(self) -> dict[str, Any]:
        """Serialisierungsformat für Logging und Persistenz."""
        return {
            "material_type": self.material_type,
            "confidence": self.confidence,
            "chain_detected": self.chain_detected,
            "transfer_chain": self.transfer_chain,
            "reasoning": self.reasoning,
            **self.medium_details,
        }


@dataclass
class TontraegerErgebnis:
    """Ergebnis der Tonträger-Erkennung (kompaktes Reporting)."""

    material_type: str
    """Erkannter Träger-Typ (z. B. 'vinyl', 'tape', 'shellac')."""

    confidence: float
    """Erkennungs-Konfidenz ∈ [0, 1]."""

    detected_media: list
    """Liste von (typ: str, konfidenz: float)-Tupeln aller Kandidaten."""

    reasoning: str
    """Laienverständliche Begründung der Träger-Erkennung."""

    recommended_phases: list
    """Empfohlene Verarbeitungsphasen für diesen Träger."""


# ---------------------------------------------------------------------------
# TontraegerDenker
# ---------------------------------------------------------------------------


class TontraegerDenker:
    """Erkennt das Quellmaterial einer Aufnahme (Tonträger-Domäne).

    Wraps `forensics.medium_detector.MediumDetector` und ergänzt
    strukturiertes Reporting, NaN-Schutz und laienverständliche
    Begründungstexte.

    Verwendung::

        denker = get_tontraeger_denker()
        info = denker.erkenne(audio, sr=44100)
        logger.debug(info.material_type, info.confidence)
    """

    def __init__(self) -> None:
        self._detector: Any | None = None
        self._detector_lock = threading.Lock()
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy-Init des MediumDetector
    # ------------------------------------------------------------------

    def _get_detector(self) -> Any:
        """Return the MediumDetector instance, initializing it lazily."""
        if not self._loaded:
            with self._detector_lock:
                if not self._loaded:
                    try:
                        from backend.core.forensics.medium_detector import MediumDetector

                        self._detector = MediumDetector()
                        logger.info("TontraegerDenker: MediumDetector geladen.")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "TontraegerDenker: MediumDetector nicht verfügbar (%s). " "Fallback auf 'unknown'.",
                            exc,
                        )
                        self._detector = None
                    self._loaded = True
        return self._detector

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def erkenne(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        validate_audio: bool = True,
    ) -> TontraegerInfo:
        """Erkennt das Quellmaterial und liefert eine strukturierten TontraegerInfo.

        Algorithmus:
            1. NaN/Inf-Schutz auf Eingabe
            2. MediumDetector.detect(audio, sr) → Dict
            3. Strukturierung + Konfidenz-Normalisierung
            4. Laienverständliche Begründung erzeugen

        Args:
            audio: Eingabe-Audio, float32/64 mono oder stereo.
            sr:    Sample-Rate in Hz.
            validate_audio: NaN/Inf-Bereinigung durchführen (Standard: True).

        Returns:
            TontraegerInfo mit material_type, confidence und Details.

        Raises:
            ValueError: Falls sr < 1000 (plausibilitätsprüfung).
        """
        if sr < 1000:
            raise ValueError(f"Sample-Rate ungültig: {sr} Hz (mindestens 1000 Hz erwartet).")

        assert sr == 48000, f"TontraegerDenker.erkenne() erwartet sr=48000 Hz, erhalten: {sr} Hz"
        if validate_audio:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        detector = self._get_detector()

        if detector is None:
            return self._fallback_result()

        try:
            raw: dict[str, Any] = detector.detect(audio, sr)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TontraegerDenker: detect() fehlgeschlagen (%s). Fallback.", exc)
            return self._fallback_result()

        return self._struktur(raw)

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _struktur(self, raw: dict[str, Any]) -> TontraegerInfo:
        """Strukturiert das Roh-Dict aus MediumDetector zu TontraegerInfo."""
        # Extrahiere bekannte Felder defensiv
        material = str(raw.get("material_type", raw.get("medium", "unknown"))).lower()
        confidence = float(raw.get("confidence", 0.5))

        # Konfidenz-Guard: muss finite sein
        if not math.isfinite(confidence):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        detected_media_raw: List[tuple] = raw.get("detected_media", [])
        chain = [m for m, _ in detected_media_raw] if detected_media_raw else []
        chain_detected = len(chain) > 1

        reasoning = self._begründung(material, confidence, chain_detected)

        # Rohe Details ohne bekannte Felder
        details = {k: v for k, v in raw.items() if k not in {"material_type", "medium", "confidence", "detected_media"}}

        logger.info(
            "TontraegerDenker: material=%s confidence=%.2f chain=%s",
            material,
            confidence,
            chain,
        )

        return TontraegerInfo(
            material_type=material,
            confidence=confidence,
            medium_details=details,
            chain_detected=chain_detected,
            transfer_chain=list(chain) if isinstance(chain, list) else [],
            reasoning=reasoning,
        )

    @staticmethod
    def _begründung(material: str, confidence: float, chain: bool) -> str:
        """Erzeugt eine laienverständliche Begründung (Deutsch)."""
        label_map = {
            "vinyl": "Schallplatte (Vinyl)",
            "tape": "Magnetband (Kassette/Tonband)",
            "reel_tape": "Profi-Spulenband",
            "shellac": "Schellack-Platte (78 rpm)",
            "wax_cylinder": "Phonograph-Wachswalze",
            "wire_recording": "Drahtbandaufnahme",
            "lacquer_disc": "Acetat-Lackfolie",
            "dat": "DAT (Digital Audio Tape)",
            "cd_digital": "CD / digitale Aufnahme",
            "mp3_low": "Stark komprimierte MP3-Datei",
            "mp3_high": "MP3-Datei",
            "aac": "AAC / M4A-Datei",
            "minidisc": "MiniDisc",
            "streaming": "Streaming-Kopie",
            "unknown": "Unbekanntes Material",
        }
        label = label_map.get(material, material)
        pct = int(confidence * 100)
        base = f"Erkanntes Material: {label} ({pct} % Konfidenz)."
        if chain:
            base += " Es wurde eine mehrstufige Übertragungskette entdeckt."
        return base

    @staticmethod
    def _fallback_result() -> TontraegerInfo:
        """Sicheres Fallback-Ergebnis wenn MediumDetector nicht verfügbar."""
        return TontraegerInfo(
            material_type="unknown",
            confidence=0.0,
            reasoning="Material-Erkennung nicht verfügbar — konservative Voreinstellung wird genutzt.",
        )


# ---------------------------------------------------------------------------
# Singleton-Accessor (§3.2 — Double-Checked Locking)
# ---------------------------------------------------------------------------

_instance: TontraegerDenker | None = None
_lock = threading.Lock()


def get_tontraeger_denker() -> TontraegerDenker:
    """Thread-sicherer Singleton-Accessor für TontraegerDenker."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = TontraegerDenker()
    return _instance
