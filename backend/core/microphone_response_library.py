"""
MicrophoneResponseLibrary — §6.4a [RELEASE_MUST]
=================================================

Historische Mikrofon-EQ-Profile für era-adaptive Signalverarbeitung.
Liefert EQ-Kurven für Phase_38 / Phase_06 um die Recording-Chain
des Originals zu modellieren.

Spec: 05_material_system.md §6.4a (v9.12.0)
"""

import json
import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_instance: "MicrophoneResponseLibrary | None" = None
_lock = threading.Lock()

_PROFILES_PATH = Path(__file__).parent.parent / "data" / "microphone_profiles.json"


def get_microphone_response_library() -> "MicrophoneResponseLibrary":
    """Singleton-Getter (thread-safe, double-checked locking)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MicrophoneResponseLibrary()
    return _instance


class MicrophoneResponseLibrary:
    """Lädt und liefert historische Mikrofon-EQ-Profile (§6.4a)."""

    def __init__(self) -> None:
        self._profiles: list[dict] = []
        self._load_profiles()

    def _load_profiles(self) -> None:
        try:
            with open(_PROFILES_PATH, encoding="utf-8") as f:
                data = json.load(f)
            self._profiles = data.get("profiles", [])
            logger.info("MicrophoneResponseLibrary loaded %d profiles", len(self._profiles))
        except Exception as exc:
            logger.warning("MicrophoneResponseLibrary: profiles not loaded (%s)", exc)
            self._profiles = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_profile(
        self,
        era_decade: int,
        genre_label: str,
        material_type: str,
    ) -> dict | None:
        """Gibt das am besten passende Mikrofon-Profil zurück.

        Scoring:
          +3 wenn era_decade im Profil-Bereich
          +2 wenn genre_label in genres
          +1 wenn material_type in materials

        Args:
            era_decade:   Aufnahme-Jahrzehnt als int (z.B. 1950 für 1950er).
            genre_label:  Genre (z.B. "jazz", "schlager", "rock").
            material_type: Material (z.B. "shellac", "vinyl", "reel_tape").

        Returns:
            Bestes Profil-Dict oder None wenn keine Profile geladen.
        """
        if not self._profiles:
            return None

        best_profile = None
        best_score = -1

        for profile in self._profiles:
            score = 0

            era_range = profile.get("era_decade", [])
            if era_range and len(era_range) >= 2:
                era_min = min(era_range)
                era_max = max(era_range)
                if era_min <= era_decade <= era_max + 10:
                    score += 3

            genres = [g.lower() for g in profile.get("genres", [])]
            if genre_label.lower() in genres:
                score += 2

            materials = [m.lower() for m in profile.get("materials", [])]
            if material_type.lower() in materials:
                score += 1

            if score > best_score:
                best_score = score
                best_profile = profile

        return best_profile

    def get_eq_curve(
        self,
        era_decade: int,
        genre_label: str,
        material_type: str,
        target_sr: int = 48000,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Gibt EQ-Kurve als (freqs_hz, gains_linear) Arrays zurück.

        Args:
            era_decade:   Aufnahme-Jahrzehnt.
            genre_label:  Genre.
            material_type: Material.
            target_sr:    Sample-Rate für Nyquist-Begrenzung.

        Returns:
            Tuple (freqs: np.ndarray, gains_linear: np.ndarray) oder None.
            Frequenzen sind aufsteigend, gains_linear >= 0.

        Notes:
            - max wet_mix = 0.35 (§6.4a Invariante — kein hartes EQ-Match)
            - Verwende np.interp für Interpolation auf Ziel-Frequenzachse
        """
        profile = self.get_profile(era_decade, genre_label, material_type)
        if profile is None:
            return None

        eq_curve = profile.get("eq_curve", [])
        if not eq_curve:
            return None

        nyq = target_sr / 2.0
        freqs = []
        db_values = []

        for point in eq_curve:
            hz = float(point["hz"])
            db = float(point["db"])
            if hz <= nyq:
                freqs.append(hz)
                db_values.append(db)

        if len(freqs) < 2:
            return None

        freqs_arr = np.array(freqs, dtype=np.float32)
        gains_db = np.array(db_values, dtype=np.float32)
        gains_linear = np.power(10.0, gains_db / 20.0).astype(np.float32)

        return freqs_arr, gains_linear

    def apply_eq_curve(
        self,
        audio: np.ndarray,
        sr: int,
        era_decade: int,
        genre_label: str,
        material_type: str,
        wet_mix: float = 0.20,
    ) -> np.ndarray:
        """Wendet die EQ-Kurve als frequency-domain Shaping an.

        Args:
            audio:        Float32 Audio.
            sr:           Sample-Rate.
            era_decade:   Aufnahme-Jahrzehnt.
            genre_label:  Genre.
            material_type: Material.
            wet_mix:      Blend-Faktor [0, 0.35] (§6.4a Hard-Cap = 0.35).

        Returns:
            Audio mit applizierter EQ-Charakteristik, selbe Form und Länge.
        """
        wet_mix = float(np.clip(wet_mix, 0.0, 0.35))  # §6.4a Hard-Cap

        eq_result = self.get_eq_curve(era_decade, genre_label, material_type, sr)
        if eq_result is None:
            return audio

        freqs_eq, gains_eq = eq_result

        try:
            original_shape = audio.shape
            mono = audio.mean(axis=0) if audio.ndim == 2 and audio.shape[0] == 2 else audio
            if mono.ndim == 2:
                mono = mono.mean(axis=1)

            n = len(mono)
            if n < 64:
                return audio

            # FFT-basiertes EQ via Interpolation auf FFT-Bins
            fft_freqs = np.fft.rfftfreq(n, d=1.0 / sr).astype(np.float32)
            gains_interp = np.interp(fft_freqs, freqs_eq, gains_eq, left=gains_eq[0], right=gains_eq[-1])

            def _apply_to_channel(ch: np.ndarray) -> np.ndarray:
                spectrum = np.fft.rfft(ch.astype(np.float64))
                spectrum_eq = spectrum * gains_interp.astype(np.float64)
                result = np.fft.irfft(spectrum_eq, n=n)
                return result.astype(np.float32)

            if audio.ndim == 1:
                eq_audio = _apply_to_channel(audio)
            elif audio.ndim == 2 and audio.shape[0] == 2:
                eq_audio = np.stack(
                    [
                        _apply_to_channel(audio[0]),
                        _apply_to_channel(audio[1]),
                    ]
                )
            elif audio.ndim == 2 and audio.shape[1] == 2:
                eq_audio = np.stack(
                    [
                        _apply_to_channel(audio[:, 0]),
                        _apply_to_channel(audio[:, 1]),
                    ],
                    axis=1,
                )
            else:
                eq_audio = _apply_to_channel(audio.flatten()).reshape(original_shape)

            # Wet/Dry-Mix
            blended = (1.0 - wet_mix) * audio + wet_mix * eq_audio
            blended = np.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0)
            blended = np.clip(blended, -1.0, 1.0)
            return blended.astype(np.float32)

        except Exception as exc:
            logger.warning("MicrophoneResponseLibrary.apply_eq_curve failed: %s", exc)
            return audio
