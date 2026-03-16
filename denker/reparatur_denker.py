"""
ReparaturDenker — Domäne: Gezielte DSP-Reparaturen.

Führt punktuelle Korrekturen durch, die nicht die gesamte Pipeline
benötigen: Click-Entfernung (Medianfilter), Netzbrumm-Unterdrückung
(Notch-Filter 50/60 Hz) und Clipping-Reparatur (Soft-Limiter +
kubische Spline-Interpolation).

Usage::

    from denker.reparatur_denker import get_reparatur_denker

    denker = get_reparatur_denker()
    ergebnis = denker.repariere(audio, sr=48000)
    sauberes_audio = ergebnis.audio
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import threading

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ergebnis-Datenstruktur
# ---------------------------------------------------------------------------


@dataclass
class ReparaturErgebnis:
    """Ergebnis einer gezielten DSP-Reparatur."""

    audio: np.ndarray
    """Repariertes Audio (float32, Bereich [-1, 1])."""

    clicks_removed: int
    """Anzahl entfernter Impulse/Clicks."""

    hum_removed: bool
    """Ob Netzbrumm erkannt und entfernt wurde."""

    clipping_repaired: bool
    """Ob Clipping-Regionen repariert wurden."""

    clipping_regions: int
    """Anzahl reparierter Clipping-Regionen."""

    processing_note: str
    """Zusammenfassung der Reparaturen (Deutsch)."""

    warnings: list[str] = field(default_factory=list)
    """Warnungen bei der Verarbeitung."""

    # --- Kompatibilitäts-Felder ---
    repairs_applied: list[str] = field(default_factory=list)
    """Angewendete Reparatur-Phasen (Compat-Alias)."""

    quality_delta: float = 0.0
    """Qualitätsverbesserung (Compat-Alias, wird nicht aktiv berechnet)."""

    material: str = ""
    """Trägermedium-Hinweis (Compat-Alias)."""

    reasoning: str = ""
    """Begründung der Reparaturen (Compat-Alias für processing_note)."""

    def as_dict(self) -> dict[str, object]:
        """Liefert alle Felder als serialisierbares Dict."""
        return {
            "clicks_removed": self.clicks_removed,
            "hum_removed": self.hum_removed,
            "clipping_repaired": self.clipping_repaired,
            "clipping_regions": self.clipping_regions,
            "processing_note": self.processing_note,
            "warnings": self.warnings,
            "audio_shape": list(self.audio.shape),
        }


# ---------------------------------------------------------------------------
# ReparaturDenker
# ---------------------------------------------------------------------------


class ReparaturDenker:
    """Reparatur-Domänendenker — gezielte DSP-Korrekturen.

    Reparatur-Kaskade (Reihenfolge einhalten)
    -----------------------------------------
    1. Click-Entfernung via Medianfilter + Interquartilsabstandserkennung.
    2. Netzbrumm-Unterdrückung via Zweistufiger Butterworth-Notch (50 + 60 Hz).
    3. Clipping-Reparatur via monotoner kubischer Spline-Interpolation.

    Invarianten
    -----------
    - Eingabe: NaN/Inf → ``nan_to_num`` vor jeder Operation.
    - Ausgabe: immer ``np.clip(..., -1.0, 1.0)``.
    - Singleton via :func:`get_reparatur_denker` (Double-Checked Locking).
    """

    # Schwellwerte
    _CLICK_IQR_MULTIPLIER: float = 6.0  # IQR-Multiplikator für Click-Detektion
    _CLICK_KERNEL_MS: float = 1.5  # Medianfilter-Halbfenster in ms
    _CLIP_THRESHOLD: float = 0.995  # Amplitude ≥ threshold gilt als Clipping
    _HUM_DETECT_DB: float = -50.0  # Energieschwelle für Brumm-Erkennung (dBFS)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def repariere(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        remove_clicks: bool = True,
        remove_hum: bool = True,
        repair_clipping: bool = True,
        validate_audio: bool = True,
        material: str = "",
        quality_before: float = 0.0,
    ) -> ReparaturErgebnis:
        """Repariert die häufigsten analogen Defekte per DSP.

        Parameter
        ---------
        audio:
            Eingabe-Audio (float32, mono oder stereo).
        sr:
            Abtastrate in Hz.
        remove_clicks:
            Ob Clicks/Impulse entfernt werden sollen.
        remove_hum:
            Ob Netzbrumm (50/60 Hz) unterdrückt werden soll.
        repair_clipping:
            Ob Clipping-Regionen interpoliert werden sollen.

        Rückgabe
        --------
        :class:`ReparaturErgebnis` mit repariertem Audio und Statistik.
        """
        assert sr == 48000, f"ReparaturDenker.repariere() erwartet sr=48000 Hz, erhalten: {sr} Hz"
        if validate_audio:
            audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            audio = audio.astype(np.float32)

        clicks_removed = 0
        hum_removed = False
        clipping_repaired = False
        clipping_regions = 0
        notes: list[str] = []
        warnings: list[str] = []

        # --- 1. Click-Entfernung ---
        if remove_clicks:
            try:
                audio, clicks_removed = self._remove_clicks(audio, sr)
                if clicks_removed > 0:
                    notes.append(f"{clicks_removed} Klicken entfernt")
            except Exception as exc:
                logger.debug("Click-Entfernung fehlgeschlagen: %s", exc)
                warnings.append(f"Click-Entfernung übersprungen: {exc}")

        # --- 2. Netzbrumm ---
        if remove_hum:
            try:
                audio, hum_removed = self._remove_hum(audio, sr)
                if hum_removed:
                    notes.append("Netzbrumm (50/60 Hz) unterdrückt")
            except Exception as exc:
                logger.debug("Hum-Entfernung fehlgeschlagen: %s", exc)
                warnings.append(f"Hum-Entfernung übersprungen: {exc}")

        # --- 3. Clipping-Reparatur ---
        if repair_clipping:
            try:
                audio, clipping_repaired, clipping_regions = self._repair_clipping(audio)
                if clipping_repaired:
                    notes.append(f"{clipping_regions} Clipping-Regionen interpoliert")
            except Exception as exc:
                logger.debug("Clipping-Reparatur fehlgeschlagen: %s", exc)
                warnings.append(f"Clipping-Reparatur übersprungen: {exc}")

        # Ausgabe sichern
        audio = np.clip(
            np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0),
            -1.0,
            1.0,
        )

        note = " | ".join(notes) if notes else "Keine Reparaturen nötig"

        logger.info(
            "🔧 ReparaturDenker: clicks=%d, hum=%s, clipping=%s (%d Regionen)",
            clicks_removed,
            hum_removed,
            clipping_repaired,
            clipping_regions,
        )

        # Compat-Felder: Phasen-Liste ableiten
        _repairs_applied: list[str] = []
        if clicks_removed > 0:
            _repairs_applied.append("phase_01_click_removal")
        if hum_removed:
            _repairs_applied.append("phase_02_hum_removal")
        if clipping_repaired:
            _repairs_applied.append("phase_23_clipping_repair")

        return ReparaturErgebnis(
            audio=audio,
            clicks_removed=clicks_removed,
            hum_removed=hum_removed,
            clipping_repaired=clipping_repaired,
            clipping_regions=clipping_regions,
            processing_note=note,
            warnings=warnings,
            repairs_applied=_repairs_applied,
            quality_delta=0.0,
            material=material,
            reasoning=note,
        )

    # ------------------------------------------------------------------
    # 1. Click-Entfernung
    # ------------------------------------------------------------------

    def _remove_clicks(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """Medianfilter-basierte Click-Entfernung.

        Algorithmus:
            1. Mono-Referenz bilden (Mittelwert aller Kanäle).
            2. Differenz: ``d = audio − median_filtered(audio, kernel)``.
            3. IQR-basierte Schwelle: ``threshold = IQR(d) × _CLICK_IQR_MULTIPLIER``.
            4. Clicks-Maske: ``|d| > threshold``.
            5. Clicks durch Medianfilter-Version ersetzen.

        Rückgabe: (bereinigtes Audio, Anzahl erkannter Clicks)
        """
        from scipy.signal import medfilt

        kernel_samples = max(3, int(self._CLICK_KERNEL_MS * sr / 1000))
        # kernel muss ungerade sein
        if kernel_samples % 2 == 0:
            kernel_samples += 1

        # Mono-Referenz für Detektion
        mono = audio.mean(axis=0) if audio.ndim > 1 else audio

        smoothed = medfilt(mono.astype(np.float64), kernel_size=kernel_samples)
        diff = mono.astype(np.float64) - smoothed

        q75, q25 = np.percentile(diff, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-9:
            return audio, 0  # Signal zu homogen, kein Click erkennbar

        threshold = iqr * self._CLICK_IQR_MULTIPLIER
        mask = np.abs(diff) > threshold

        # Clicks zählen (verbundene Regionen)
        clicks = int(np.sum(np.diff(mask.astype(np.int8)) > 0))

        if clicks == 0:
            return audio, 0

        # Clicks ersetzen
        result = audio.copy()
        if audio.ndim > 1:
            for ch in range(audio.shape[0]):
                ch_smoothed = medfilt(audio[ch].astype(np.float64), kernel_size=kernel_samples)
                result[ch][mask] = ch_smoothed[mask]
        else:
            result[mask] = smoothed[mask]

        return result.astype(np.float32), clicks

    # ------------------------------------------------------------------
    # 2. Netzbrumm-Unterdrückung
    # ------------------------------------------------------------------

    def _remove_hum(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, bool]:
        """Zweistufiger IIR-Notch-Filter (50 Hz + 60 Hz + erste Obertöne).

        Nur aktiv wenn Brumm-Energie über ``_HUM_DETECT_DB`` liegt.
        """
        from scipy.signal import iirnotch, sosfilt, tf2sos

        # Brumm-Energie prüfen (50 Hz ± 5 Hz via FFT)
        mono = audio.mean(axis=0) if audio.ndim > 1 else audio
        N = len(mono)
        if N < 512:
            return audio, False

        fft = np.fft.rfft(mono)
        freqs = np.fft.rfftfreq(N, d=1.0 / sr)

        def _band_energy(f_center: float, bw: float = 5.0) -> float:
            mask = (freqs >= f_center - bw) & (freqs <= f_center + bw)
            e = float(np.mean(np.abs(fft[mask]) ** 2)) if mask.any() else 0.0
            return 10 * math.log10(e + 1e-12)

        detected = False
        result = audio.copy().astype(np.float64)

        # Filterfrequenzen (Grundton + 2. Oberton)
        for base_hz in (50.0, 60.0):
            for harmonic in (1, 2):
                f_hz = base_hz * harmonic
                if f_hz >= sr / 2:
                    continue
                energy_db = _band_energy(f_hz)
                if energy_db >= self._HUM_DETECT_DB:
                    detected = True
                    Q = 35.0  # Gütefaktor — schmaler Notch
                    b, a = iirnotch(f_hz, Q, sr)
                    try:
                        sos = tf2sos(b, a)
                    except Exception:
                        continue
                    if result.ndim > 1:
                        for ch in range(result.shape[0]):
                            result[ch] = sosfilt(sos, result[ch])
                    else:
                        result = sosfilt(sos, result)

        return result.astype(np.float32), detected

    # ------------------------------------------------------------------
    # 3. Clipping-Reparatur
    # ------------------------------------------------------------------

    def _repair_clipping(self, audio: np.ndarray) -> tuple[np.ndarray, bool, int]:
        """Kubische Spline-Interpolation in Clipping-Regionen.

        Algorithmus:
            1. Clipping-Maske: ``|sample| ≥ _CLIP_THRESHOLD``.
            2. Verbundene Clipping-Regionen identifizieren.
            3. Jede Region per ``np.interp`` (linear, dann monoton kubisch)
               aus den Randpunkten interpolieren.

        Rückgabe: (repariertes Audio, ob repariert, Anzahl Regionen)
        """
        threshold = self._CLIP_THRESHOLD
        result = audio.copy()
        total_regions = 0

        def _repair_channel(ch_data: np.ndarray) -> np.ndarray:
            nonlocal total_regions
            clipped = np.abs(ch_data) >= threshold
            if not clipped.any():
                return ch_data

            repaired = ch_data.copy()
            n = len(ch_data)
            indices = np.arange(n)  # noqa: F841

            # Verbundene Regionen
            in_region = False
            region_start = 0
            regions = []
            for i in range(n):
                if clipped[i] and not in_region:
                    in_region = True
                    region_start = i
                elif not clipped[i] and in_region:
                    in_region = False
                    regions.append((region_start, i))
            if in_region:
                regions.append((region_start, n))

            total_regions += len(regions)

            for start, end in regions:
                # Randpunkte (min. 1 Sample Puffer)
                left = max(0, start - 1)
                right = min(n - 1, end)
                if left == right:
                    continue
                x_known = [left, right]
                y_known = [ch_data[left], ch_data[right]]
                x_fill = np.arange(start, end)
                repaired[start:end] = np.interp(x_fill, x_known, y_known)

            return repaired

        if result.ndim > 1:
            for ch in range(result.shape[0]):
                result[ch] = _repair_channel(result[ch])
        else:
            result = _repair_channel(result)

        repaired = total_regions > 0
        return result.astype(np.float32), repaired, total_regions


# ---------------------------------------------------------------------------
# Thread-sicherer Singleton (Double-Checked Locking — §3.2)
# ---------------------------------------------------------------------------

_instance: ReparaturDenker | None = None
_lock: threading.Lock = threading.Lock()


def get_reparatur_denker() -> ReparaturDenker:
    """Gibt den thread-sicheren Singleton-ReparaturDenker zurück."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ReparaturDenker()
    return _instance
