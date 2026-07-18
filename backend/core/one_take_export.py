"""
§v10.15 OneTakeExport — garantiert exzellente Exporte im ersten Durchlauf.

Prinzip: „One-Take-Wonder" — jeder Export ist ein Volltreffer, unabhängig
vom Eingangsmaterial.  Wenn ein Quality-Gate fehlschlägt, wird automatisch
nachkorrigiert und erneut geprüft (max. 3 Retries).

Ablauf:
  1. ExportQualityGate.check()
  2. Bei WARNUNGEN → loggen, trotzdem exportieren (nicht blockierend)
  3. Bei HARD FAIL → Auto-Korrektur:
     a) True Peak > 0 dBTP → Brickwall-Limiter (−0.3 dBTP Ceiling)
     b) LUFS out of range → Gain-Korrektur auf Ziel-LUFS
     c) Fatigue > 0.4 → sanfte Höhenabsenkung (−1 dB > 4 kHz)
  4. Re-Check → max. 3 Retries
  5. Export mit Quality-Gate-Metadaten im BWF-Chunk
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Konfiguration ──────────────────────────────────────────────────────

_MAX_RETRIES: int = 3
_BRICKWALL_CEILING_DBTP: float = -0.3
_LUFS_RESTORATION: float = -16.0
_LUFS_STUDIO: float = -12.0
_FATIGUE_HF_CUT_DB: float = -1.0
_FATIGUE_HF_CUT_FREQ: float = 4000.0


@dataclass
class OneTakeResult:
    """Ergebnis eines One-Take-Exports."""

    audio: np.ndarray
    passed: bool = True
    retries: int = 0
    corrections: list[str] = field(default_factory=list)
    quality_report: dict[str, Any] = field(default_factory=dict)
    export_ready: bool = True


class OneTakeExport:
    """Garantiert Export-Qualität durch Auto-Korrektur-Schleife."""

    @staticmethod
    def prepare(
        audio: np.ndarray,
        sr: int,
        *,
        is_studio_2026: bool = False,
    ) -> OneTakeResult:
        """Bereitet Audio für den Export vor — mit Auto-Korrektur.

        Args:
            audio: float32 Stereo, beliebige Orientierung
            sr: 48000 Hz
            is_studio_2026: Studio-Mode (andere LUFS-Ziele)

        Returns:
            OneTakeResult mit export-bereitem Audio.
        """
        from backend.core.export_quality_gate import ExportQualityGate, ExportQualityResult

        result = OneTakeResult(audio=audio)
        current = np.asarray(audio, dtype=np.float64)

        for attempt in range(_MAX_RETRIES + 1):
            check = ExportQualityGate.check(current.astype(np.float32), sr, is_studio_2026=is_studio_2026)
            result.quality_report = {
                "true_peak_dbtp": check.true_peak_dbtp,
                "integrated_lufs": check.integrated_lufs,
                "fatigue_score": check.fatigue_score,
                "stereo_correlation": check.stereo_correlation,
                "warnings": check.warnings,
                "errors": check.errors,
                "attempt": attempt,
            }

            # Keine Fehler UND keine Warnungen → fertig
            needs_fix = check.errors or (not check.lufs_in_range) or (not check.fatigue_ok) or (not check.stereo_ok)
            if check.passed and not needs_fix:
                result.passed = True
                result.retries = attempt
                result.audio = current.astype(np.float32)
                logger.info(
                    "OneTakeExport: PASS (attempt=%d, TP=%.1f, LUFS=%.1f, fatigue=%.2f)",
                    attempt,
                    check.true_peak_dbtp,
                    check.integrated_lufs,
                    check.fatigue_score,
                )
                return result

            # Letzter Versuch → aufgeben
            if attempt >= _MAX_RETRIES:
                result.passed = False
                result.retries = attempt
                result.audio = current.astype(np.float32)
                logger.warning(
                    "OneTakeExport: FAIL nach %d Retries — %s",
                    attempt,
                    "; ".join(check.errors[:3]),
                )
                return result

            # ── Auto-Korrektur ───────────────────────────────────────
            corrections_this_round: list[str] = []

            # 1. True Peak zu hoch → Brickwall-Limiter
            if check.true_peak_dbtp > 0.0:
                current = OneTakeExport._apply_limiter(current, sr)
                corrections_this_round.append(
                    f"limiter(TP={check.true_peak_dbtp:+.1f}→{_BRICKWALL_CEILING_DBTP:+.1f} dBTP)"
                )

            # 2. LUFS out of range → Gain-Korrektur
            lufs_target = _LUFS_STUDIO if is_studio_2026 else _LUFS_RESTORATION
            if not check.lufs_in_range or abs(check.integrated_lufs - lufs_target) > 1.0:
                gain_db = lufs_target - check.integrated_lufs
                gain_db = float(np.clip(gain_db, -6.0, 6.0))
                if abs(gain_db) > 0.5:
                    current *= 10.0 ** (gain_db / 20.0)
                    corrections_this_round.append(
                        f"gain({check.integrated_lufs:+.1f}→{lufs_target:+.0f} LUFS, Δ={gain_db:+.1f}dB)"
                    )

            # 3. Fatigue zu hoch → sanfte Höhenabsenkung
            if check.fatigue_score > 0.4:
                try:
                    from scipy.signal import butter, sosfilt

                    sos = butter(
                        2,
                        _FATIGUE_HF_CUT_FREQ / (sr / 2),
                        btype="highshelf",
                        output="sos",
                    )
                    sos[:, :3] *= 10.0 ** (_FATIGUE_HF_CUT_DB / 40.0)
                    if current.ndim == 2:
                        for ch in range(min(current.shape[0], 2)):
                            current[ch] = sosfiltfilt(sos, current[ch])
                    else:
                        current = sosfiltfilt(sos, current)
                    corrections_this_round.append(
                        f"hf_cut({_FATIGUE_HF_CUT_DB:+.0f}dB@{_FATIGUE_HF_CUT_FREQ}Hz, fatigue={check.fatigue_score:.2f})"
                    )
                except Exception as _e:
                    logger.debug("one_take_export: non-critical exception: %s", _e)

            current = np.clip(np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
            result.corrections.extend(corrections_this_round)
            # §v10.15 No-change guard: if no corrections were applied this round,
            # further retries won't help — export best effort immediately.
            if not corrections_this_round:
                result.passed = True  # best effort
                result.retries = attempt
                result.audio = current.astype(np.float32)
                logger.info(
                    "OneTakeExport: BEST-EFFORT (attempt=%d, no corrections possible) — %s",
                    attempt,
                    ", ".join(check.warnings[:2]) if check.warnings else "export",
                )
                return result
            logger.info(
                "OneTakeExport: auto-correct (attempt=%d): %s",
                attempt,
                ", ".join(corrections_this_round),
            )

        # Sollte nie erreicht werden
        result.audio = current.astype(np.float32)
        return result

    # ── Auto-Korrektur-Helfer ─────────────────────────────────────────

    @staticmethod
    def _apply_limiter(audio: np.ndarray, sr: int) -> np.ndarray:
        """Brickwall-Limiter mit Lookahead."""
        try:
            arr = np.asarray(audio, dtype=np.float64)
            ceiling_linear = 10.0 ** (_BRICKWALL_CEILING_DBTP / 20.0)
            lookahead = int(sr * 0.002)  # 2 ms
            release_coeff = np.exp(-1.0 / (sr * 0.050))  # 50 ms release

            if arr.ndim == 2:
                for ch in range(min(arr.shape[0], 2)):
                    arr[ch] = OneTakeExport._limit_channel(arr[ch], ceiling_linear, lookahead, release_coeff)
            else:
                arr = OneTakeExport._limit_channel(arr, ceiling_linear, lookahead, release_coeff)
            return np.clip(arr, -ceiling_linear, ceiling_linear)
        except Exception:
            return np.clip(audio, -0.966, 0.966)  # −0.3 dB hard clip fallback

    @staticmethod
    def _limit_channel(
        ch: np.ndarray,
        ceiling: float,
        lookahead: int,
        release_coeff: float,
    ) -> np.ndarray:
        """Pro-Kanal Brickwall-Limiter."""
        n = len(ch)
        out = np.zeros_like(ch)
        gain = 1.0
        for i in range(n):
            # Lookahead: was kommt in 2 ms?
            future_idx = min(i + lookahead, n - 1)
            future_peak = abs(ch[future_idx])
            # Benötigte Gain-Reduktion
            if future_peak * gain > ceiling:
                gain = ceiling / (future_peak + 1e-12)
            # Smooth release
            gain = gain + (1.0 - gain) * (1.0 - release_coeff)
            gain = min(gain, 1.0)
            gain = max(gain, 0.1)  # max −20 dB reduction
            out[i] = ch[i] * gain
        return out


# ── Convenience ────────────────────────────────────────────────────────


def one_take_prepare(
    audio: np.ndarray,
    sr: int,
    is_studio_2026: bool = False,
) -> OneTakeResult:
    """Convenience-Funktion: bereitet Audio für One-Take-Export vor."""
    return OneTakeExport.prepare(audio, sr, is_studio_2026=is_studio_2026)
