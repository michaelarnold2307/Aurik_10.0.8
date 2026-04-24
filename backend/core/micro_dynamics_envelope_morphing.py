"""
backend/core/micro_dynamics_envelope_morphing.py
Aurik 9 -- Spec §2.30: MicroDynamicsEnvelopeMorphing (MDEM)

Stellt origales Mikro-Dynamik-Profil im restaurierten Signal wieder her.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


def _lufs_frame(frame: np.ndarray) -> float:
    """ITU-R BS.1770: momentane LUFS eines Audio-Frames (vereinfacht)."""
    if frame.ndim == 2:
        frame = frame.mean(axis=0)
    rms = math.sqrt(max(1e-15, float(np.mean(frame.astype(np.float64) ** 2))))
    lufs = 20.0 * math.log10(rms + 1e-15)
    return lufs


def _savgol_smooth(arr: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """Vereinfachte Savitzky-Golay Glaettung (boxcar wenn scipy nicht verfuegbar)."""
    try:
        from scipy.signal import savgol_filter

        return savgol_filter(arr, window_length=window, polyorder=polyorder).astype(np.float32)
    except Exception:
        # Boxcar-Fallback
        half = window // 2
        out = arr.copy()
        for i in range(len(arr)):
            lo = max(0, i - half)
            hi = min(len(arr), i + half + 1)
            out[i] = np.mean(arr[lo:hi])
        return out


@dataclass
class MorphResult:
    """Ergebnis des Envelope-Morphing."""

    pearson_correlation: float
    max_gain_applied_lu: float
    retried: bool
    audio: np.ndarray


class MicroDynamicsEnvelopeMorphing:
    """Spec §2.30: Mikro-Dynamik-Profil aus Original im Restaurierten wiederherstellen.

    Algorithmus:
        1. 400-ms-LUFS-Profile beider Signale (hop 200 ms, 50 % Ueberlappung)
        2. G[k] = L_orig[k] - L_rest[k], geclippt auf ±MAX_GAIN_LU (mode-adaptive)
        3. Savitzky-Golay-Glaettung
        4. Frame-weise lineare Gain-Interpolation
        5. True-Peak 1 dBTP nach Morphing kontrollen
    """

    MAX_GAIN_LU: float = 6.0  # Studio 2026 mode; Restoration nutzt 4.0 (siehe morph())
    FRAME_SIZE_SAMPLES: int = 19200  # 400 ms @ 48000 Hz
    HOP_SIZE_SAMPLES: int = 9600  # 200 ms
    PEARSON_TARGET: float = 0.93
    MIN_LEVEL_LUFS: float = -60.0
    TRUE_PEAK_LIMIT: float = 0.98  # ~-1.0 dBTP linear

    def compute_lufs_profile(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        """400-ms-momentane LUFS-Kurve, float32 [n_frames]."""
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        if arr.ndim == 2:
            arr = arr.mean(axis=0)

        n = len(arr)
        hop = self.HOP_SIZE_SAMPLES
        fsize = self.FRAME_SIZE_SAMPLES
        frames = max(1, (n - fsize) // hop + 1)
        profile = np.zeros(frames, dtype=np.float32)
        for i in range(frames):
            start = i * hop
            end = start + fsize
            frame = arr[start : min(end, n)]
            profile[i] = _lufs_frame(frame)
        return profile

    def morph(
        self,
        restored: np.ndarray,
        original: np.ndarray,
        sr: int = 48000,
        mode: str = "restoration",
        phoneme_timeline=None,
    ) -> np.ndarray:
        """Morphed restauriertes Signal auf Original-Mikrodynamik. NaN/Inf-sicher.

        §2.30 v9.10.79 Mode-basierte MAX_GAIN_LU-Kalibrierung:
        - Restoration: 4.0 dB (bewahrt emotionale Spitzen, aber konservativ für degradierte Quellen)
        - Studio 2026: 6.0 dB (maximale Dynamik-Restauration für hochwertige Eingaben)

        §2.36a phoneme_timeline: stressed vowel frames erhalten +0.5 dB extra max_gain
        (bis zum Klassen-Maximum MAX_GAIN_LU), um Vokal-Intensität zu bewahren.
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        max_gain = 4.0 if mode == "restoration" else 6.0  # §2.30 v9.10.79 Psychoakustische Kalibrierung

        # §2.36a: Vorberechnung stressed-vowel frame mask (frame index → extra gain headroom)
        _stressed_frames: set[int] = set()
        if phoneme_timeline is not None:
            try:
                _sv_segs = phoneme_timeline.stressed_vowel_segments()
                for _sv_seg in _sv_segs:
                    _f_start = int(_sv_seg.start_s * sr / self.HOP_SIZE_SAMPLES)
                    _f_end = int(_sv_seg.end_s * sr / self.HOP_SIZE_SAMPLES) + 1
                    for _fi in range(max(0, _f_start), _f_end):
                        _stressed_frames.add(_fi)
            except Exception as _sv_exc:
                logger.debug("MDEM stressed_vowel_segments failed: %s", _sv_exc)

        res = np.nan_to_num(np.asarray(restored, dtype=np.float32))
        orig = np.nan_to_num(np.asarray(original, dtype=np.float32))

        is_stereo = res.ndim == 2
        if is_stereo:
            res_mono = res.mean(axis=0)
            orig_mono = orig.mean(axis=0) if orig.ndim == 2 else orig
        else:
            res_mono = res
            orig_mono = orig if orig.ndim == 1 else orig.mean(axis=0)

        L_orig = self.compute_lufs_profile(orig_mono, sr)
        L_rest = self.compute_lufs_profile(res_mono, sr)

        # Gain-Profil berechnen
        n_frames = min(len(L_orig), len(L_rest))
        G = np.zeros(n_frames, dtype=np.float32)
        for k in range(n_frames):
            lo = L_orig[k]
            lr = L_rest[k]
            # §2.36a: +0.5 dB headroom for stressed-vowel frames (capped at MAX_GAIN_LU)
            _frame_max = min(max_gain + 0.5, self.MAX_GAIN_LU) if k in _stressed_frames else max_gain
            if not (math.isfinite(lo) and math.isfinite(lr)):
                G[k] = 0.0
                continue
            if lo < self.MIN_LEVEL_LUFS:
                G[k] = 0.0  # Original silent: no adjustment
                continue
            if lr < self.MIN_LEVEL_LUFS:
                # Restored is near-silent (denoising removed noise floor).
                # Never apply positive boost here — that would re-amplify the
                # cleaned noise floor, inflate integrated LUFS and force
                # phase_40 to attenuate the musical content (regression).
                G[k] = float(np.clip(lo - lr, -_frame_max, 0.0))
                continue
            # §2.30 Bug-Fix: Denoised-Fade-Out-Guard — if restored frame is in the
            # quiet/fade-out zone (< -36 dBFS), any positive gain would boost the
            # cleaned noise floor. Original at -40 dBFS was carrier noise, not music.
            # Applying positive gain here would cause Pegelexplosion in the fade-out.
            # -36 dBFS covers vinyl fade-out noise floor range (-40 to -46 dBFS restored).
            _FADEOUT_QUIET_LUFS = -36.0  # dBFS threshold for fade-out detection
            if lr < _FADEOUT_QUIET_LUFS:
                # Restored is in the quiet/noise-floor zone → no positive boost allowed
                G[k] = float(np.clip(lo - lr, -_frame_max, 0.0))
                continue
            G[k] = np.clip(lo - lr, -_frame_max, _frame_max)

        # Glaettung
        G_smooth = _savgol_smooth(G)

        # §2.30 Post-Smoothing Quiet-Zone-Clamp: SG-Glaettung kann den Boost aus
        # Musik-Frames in angrenzende Fadeout-Frames verschleppen (window=7 → 1.4 s).
        # Nach der Glaettung alle Frames nochmals auf kein-positiver-Boost prüfen,
        # wenn das restaurierte Signal in der Quiet-Zone (< -36 dBFS) liegt.
        for k in range(n_frames):
            if L_rest[k] < -36.0 and G_smooth[k] > 0.0:
                G_smooth[k] = 0.0

        # Gain-Anwendung: frame-weise lineare Interpolation
        hop = self.HOP_SIZE_SAMPLES
        fsize = self.FRAME_SIZE_SAMPLES
        n = len(res_mono)

        gain_envelope = np.ones(n, dtype=np.float32)
        for k in range(n_frames):
            start = k * hop
            end = start + fsize
            linear_gain = 10.0 ** (G_smooth[k] / 20.0)
            ce = min(end, n)
            if start < n:
                # Lineare Interpolation zu naechstem Frame
                nxt_gain = 10.0 ** (G_smooth[k + 1] / 20.0) if k + 1 < n_frames else linear_gain
                ramp = np.linspace(linear_gain, nxt_gain, ce - start, dtype=np.float32)
                gain_envelope[start:ce] = ramp

        # §2.30 tail-gap fix (§9.10.119 improved): Samples nach dem letzten
        # Frame-Ende erhalten eine sanfte Rückkehr statt eines harten Wert-Kopierens.
        # §9.11.2: Tail-Gain darf in stillen Regionen NICHT auf 1.0 (Unity) hoch-
        # interpolieren — das hebt den Rauschboden an und erzeugt hörbaren Pegelsprung
        # am Musikende.  Stattdessen: Wenn das Tail-Audio leise ist (Stille/Rauschboden),
        # bleibt der letzte Gain erhalten oder wird sanft auf _last_gain gehalten.
        _last_covered = min((n_frames - 1) * hop + fsize, n)
        if _last_covered < n and _last_covered > 0:
            _tail_len = n - _last_covered
            _last_gain = gain_envelope[_last_covered - 1]
            # Check if tail audio is silence/noise floor (RMS < -50 dBFS ≈ 0.003)
            _tail_audio = res_mono[_last_covered:] if _last_covered < len(res_mono) else np.zeros(1)
            _tail_rms = float(np.sqrt(np.mean(_tail_audio**2) + 1e-12))
            _tail_rms_dbfs = 20.0 * math.log10(_tail_rms + 1e-12)
            # §2.30 Tail-Guard: covers both digital silence (<-50 dBFS) and
            # vinyl/tape fade-out noise floor (<-36 dBFS). In both cases do NOT
            # boost — clamp gain to min(last_gain, 1.0) to avoid Pegelexplosion.
            _tail_in_quiet_zone = _tail_rms_dbfs < -36.0  # vinyl noise floor threshold
            if _tail_in_quiet_zone:
                # Tail is silence or noise floor: cap gain at unity (no boost)
                _safe_gain = min(_last_gain, 1.0)
                gain_envelope[_last_covered:] = _safe_gain
            else:
                # Tail has musical content: smooth interpolation to unity,
                # but never exceed 1.0 when last_gain > 1.0 (avoids fade-out boost)
                _interp = np.linspace(_last_gain, 1.0, _tail_len, dtype=np.float32)
                if _last_gain > 1.0:
                    _interp = np.minimum(_interp, 1.0)
                gain_envelope[_last_covered:] = _interp

        # Auf Stereo/Mono anwenden
        if is_stereo:
            out = res * gain_envelope[np.newaxis, : res.shape[1]] if res.shape[0] == 2 else res
            # Sicherere Anwendung
            if res.ndim == 2:
                n_ch = res.shape[0]
                out = np.zeros_like(res)
                for ch in range(n_ch):
                    n_samp = min(len(res[ch]), len(gain_envelope))
                    out[ch, :n_samp] = res[ch, :n_samp] * gain_envelope[:n_samp]
                    if n_samp < res.shape[1]:
                        out[ch, n_samp:] = res[ch, n_samp:]
        else:
            n_samp = min(n, len(gain_envelope))
            out = res.copy()
            out[:n_samp] = res[:n_samp] * gain_envelope[:n_samp]

        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        out = np.clip(out, -self.TRUE_PEAK_LIMIT, self.TRUE_PEAK_LIMIT)

        # Pearson-Korrelation pruefen, ggf. Retry
        out_mono = out.mean(axis=0) if out.ndim == 2 else out
        r = self._pearson(orig_mono[: len(out_mono)], out_mono[: len(orig_mono)])

        if r < self.PEARSON_TARGET and max_gain < self.MAX_GAIN_LU:
            _retry_gain = min(max(max_gain * 1.5, 4.0), self.MAX_GAIN_LU)  # §2.54: mindestens 4.0, max MAX_GAIN_LU
            logger.debug("MDEM Retry mit erweitertem MAX_GAIN=%.1f dB (aktuell r=%.3f)", _retry_gain, r)
            # Einmaliger Retry mit erweitertem Gain — kein weiterer rekursiver Aufruf
            out2 = self._morph_internal(res_mono, orig_mono, max_gain=_retry_gain)
            out2 = np.nan_to_num(out2, nan=0.0, posinf=1.0, neginf=-1.0)
            out2 = np.clip(out2, -self.TRUE_PEAK_LIMIT, self.TRUE_PEAK_LIMIT)
            if is_stereo and res.ndim == 2:
                gain2 = out2 / np.where(np.abs(res_mono) > 1e-8, res_mono, 1.0)
                out_retry = np.zeros_like(res)
                for ch in range(res.shape[0]):
                    n_s = min(len(res[ch]), len(gain2))
                    out_retry[ch, :n_s] = res[ch, :n_s] * gain2[:n_s]
                    out_retry[ch, n_s:] = res[ch, n_s:]
                final = np.clip(np.nan_to_num(out_retry), -self.TRUE_PEAK_LIMIT, self.TRUE_PEAK_LIMIT).astype(
                    np.float32
                )
            else:
                final = out2.astype(np.float32)
            # §8.2 Observability: log final pearson after retry (universal guarantee ≥ 0.92)
            _final_mono = final.mean(axis=0) if final.ndim == 2 else final
            r_final = self._pearson(orig_mono[: len(_final_mono)], _final_mono[: len(orig_mono)])
            if r_final < 0.92:
                # Pearson still < 0.92 after retry — signal structure limits correlation
                # (e.g. heavily compressed 1970s vinyl).  Only warn if retry made no progress.
                _delta = r_final - r
                _pearson_improved = _delta > 0.001  # < 0.001 = konvergiert, kein weiterer Retry sinnvoll
                _log_fn = logger.debug if _pearson_improved else logger.warning
                _log_fn(
                    "§8.2 MDEM Mikro-Dynamik-Guarantee VERFEHLT: pearson=%.4f < 0.92 "
                    "(max_gain=%.1f dB, mode=retry, Δ=%.4f) — Rohausgabe nicht unterdrückt",
                    r_final,
                    max_gain,
                    _delta,
                )
            else:
                logger.info("§8.2 MDEM Micro-Dynamics pearson=%.4f ≥ 0.92 (retry, r_before=%.4f)", r_final, r)
            return final

        # §8.2 Observability: log final pearson (universal guarantee ≥ 0.92)
        if r < 0.92:
            logger.warning(
                "§8.2 MDEM Mikro-Dynamik-Guarantee VERFEHLT: pearson=%.4f < 0.92 "
                "(max_gain=%.1f dB) — Rohausgabe nicht unterdrückt",
                r,
                max_gain,
            )
        else:
            logger.debug("§8.2 MDEM Micro-Dynamics pearson=%.4f ≥ 0.92 (max_gain=%.1f dB)", r, max_gain)

        return out.astype(np.float32)

    def _morph_internal(
        self,
        res_mono: np.ndarray,
        orig_mono: np.ndarray,
        max_gain: float = 4.0,
    ) -> np.ndarray:
        """Interne Gain-Envelope-Berechnung und -Anwendung auf Mono-Signale (kein Retry).

        Default max_gain = 4.0 LU (Restoration mode, §2.30 v9.10.79).
        VERBOTEN: einheitliches 3.0 LU — nicht spec-konform (Fix X2 §2.30).
        """
        L_orig = self.compute_lufs_profile(orig_mono)
        L_rest = self.compute_lufs_profile(res_mono)
        n_frames = min(len(L_orig), len(L_rest))
        G = np.zeros(n_frames, dtype=np.float32)
        _QUIET_LUFS = -36.0  # §2.30 quiet-zone threshold (vinyl fade-out / restored noise floor)
        for k in range(n_frames):
            lo, lr = L_orig[k], L_rest[k]
            if not (math.isfinite(lo) and math.isfinite(lr)) or lo < self.MIN_LEVEL_LUFS:
                G[k] = 0.0
            elif lr < self.MIN_LEVEL_LUFS:
                # Restored near-silent: suppress positive boost (see morph() §2.30 guard)
                G[k] = float(np.clip(lo - lr, -max_gain, 0.0))
            elif lr < _QUIET_LUFS:
                # §2.30 quiet-zone guard: restored signal in vinyl/tape fade-out noise floor range
                # (-36 to -60 dBFS). Any positive boost here amplifies cleaned noise at
                # beginning/end of track. No positive gain allowed in quiet zone.
                G[k] = float(np.clip(lo - lr, -max_gain, 0.0))
            else:
                G[k] = float(np.clip(lo - lr, -max_gain, max_gain))
        G_smooth = _savgol_smooth(G)
        # §2.30 post-smoothing quiet-zone clamp: Savitzky-Golay can distribute positive gain
        # from music segments back into quiet regions. Clamp to 0 wherever restored is quiet.
        for k in range(n_frames):
            if L_rest[k] < _QUIET_LUFS and G_smooth[k] > 0.0:
                G_smooth[k] = 0.0
        hop = self.HOP_SIZE_SAMPLES
        n = len(res_mono)
        gain_envelope = np.ones(n, dtype=np.float32)
        for k in range(n_frames):
            start = k * hop
            ce = min(start + self.FRAME_SIZE_SAMPLES, n)
            if start >= n:
                break
            lg = float(10.0 ** (G_smooth[k] / 20.0))
            nxt = float(10.0 ** (G_smooth[k + 1] / 20.0)) if k + 1 < n_frames else lg
            gain_envelope[start:ce] = np.linspace(lg, nxt, ce - start, dtype=np.float32)

        # §2.30 tail-gap fix: Samples jenseits des letzten Frame-Endes fortsetzen.
        _last_covered = min((n_frames - 1) * hop + self.FRAME_SIZE_SAMPLES, n)
        if _last_covered < n and _last_covered > 0:
            _tail_audio = res_mono[_last_covered:]
            _tail_rms = float(np.sqrt(np.mean(_tail_audio**2) + 1e-12))
            _tail_rms_dbfs = 20.0 * math.log10(_tail_rms + 1e-12)
            _tail_in_quiet_zone = _tail_rms_dbfs < _QUIET_LUFS
            if _tail_in_quiet_zone:
                # §2.30 tail quiet-zone guard: restored tail is vinyl/tape noise floor.
                # Copy last gain but cap at 1.0 (no positive boost on tail).
                gain_envelope[_last_covered:] = min(float(gain_envelope[_last_covered - 1]), 1.0)
            else:
                gain_envelope[_last_covered:] = gain_envelope[_last_covered - 1]

        out = res_mono.copy()
        n_s = min(n, len(gain_envelope))
        out[:n_s] = res_mono[:n_s] * gain_envelope[:n_s]
        return out

    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        n = min(len(a), len(b))
        if n < 2:
            return 1.0
        a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
        am, bm = a.mean(), b.mean()
        num = float(np.mean((a - am) * (b - bm)))
        den = max(1e-15, float(np.std(a) * np.std(b)))
        val = num / den
        return float(np.clip(val, -1.0, 1.0)) if math.isfinite(val) else 0.0


# ---------------------------------------------------------------------------
# Singleton + Convenience
# ---------------------------------------------------------------------------

_instance: MicroDynamicsEnvelopeMorphing | None = None
_lock = threading.Lock()


def get_mdem() -> MicroDynamicsEnvelopeMorphing:
    """Thread-sicherer Singleton (§3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MicroDynamicsEnvelopeMorphing()
    return _instance


def morph_micro_dynamics(
    restored: np.ndarray,
    original: np.ndarray,
    sr: int = 48000,
    mode: str = "restoration",
) -> np.ndarray:
    """Convenience-Wrapper."""
    return get_mdem().morph(restored, original, sr, mode)


# ── Modul-Level-Konstanten (für direkten Import durch Tests und Consumer-Code) ──
# Spiegeln die Klassen-Attribute wider, sodass `from backend.core.micro_dynamics_envelope_morphing
# import FRAME_SIZE_SAMPLES` ohne Klassen-Instantiierung funktioniert.
FRAME_SIZE_SAMPLES: int = MicroDynamicsEnvelopeMorphing.FRAME_SIZE_SAMPLES
HOP_SIZE_SAMPLES: int = MicroDynamicsEnvelopeMorphing.HOP_SIZE_SAMPLES
MAX_GAIN_LU: float = MicroDynamicsEnvelopeMorphing.MAX_GAIN_LU
MIN_LEVEL_LUFS: float = MicroDynamicsEnvelopeMorphing.MIN_LEVEL_LUFS
PEARSON_TARGET: float = MicroDynamicsEnvelopeMorphing.PEARSON_TARGET


def compute_lufs_profile(audio: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Convenience-Wrapper für MicroDynamicsEnvelopeMorphing.compute_lufs_profile()."""
    return get_mdem().compute_lufs_profile(audio, sr)


__all__ = [
    # Modul-Level-Konstanten:
    "FRAME_SIZE_SAMPLES",
    "HOP_SIZE_SAMPLES",
    "MAX_GAIN_LU",
    "MIN_LEVEL_LUFS",
    "PEARSON_TARGET",
    "MicroDynamicsEnvelopeMorphing",
    "MorphResult",
    "compute_lufs_profile",
    "get_mdem",
    "morph_micro_dynamics",
]
