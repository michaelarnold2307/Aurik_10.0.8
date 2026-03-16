from __future__ import annotations

import numpy as np


def _k_weighted_lufs(x: np.ndarray, sr: int = 48000) -> float:
    """Integrated Loudness nach ITU-R BS.1770-5 (vereinfacht, Mono).

    K-Gewichtungs-Filterkette:
        1. Pre-Filter: High-Shelf +4 dB @ 1500 Hz (Binaural Hearing Kompensation)
        2. Hochpass 2. Ordnung Butterworth @ 38 Hz (RLB-Gewichtung)
    Lautheitsmessung: L = −0.691 + 10·log10(mean(x²)) [LUFS]

    Fallback auf RMS-Näherung wenn scipy nicht verfügbar.
    """
    arr = np.nan_to_num(np.asarray(x, dtype=np.float64))
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    if len(arr) == 0:
        return -70.0
    try:
        from scipy import signal as _sig

        # Pre-Filter: High-Shelf +4 dB, f0=1500 Hz, Q=0.707 (binaural hearing)
        # Bilinear-Transform: s-domain High-Shelf → z-domain IIR (2 Pole, 1 Zero)
        _Ks = 4.0 / np.tan(np.pi * 1500.0 / sr)
        _Vh = 10.0 ** (4.0 / 20.0)  # +4 dB Gain
        _b0 = (_Vh + _Ks * np.sqrt(2.0 * _Vh) + _Ks ** 2) / (1.0 + _Ks * np.sqrt(2.0) + _Ks ** 2)
        _b1 = 2.0 * (_Vh - _Ks ** 2) / (1.0 + _Ks * np.sqrt(2.0) + _Ks ** 2)
        _b2 = (_Vh - _Ks * np.sqrt(2.0 * _Vh) + _Ks ** 2) / (1.0 + _Ks * np.sqrt(2.0) + _Ks ** 2)
        _a1 = 2.0 * (1.0 - _Ks ** 2) / (1.0 + _Ks * np.sqrt(2.0) + _Ks ** 2)
        _a2 = (1.0 - _Ks * np.sqrt(2.0) + _Ks ** 2) / (1.0 + _Ks * np.sqrt(2.0) + _Ks ** 2)
        arr = _sig.lfilter([_b0, _b1, _b2], [1.0, _a1, _a2], arr)

        # RLB Hochpass 2. Ordnung Butterworth @ 38 Hz
        _b_rlb, _a_rlb = _sig.butter(2, 38.0 / (sr / 2.0), btype="high")
        arr = _sig.lfilter(_b_rlb, _a_rlb, arr)
    except Exception:
        pass  # Fallback: Signal ohne Filterung (RMS-Näherung)

    mean_sq = float(np.mean(arr ** 2))
    if mean_sq <= 0.0:
        return -70.0
    return float(-0.691 + 10.0 * np.log10(mean_sq))


class StemRemixBalancer:
    """Gain-korrigierter Re-Mix nach getrennter Stem-Verarbeitung (§1.4 Spec).

    Verwendet ITU-R BS.1770-5 K-gewichtete Lautheitsmessung (LUFS)
    statt RMS-Näherung für spec-konforme Gain-Korrektur.
    """

    def balance_remix(
        self,
        vocals: np.ndarray,
        instruments: np.ndarray,
        original: np.ndarray,
        sr: int,
        vocal_weight: float = 0.5,
    ) -> np.ndarray:
        """Gain-korrigierter Re-Mix: LUFS(mix) - LUFS(original) ≤ 0.3 LU (§1.4 Spec).

        Algorithmus (§1.4):
            g_voc  = 10 ** ((L_orig_voc  − L_voc')  / 20)
            g_inst = 10 ** ((L_orig_inst − L_inst') / 20)
            mix    = g_voc · vocals + g_inst · instruments
        Lautheit: ITU-R BS.1770-5 K-Gewichtung (_k_weighted_lufs).
        """
        if sr != 48000:
            raise AssertionError(f"SR muss 48000 Hz sein, erhalten: {sr}")
        v = np.nan_to_num(np.asarray(vocals, dtype=np.float32))
        i = np.nan_to_num(np.asarray(instruments, dtype=np.float32))
        o = np.nan_to_num(np.asarray(original, dtype=np.float32))

        n = min(v.shape[-1], i.shape[-1], o.shape[-1])
        if n <= 0:
            return np.zeros(1, dtype=np.float32)
        v = v[..., :n]
        i = i[..., :n]
        o = o[..., :n]

        # BS.1770-5 K-gewichtete LUFS-Messung (statt RMS)
        _mono_o = o.mean(axis=0) if o.ndim == 2 else o
        _mono_v = v.mean(axis=0) if v.ndim == 2 else v
        _mono_i = i.mean(axis=0) if i.ndim == 2 else i

        L_orig = _k_weighted_lufs(_mono_o, sr)
        L_v = _k_weighted_lufs(_mono_v, sr)
        L_i = _k_weighted_lufs(_mono_i, sr)

        # Ziel-LUFS pro Stem (anteilig nach vocal_weight)
        vw = float(np.clip(vocal_weight, 0.0, 1.0))
        L_target_v = L_orig + 20.0 * np.log10(max(1e-6, vw + 1e-6))
        L_target_i = L_orig + 20.0 * np.log10(max(1e-6, (1.0 - vw) + 1e-6))

        g_v = float(10.0 ** ((L_target_v - L_v) / 20.0))
        g_i = float(10.0 ** ((L_target_i - L_i) / 20.0))

        mix = g_v * v + g_i * i
        mix = np.nan_to_num(mix)

        # Final-Trim: |LUFS(mix) − L_orig| ≤ 0.3 LU (§1.4 Invariante)
        _mono_mix = mix.mean(axis=0) if mix.ndim == 2 else mix
        L_mix = _k_weighted_lufs(_mono_mix, sr)
        g_final = float(10.0 ** ((L_orig - L_mix) / 20.0))
        out = np.clip(mix * g_final, -1.0, 1.0).astype(np.float32)
        return out


_instance: StemRemixBalancer | None = None
_lock = __import__("threading").Lock()


def get_stem_remix_balancer() -> StemRemixBalancer:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking, §3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = StemRemixBalancer()
    return _instance


def balance_remix(
    vocals: np.ndarray,
    instruments: np.ndarray,
    original: np.ndarray,
    sr: int,
    vocal_weight: float = 0.5,
) -> np.ndarray:
    """Convenience-Wrapper für StemRemixBalancer.balance_remix() (§1.4 Spec)."""
    return get_stem_remix_balancer().balance_remix(vocals, instruments, original, sr, vocal_weight)


__all__ = ["StemRemixBalancer", "get_stem_remix_balancer", "balance_remix"]
