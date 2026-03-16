"""WpePlugin - Musikorientierte Hallentfernung via WPE.

Primaer-Algorithmus: WPE (Weighted Prediction Error) - Nakatani et al. (2010),
Kinoshita et al. (2017).  Macht keine Sprachangaben, arbeitet rein auf dem
statistischen Schallfeldmodell -> korrekt fuer Musik, Orchester, historische
Aufnahmen.

Kaskade:
    1. nara_wpe (falls installiert) -- Referenz-Implementierung
    2. NumPy-WPE  (eingebaut, keine Abhaengigkeiten) -- vollstaendiger Algorithmus
    3. OMLSA-Wiener (Letzfall, nur Rauschreduktion ohne echte Hallentfernung)

Referenzen:
    Nakatani et al. (2010): "Speech Dereverberation Based on Variance-Normalized
        Delayed Linear Prediction"
    Kinoshita et al. (2017): "A Summary of the REVERB Challenge"
    Cohen (2003): OMLSA / IMCRA (Letzfall)
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_inst: WpePlugin | None = None

# WPE-Parameter (Spec §4.4 / §4.5)
_N_FFT: int = 2048  # Bessere Frequenzaufloesung fuer Musik (war: 1024)
_HOP: int = 512
_DELAY: int = 3  # Delta = 3 Frames (~32 ms bei 512 hop / 48 kHz)
_TAPS: int = 10  # L = Filterlaenge
_ITER: int = 3  # WPE-Iterationen (Konvergenz nach ~3 ausreichend)


# ---------------------------------------------------------------------------
# WPE -- numpy-only Implementierung (Nakatani 2010)
# ---------------------------------------------------------------------------


def _wpe_numpy(
    Y: np.ndarray,
    delay: int = _DELAY,
    L: int = _TAPS,
    n_iter: int = _ITER,
) -> np.ndarray:
    """Single-channel WPE-Dereverberation (vektorisiert, NumPy-only).

    Algorithmus (Nakatani 2010, Gl. 9-12):
        1. lambda[k,n] = |D[k,n]|^2           (Schaetzung der Rausch-PSD)
        2. Phi[k]   = sum_t y_bar y_bar* / lambda  (gewichtete Autokov.)
        3. psi[k]   = sum_t y_bar * y*  / lambda   (gewichtete Kreuzkov.)
        4. w[k]     = Phi[k]^(-1) * psi[k]         (Tikhonov-regularisiert)
        5. D[k,n]   = Y[k,n] - w[k]* . y_bar[n]

    Args:
        Y     : Komplexes STFT [freq_bins, time_frames]
        delay : Verzoegerung zum Trennen von Frueh-/Spaetreflexionen
        L     : Filter-Ordnung (Tap-Anzahl)
        n_iter: WPE-Iterationen

    Returns: Dereverberated complex STFT, same shape as Y
    """
    K, N = Y.shape
    T_full = N - delay - L  # nutzbare Frames nach Einblendung
    if T_full <= L:  # zu kurz -> no-op
        return Y.copy()

    D = Y.copy().astype(np.complex64)

    # Stapel verzoegerter Beobachtungen einmalig aufbauen [K, L, T_full]
    # Y_del[k, l, t] = Y[k, t + delay + l]   l=0..L-1
    Y_del = np.empty((K, L, T_full), dtype=np.complex64)
    for l_idx in range(L):
        Y_del[:, l_idx, :] = Y[:, l_idx + delay : l_idx + delay + T_full]

    tgt_slice = slice(delay + L, delay + L + T_full)

    for _ in range(n_iter):
        # 1. PSD-Schaetzung aus dereverberierten Signalanteil
        lam = np.maximum(np.abs(D[:, tgt_slice]) ** 2, 1e-12)  # [K, T_full]
        lam_inv = (1.0 / lam).astype(np.float32)  # [K, T_full]

        # 2. Gewichtete Kovarianzmatrizen (vektorisiert ueber K)
        Y_w = Y_del * lam_inv[:, np.newaxis, :]  # [K, L, T_full]
        Phi = np.einsum("klt,kmt->klm", Y_w, Y_del.conj()) / T_full  # [K, L, L]
        y_tgt = Y[:, tgt_slice].astype(np.complex64)  # [K, T_full]
        psi = np.einsum("klt,kt->kl", Y_w, y_tgt.conj()) / T_full  # [K, L]

        # 3. Tikhonov-Regularisierung & Loesung
        Phi += 1e-6 * np.eye(L, dtype=np.complex64)[np.newaxis, ...]

        try:
            w = np.linalg.solve(Phi, psi)  # [K, L]
        except np.linalg.LinAlgError:
            break  # singulaere Matrix -> Abbruch

        # 4. Subtraktion der Spaetreflexionsschaetzung
        D[:, tgt_slice] = y_tgt - np.einsum("kl,klt->kt", w.conj(), Y_del)

        np.nan_to_num(D, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return D


def _wpe_stft(
    mono: np.ndarray,
    sr: int,
    strength: float = 1.0,
) -> np.ndarray:
    """WPE auf Mono-Signal anwenden (STFT-Domaene mit scipy).

    strength: 0.0 = no-op, 1.0 = volle WPE-Subtraktion (lineares Mischen).
    """
    from scipy.signal import istft, stft

    _, _, Z = stft(mono, fs=sr, nperseg=_N_FFT, noverlap=_N_FFT - _HOP, window="hann")
    Z = Z.astype(np.complex64)
    Z_wpe = _wpe_numpy(Z, delay=_DELAY, L=_TAPS, n_iter=_ITER)

    if strength < 1.0:
        Z_wpe = (1.0 - strength) * Z + strength * Z_wpe

    _, out = istft(Z_wpe, fs=sr, nperseg=_N_FFT, noverlap=_N_FFT - _HOP, window="hann")
    out = np.clip(np.nan_to_num(out.astype(np.float32), 0.0), -1.0, 1.0)
    return out[: len(mono)]


def _wpe_nara(mono: np.ndarray, sr: int) -> np.ndarray | None:
    """nara_wpe-Bibliothek als Tier-1 (falls installiert)."""
    try:
        from nara_wpe.utils import istft as nwpe_istft, stft as nwpe_stft  # noqa: PLC0415
        from nara_wpe.wpe import wpe  # noqa: PLC0415

        Y = nwpe_stft(mono, size=_N_FFT, shift=_HOP)  # [T, K]
        Y_e = wpe(Y.T[..., np.newaxis])  # [K, T, 1]
        out = nwpe_istft(Y_e[:, :, 0].T, size=_N_FFT, shift=_HOP)
        out = np.clip(np.nan_to_num(out.astype(np.float32), 0.0), -1.0, 1.0)
        return out[: len(mono)]
    except Exception:
        return None


def _omlsa_fallback(
    mono: np.ndarray,
    sr: int,
    strength: float = 0.7,
) -> np.ndarray:
    """OMLSA-naher Wiener-Filter als absoluter Letzfall (nur Rauschreduktion)."""
    from scipy.ndimage import uniform_filter
    from scipy.signal import istft, stft

    _, _, Z = stft(mono, fs=sr, nperseg=1024, noverlap=768, window="hann")
    mag = np.abs(Z)
    phase = np.angle(Z)
    noise = np.maximum(uniform_filter(np.minimum.accumulate(mag, axis=1), size=(1, 25)), 1e-8)
    snr = mag / noise
    gain = np.clip(1.0 - strength / (snr + 1e-6), 0.1, 1.0)
    _, out = istft(gain * mag * np.exp(1j * phase), fs=sr, nperseg=1024, noverlap=768, window="hann")
    out = np.clip(np.nan_to_num(out.astype(np.float32), 0.0), -1.0, 1.0)
    return out[: len(mono)]


# ---------------------------------------------------------------------------
# Plugin-Klasse
# ---------------------------------------------------------------------------


class WpePlugin:
    """Musikorientierte Hallentfernung via WPE (Weighted Prediction Error).

    Kaskade: nara_wpe (optional) -> NumPy-WPE -> OMLSA-Wiener (Letzfall).

    WPE entfernt Spaetreflexionen durch iterative gewichtete lineare
    Praediktion auf verzoegerten Signalrahmen.  Keine Sprachangaben,
    voll musiksicher.
    """

    def __init__(self) -> None:
        logger.info("WpePlugin: WPE-Dereverberation initialisiert " "(Nakatani 2010, Kinoshita 2017).")

    def enhance(
        self,
        audio: np.ndarray,
        sr: int,
        strength: float = 0.85,
    ) -> np.ndarray:
        """WPE-Dereverberation.

        Args:
            audio   : float32 [samples], [samples, ch] oder [ch, samples]
            sr      : Sample-Rate (Spec-Invariante: 48 000 Hz)
            strength: Eingriffstaerke 0.0-1.0

        Returns: float32, selbe Form wie Eingang.
        """
        assert sr == 48_000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        if audio.ndim == 2:
            if audio.shape[0] > audio.shape[1]:  # [samples, ch]
                return np.stack(
                    [self.enhance(audio[:, c], sr, strength) for c in range(audio.shape[1])],
                    axis=1,
                )
            return np.stack(  # [ch, samples]
                [self.enhance(audio[c], sr, strength) for c in range(audio.shape[0])],
                axis=0,
            )

        mono = audio

        # Tier-1: nara_wpe (Referenz-Bibliothek, falls installiert)
        result = _wpe_nara(mono, sr)
        if result is not None:
            logger.debug("WpePlugin: nara_wpe Tier-1 erfolgreich.")
            return result

        # Tier-2: NumPy-WPE (eingebaut, keine Abhaengigkeiten)
        try:
            result = _wpe_stft(mono, sr, strength=strength)
            logger.debug("WpePlugin: NumPy-WPE Tier-2 erfolgreich.")
            return result
        except Exception as exc:
            logger.warning("WpePlugin: NumPy-WPE fehlgeschlagen (%s) -- OMLSA-Fallback.", exc)

        # Tier-3: OMLSA-Wiener (Letzfall)
        logger.warning("WpePlugin: OMLSA-Letzfall aktiv (kein echtes Dereverb).")
        return _omlsa_fallback(mono, sr, strength=strength)

    def dereverberate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Alias fuer enhance() mit voller WPE-Staerke."""
        return self.enhance(audio, sr, strength=1.0)


# ---------------------------------------------------------------------------
# Singleton + Convenience
# ---------------------------------------------------------------------------


def get_wpe_plugin() -> WpePlugin:
    """Thread-sicherer Singleton (Double-Checked Locking)."""
    global _inst
    if _inst is None:
        with _lock:
            if _inst is None:
                _inst = WpePlugin()
    return _inst


def enhance(audio: np.ndarray, sr: int, strength: float = 0.85) -> np.ndarray:
    """Convenience-Wrapper: WPE-Dereverberation."""
    return get_wpe_plugin().enhance(audio, sr, strength)


# ---------------------------------------------------------------------------
# Rueckwaertskompatibilitaet (deprecated names)
# ---------------------------------------------------------------------------
SgmsePlugin = WpePlugin
SGMSEPlugin = WpePlugin
get_sgmse_plugin = get_wpe_plugin
