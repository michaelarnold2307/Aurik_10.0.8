"""
media_chain_detector.py

Modul zur Erkennung und Rekonstruktion der Medienkette (Signalübertragungsstufen) aus Audiodateien.
Erkennt typische Merkmale von Vinyl, Tape, Kassette, Digital, MP3 usw. und gibt eine plausible
Kette mit Konfidenz zurück.

Algorithmen (alle physikalisch begründet, Post-2018-DSP, kein Legacy-Code):
    rumble:             RMS-Energie 20–80 Hz / Gesamt-RMS  (Zwicker 1990, Vinyl/Shellac-Kennzeichen)
    wow_flutter:        Autokorrelations-Pitch-Tracking → CoV über Zeit  (Mauch & Dixon 2014 Referenz)
    lf_modulation:      Hilbert-Hüllkurve → LF-Spektrum 2–100 Hz  (AM-Demodulation)
    mp3_lowpass:        HF-Energie-Verhältnis + Spectral Flatness >15 kHz  (ISO 11172-3)
    hiss_level:         10. Perzentil der Frame-RMS im 4–8 kHz-Band (Tape-Rauschboden-Proxy)
    crackle_density:    Impulsdichte: Frames mit >10× Median-Energie in 1-ms-Fenstern
    hf_rolloff_hz:      Effektive Bandbreite (95%-Energieanteil, STFT-Rolloff)
    dc_offset:          |mean(signal)|
    noise_floor_db:     5. Perzentil der Frame-RMS → dBFS (Rauschboden)
    saturation:         Gerader-Harmonik-Anteil (H2+H4) / H1 (Tape/Röhren-Kompression)
    pre_echo:           Energie vor Transienten / Energie nach Transienten (MP3/AAC-Artefakt)
    stereo_correlation: L-R-Pearson-Korrelation im 4 kHz-Breitband
    hf_flatness:        Geometr./Arithm. Mittelverhältnis im 12–20 kHz-Band (MP3-Null-Indikator)
"""

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MediaChainDetector:
    """Erkennt Medienketten aus physikalisch messbaren Audiosignal-Merkmalen.

    Unterstützte Medien:
        Analoge: Vinyl, Shellac, Tape (offen), Kassette, Drahtton
        Digitale: CD, DAT, MiniDisc, MP3, AAC, FLAC, WAV, OGG, WMA, M4A,
                  DSD, Streaming, Download, Blu-ray, SACD, DVD-Audio
    """

    MEDIA_TYPES = [
        # Analoge Medien
        "vinyl",
        "shellac",
        "tape",
        "cassette",
        "open_reel",
        "wire_recording",
        # Digitale Medien
        "cd",
        "dat",
        "minidisc",
        "mp3",
        "aac",
        "flac",
        "wav",
        "ogg",
        "wma",
        "m4a",
        "dsd",
        "streaming",
        "digital_download",
        "blu_ray",
        "sacd",
        "dvd_audio",
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # ÖFFENTLICH: Kettenerkennungs-API
    # ─────────────────────────────────────────────────────────────────────────

    def detect_chain(self, audio: np.ndarray, sr: int) -> list[dict[str, Any]]:
        """Analysiert das Audiosignal und gibt die erkannte Medienkette zurück.

        Returns:
            Liste von Medienstufen mit Konfidenz, z.B.::

                [{"medium": "vinyl", "confidence": 0.91},
                 {"medium": "mp3",   "confidence": 0.83}]
        """
        features = self._extract_features(audio, sr)
        chain: list[dict[str, Any]] = []

        rumble = features.get("rumble", 0.0)
        wow_flutter = features.get("wow_flutter", 0.0)
        lf_mod = features.get("lf_modulation", 0.0)
        mp3_lowpass = features.get("mp3_lowpass", 0.0)
        hiss = features.get("hiss_level", 0.0)
        crackle = features.get("crackle_density", 0.0)
        rolloff = features.get("hf_rolloff_hz", float(sr / 2))
        noise_floor = features.get("noise_floor_db", -90.0)
        saturation = features.get("saturation", 0.0)
        pre_echo = features.get("pre_echo", 0.0)
        stereo_corr = features.get("stereo_correlation", 1.0)
        hf_flatness = features.get("hf_flatness", 0.5)

        def _clip(v: float) -> float:
            return float(np.clip(v, 0.0, 0.99))

        # ── Shellac (extrem hoher Rauschboden, BW < 8 kHz, starkes Knistern)
        if noise_floor > -30.0 and rolloff < 9000.0 and (rumble > 0.12 or crackle > 0.08):
            conf = _clip(
                0.40 * min((noise_floor + 50.0) / 20.0, 1.0)
                + 0.35 * max(0.0, 1.0 - rolloff / 9000.0)
                + 0.15 * rumble
                + 0.10 * crackle,
            )
            chain.append({"medium": "shellac", "confidence": max(conf, 0.70)})

        # ── Vinyl (Rumble + Crackle + gut erhaltene HF)
        elif rumble > 0.08 and crackle > 0.04:
            conf = _clip(0.50 * rumble + 0.30 * crackle + 0.20 * (1.0 - max(0.0, hf_flatness - 0.3)))
            chain.append({"medium": "vinyl", "confidence": max(conf, 0.65)})

        # ── Wire Recording (extrem enge BW < 5 kHz + starker Jitter)
        if rolloff < 5000.0 and (wow_flutter > 0.18 or lf_mod > 0.20):
            chain.append({"medium": "wire_recording", "confidence": 0.78})

        # ── Kassette / Spulenband (Wow/Flutter + Hiss + BW-Begrenzung)
        if (wow_flutter > 0.08 or lf_mod > 0.08) and hiss > 0.015:
            conf = _clip(
                0.35 * max(wow_flutter, lf_mod)
                + 0.30 * hiss
                + 0.20 * max(0.0, 1.0 - rolloff / 19000.0)
                + 0.15 * max(0.0, 1.0 - stereo_corr),
            )
            # Spulenband: mehr Saturation und besseres HF als Kassette
            if saturation > 0.12 and rolloff > 12000.0:
                chain.append({"medium": "open_reel", "confidence": max(conf, 0.68)})
            else:
                chain.append({"medium": "cassette", "confidence": max(conf, 0.68)})

        # ── Tape generisch (Hiss + Röhren-Saturation ohne klares Wow/Flutter)
        elif hiss > 0.04 and saturation > 0.06 and wow_flutter < 0.08:
            conf = _clip(0.50 * hiss + 0.40 * saturation + 0.10)
            chain.append({"medium": "tape", "confidence": max(conf, 0.62)})

        # ── MP3 (harter Lowpass + fehlende HF-Flachheit + Pre-Echo)
        if mp3_lowpass > 0.35 and hf_flatness < 0.20:
            conf = _clip(
                0.45 * mp3_lowpass + 0.30 * (1.0 - hf_flatness) + 0.25 * max(pre_echo, 0.0),
            )
            chain.append({"medium": "mp3", "confidence": max(conf, 0.78)})

        # ── AAC (weicherer MP3-ähnlicher Lowpass, weniger Pre-Echo)
        elif mp3_lowpass > 0.20 and hf_flatness < 0.35 and pre_echo < 0.12:
            conf = _clip(0.60 * mp3_lowpass + 0.35 * (1.0 - hf_flatness) + 0.05)
            chain.append({"medium": "aac", "confidence": max(conf, 0.68)})

        # ── MiniDisc (ATRAC-Artefakte: Pre-Echo + moderater Lowpass)
        if pre_echo > 0.12 and mp3_lowpass > 0.10 and rolloff < 20000.0:
            if not any(m["medium"] in ("mp3", "aac") for m in chain):
                chain.append({"medium": "minidisc", "confidence": 0.66})

        # ── Streaming (allgemeines Kompressionszeichen, variables Profil)
        if mp3_lowpass > 0.12 and rolloff < 18000.0:
            if not any(m["medium"] in ("mp3", "aac", "minidisc") for m in chain):
                chain.append({"medium": "streaming", "confidence": 0.62})

        # ── DAT (breites BW, exzellente Dynamik, leichtes digitales Dither-Rauschen)
        if rolloff > 20500.0 and mp3_lowpass < 0.05 and noise_floor < -80.0:
            chain.append({"medium": "dat", "confidence": 0.74})

        # ── CD (gute BW, typisches Dither-Rauschen)
        if rolloff > 19000.0 and noise_floor < -70.0 and mp3_lowpass < 0.10:
            if not any(m["medium"] in ("dat",) for m in chain):
                chain.append({"medium": "cd", "confidence": 0.70})

        # ── Fallback: sauberes digitales Material
        if not chain:
            if rolloff > 18000.0 and noise_floor < -60.0:
                chain.append({"medium": "wav", "confidence": 0.68})
            else:
                chain.append({"medium": "digital", "confidence": 0.60})

        logger.debug("MediaChainDetector: Erkannte Kette = %s (features=%s)", chain, features)
        return chain

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVAT: Physikalisch begründete Feature-Extraktion
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_features(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Extrahiert 13 physikalisch begründete Merkmale für die Medienerkennung.

        Alle Features sind in [0, 1] normalisiert (außer ``noise_floor_db`` in dBFS
        und ``hf_rolloff_hz`` in Hz) und NaN/Inf-sicher.

        Algorithmus:
            1.  rumble:            Butterworth-BP 20–80 Hz → RMS-Verhältnis
            2.  wow_flutter:       Autokorrelations-Pitch-Tracking 50-ms-Fenster → CoV × 5
            3.  mp3_lowpass:       RFFT → 1 − tanh(E_HF/E_LF · 30)  (E_HF = >15 kHz)
            4.  hiss_level:        BP 4–8 kHz → 10. Perzentil Frame-RMS / Gesamt-RMS
            5.  crackle_density:   1-ms-Frame-Energie > 10 × Median → Dichte
            6.  hf_rolloff_hz:     95 % Energie-Rolloff via kumuliertem RFFT-Spektrum
            7.  dc_offset:         |mean(signal)|
            8.  noise_floor_db:    5. Perzentil Frame-RMS (50 ms) → dBFS
            9.  saturation:        (H2+H4)/H1 × 2 (geradharm. Anteil, Tape/Röhre)
            10. pre_echo:          Energie der Pre-Frames bei Transienten / Post-Frames
            11. stereo_correlation: Pearson L-R-Korrelation (nur Stereo)
            12. lf_modulation:     Hilbert-Hüllkurve → Energie 2–100 Hz / Gesamt
            13. hf_flatness:       Geom./Arithm.-Mittel im 12–20 kHz-Band
        """
        try:
            from scipy import signal as scipy_signal
        except ImportError:
            scipy_signal = None

        features: dict[str, float] = {}

        # ── Vorverarbeitung: Mono, Float32 ──────────────────────────────────
        if audio is None or len(audio) == 0:
            return self._zero_features()
        if audio.ndim > 1:
            mono = audio.mean(axis=1).astype(np.float32)
        else:
            mono = audio.astype(np.float32)
        # NaN/Inf-Guard
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(mono)
        nyq = sr / 2.0
        if n < sr // 10:  # < 100 ms → zu kurz
            return self._zero_features()

        # ── 1. RUMBLE: Energie 20–80 Hz / Gesamt-RMS ────────────────────────
        try:
            if scipy_signal is not None:
                b, a = scipy_signal.butter(4, [20.0 / nyq, 80.0 / nyq], btype="band")
                rumble_sig = scipy_signal.lfilter(b, a, mono)
            else:
                rumble_sig = self._simple_bandpass_fft(mono, sr, 20.0, 80.0)
            rms_tot = math.sqrt(float(np.mean(mono**2)) + 1e-12)
            rms_rumble = math.sqrt(float(np.mean(rumble_sig**2)) + 1e-12)
            features["rumble"] = float(np.clip(rms_rumble / rms_tot, 0.0, 1.0))
        except Exception:
            features["rumble"] = 0.0

        # ── 2. WOW/FLUTTER: Pitch-Varianz über Zeitfenster ──────────────────
        # Autokorrelations-Methode: Pitch pro 50-ms-Frame → CV (Coefficient of Variation)
        try:
            frame_len = int(sr * 0.05)
            hop = frame_len // 2
            min_lag = max(1, int(sr / 600))  # bis 600 Hz
            max_lag = min(int(sr / 50), frame_len - 1)  # ab 50 Hz
            pitches = []
            for start in range(0, n - frame_len, hop):
                frame = mono[start : start + frame_len].copy()
                frame -= frame.mean()
                frame *= np.hanning(frame_len)
                corr = np.correlate(frame, frame, mode="full")[frame_len - 1 :]
                lag_range = corr[min_lag:max_lag]
                if len(lag_range) > 0 and lag_range.max() > 0:
                    pitches.append(sr / (int(np.argmax(lag_range)) + min_lag + 1))
            if len(pitches) >= 4:
                pa = np.array(pitches)
                cv = float(np.std(pa)) / (float(np.mean(pa)) + 1e-9)
                features["wow_flutter"] = float(np.clip(cv * 5.0, 0.0, 1.0))
            else:
                features["wow_flutter"] = 0.0
        except Exception:
            features["wow_flutter"] = 0.0

        # ── 3. MP3 LOWPASS: HF-Energie 15–20 kHz / Low-Band-Energie ────────
        try:
            fft_n = min(4096, n)
            spec = np.abs(np.fft.rfft(mono[:fft_n], n=fft_n)) ** 2
            freqs = np.fft.rfftfreq(fft_n, d=1.0 / sr)
            e_low = float(np.sum(spec[freqs < 15000.0]) + 1e-12)
            e_high = float(np.sum(spec[freqs >= 15000.0]) + 1e-12)
            hf_ratio = e_high / e_low
            # Bei MP3: nahezu 0; bei unkomprimiert: 0.03–0.12
            features["mp3_lowpass"] = float(np.clip(1.0 - math.tanh(hf_ratio * 30.0), 0.0, 1.0))
        except Exception:
            features["mp3_lowpass"] = 0.0

        # ── 4. HISS-LEVEL: Rauschboden 4–8 kHz ─────────────────────────────
        try:
            if scipy_signal is not None:
                lo = 4000.0 / nyq
                hi = min(8000.0 / nyq, 0.995)
                b_h, a_h = scipy_signal.butter(4, [lo, hi], btype="band")
                hiss_sig = scipy_signal.lfilter(b_h, a_h, mono)
            else:
                hiss_sig = self._simple_bandpass_fft(mono, sr, 4000.0, 8000.0)
            fw = int(sr * 0.02)
            if fw > 0 and n >= fw:
                rms_h = np.array(
                    [math.sqrt(float(np.mean(hiss_sig[i * fw : (i + 1) * fw] ** 2)) + 1e-12) for i in range(n // fw)]
                )
                hiss_floor = float(np.percentile(rms_h, 10))
                rms_main = math.sqrt(float(np.mean(mono**2)) + 1e-12)
                features["hiss_level"] = float(np.clip(hiss_floor / rms_main, 0.0, 1.0))
            else:
                features["hiss_level"] = 0.0
        except Exception:
            features["hiss_level"] = 0.0

        # ── 5. CRACKLE DENSITY: Impulsdichte ────────────────────────────────
        try:
            w = max(1, int(sr * 0.001))  # 1 ms
            n_w = n // w
            if n_w > 2:
                fe = np.array([math.sqrt(float(np.mean(mono[i * w : (i + 1) * w] ** 2)) + 1e-12) for i in range(n_w)])
                med_e = float(np.median(fe)) + 1e-12
                n_crackle = int(np.sum(fe > med_e * 10.0))
                features["crackle_density"] = float(np.clip(n_crackle / n_w, 0.0, 1.0))
            else:
                features["crackle_density"] = 0.0
        except Exception:
            features["crackle_density"] = 0.0

        # ── 6. HF ROLLOFF: Effektive Bandbreite (95%-Energie) ───────────────
        try:
            fft_n2 = min(4096, n)
            spec2 = np.abs(np.fft.rfft(mono[:fft_n2], n=fft_n2)) ** 2
            freqs2 = np.fft.rfftfreq(fft_n2, d=1.0 / sr)
            cumsum = np.cumsum(spec2)
            threshold = cumsum[-1] * 0.95
            idx = int(np.searchsorted(cumsum, threshold))
            idx = min(idx, len(freqs2) - 1)
            features["hf_rolloff_hz"] = float(freqs2[idx])
        except Exception:
            features["hf_rolloff_hz"] = float(nyq)

        # ── 7. DC OFFSET ────────────────────────────────────────────────────
        features["dc_offset"] = float(np.clip(abs(float(np.mean(mono))), 0.0, 1.0))

        # ── 8. NOISE FLOOR (dBFS) ───────────────────────────────────────────
        try:
            flw = int(sr * 0.05)
            if flw > 0 and n >= flw:
                rms_nf = np.array(
                    [math.sqrt(float(np.mean(mono[i * flw : (i + 1) * flw] ** 2)) + 1e-12) for i in range(n // flw)]
                )
                floor = float(np.percentile(rms_nf, 5)) + 1e-12
                features["noise_floor_db"] = float(20.0 * math.log10(floor))
            else:
                features["noise_floor_db"] = -90.0
        except Exception:
            features["noise_floor_db"] = -90.0

        # ── 9. SATURATION: Gerader Harmonischer Anteil (H2, H4) / H1 ────────
        # Physik: Tape- und Röhren-Kompression → dominante H2/H4 gegenüber H3/H5 (Clipping)
        try:
            fft_ns = min(8192, n)
            spec_s = np.abs(np.fft.rfft(mono[:fft_ns], n=fft_ns))
            freqs_s = np.fft.rfftfreq(fft_ns, d=1.0 / sr)
            valid = freqs_s > 80.0
            spec_v = spec_s.copy()
            spec_v[~valid] = 0.0
            if spec_v.max() > 0:
                f0_idx = int(np.argmax(spec_v))
                f0 = float(freqs_s[f0_idx])
                if f0 > 60.0:

                    def _harm(nk: int) -> float:
                        fh = f0 * nk
                        if fh >= nyq:
                            return 0.0
                        ih = int(round(fh * fft_ns / sr))
                        ih = min(ih, len(spec_s) - 1)
                        lo = max(0, ih - 3)
                        hi = min(len(spec_s), ih + 4)
                        return float(np.max(spec_s[lo:hi]))

                    h1 = _harm(1) + 1e-9
                    h2, h4 = _harm(2), _harm(4)
                    features["saturation"] = float(np.clip((h2 + h4) / h1 * 2.0, 0.0, 1.0))
                else:
                    features["saturation"] = 0.0
            else:
                features["saturation"] = 0.0
        except Exception:
            features["saturation"] = 0.0

        # ── 10. PRE-ECHO: Energie vor Transienten (Codec-Artefakt) ──────────
        try:
            fw_pe = int(sr * 0.02)
            if fw_pe > 0 and n >= fw_pe * 4:
                ef = np.array(
                    [
                        math.sqrt(float(np.mean(mono[i * fw_pe : (i + 1) * fw_pe] ** 2)) + 1e-12)
                        for i in range(n // fw_pe)
                    ]
                )
                if len(ef) > 3:
                    ratios = ef[1:] / (ef[:-1] + 1e-9)
                    scores = []
                    for i, r in enumerate(ratios):
                        if r > 3.0 and i > 0:
                            scores.append(ef[i] / (ef[i + 1] + 1e-9))
                    features["pre_echo"] = float(np.clip(float(np.mean(scores)) * 2.0 if scores else 0.0, 0.0, 1.0))
                else:
                    features["pre_echo"] = 0.0
            else:
                features["pre_echo"] = 0.0
        except Exception:
            features["pre_echo"] = 0.0

        # ── 11. STEREO-KORRELATION ──────────────────────────────────────────
        try:
            if audio.ndim == 2 and audio.shape[1] == 2:
                L = audio[:, 0].astype(np.float64)
                R = audio[:, 1].astype(np.float64)
                denom = math.sqrt(float(np.mean(L**2)) * float(np.mean(R**2))) + 1e-12
                features["stereo_correlation"] = float(np.clip(float(np.mean(L * R)) / denom, -1.0, 1.0))
            else:
                features["stereo_correlation"] = 1.0
        except Exception:
            features["stereo_correlation"] = 1.0

        # ── 12. LF MODULATION: AM-Demodulation 2–100 Hz (Wow+Flutter) ───────
        try:
            from scipy.signal import hilbert as _hilbert

            envelope = np.abs(_hilbert(mono)).astype(np.float32)
            env_n = min(4096, len(envelope))
            env_fft = np.abs(np.fft.rfft(envelope[:env_n], n=env_n)) ** 2
            env_freqs = np.fft.rfftfreq(env_n, d=1.0 / sr)
            mask_lf = (env_freqs >= 2.0) & (env_freqs <= 100.0)
            e_total = float(np.sum(env_fft)) + 1e-12
            e_mod = float(np.sum(env_fft[mask_lf]))
            features["lf_modulation"] = float(np.clip(e_mod / e_total * 20.0, 0.0, 1.0))
        except Exception:
            features["lf_modulation"] = features.get("wow_flutter", 0.0)

        # ── 13. HF SPECTRAL FLATNESS 12–20 kHz (MP3-Null-Indikator) ────────
        try:
            fft_n3 = min(4096, n)
            spec3 = np.abs(np.fft.rfft(mono[:fft_n3], n=fft_n3))
            freqs3 = np.fft.rfftfreq(fft_n3, d=1.0 / sr)
            hf_mask = freqs3 >= 12000.0
            hf_spec = spec3[hf_mask] + 1e-9
            if len(hf_spec) > 1:
                geo = math.exp(float(np.mean(np.log(hf_spec))))
                ari = float(np.mean(hf_spec))
                features["hf_flatness"] = float(np.clip(geo / ari, 0.0, 1.0))
            else:
                features["hf_flatness"] = 0.0
        except Exception:
            features["hf_flatness"] = 0.0

        # Finale NaN/Inf-Bereinigung
        for k, v in features.items():
            if not math.isfinite(v):
                features[k] = 0.0

        return features

    # ─────────────────────────────────────────────────────────────────────────
    # HILFSMETHODEN
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _zero_features() -> dict[str, float]:
        """Gibt einen sicheren Null-Feature-Dict zurück (für leeres Audio)."""
        return {
            "rumble": 0.0,
            "wow_flutter": 0.0,
            "mp3_lowpass": 0.0,
            "hiss_level": 0.0,
            "crackle_density": 0.0,
            "hf_rolloff_hz": 0.0,
            "dc_offset": 0.0,
            "noise_floor_db": -90.0,
            "saturation": 0.0,
            "pre_echo": 0.0,
            "stereo_correlation": 1.0,
            "lf_modulation": 0.0,
            "hf_flatness": 0.0,
        }

    @staticmethod
    def _simple_bandpass_fft(signal: np.ndarray, sr: int, lo_hz: float, hi_hz: float) -> np.ndarray:
        """FFT-basierter Bandpass-Filter (scipy-Fallback).

        Wendet eine Rechteck-Maske im Frequenzbereich an und transformiert zurück.
        """
        n = len(signal)
        spectrum = np.fft.rfft(signal, n=n)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        mask = (freqs >= lo_hz) & (freqs <= hi_hz)
        spectrum_masked = spectrum * mask
        return np.fft.irfft(spectrum_masked, n=n).astype(np.float32)
