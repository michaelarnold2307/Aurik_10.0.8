#!/usr/bin/env python3
"""
Phase 39: Air Band Enhancement v2.0 - Professional
High-frequency shimmer and "air" enhancement (12-20 kHz).

Algorithm Overview:
1. Frequency Focus: 12-20 kHz (air band)
2. Harmonic Excitation: Generate missing HF content
3. Shelving EQ: Smooth HF lift
4. Saturation: Subtle harmonic distortion for warmth
5. Material Adaptation:
   - Shellac: Strong (restore bandwidth-limited treble)
   - Vinyl: Moderate (add air and sparkle)
   - Tape: Light (tape often has natural HF roll-off)
   - Digital: Moderate (add analog-style air)

Scientific Foundation:
- Fastl & Zwicker (2007): Psychoacoustics - Facts and Models
- Gabrielsson & Sjögren (1979): Perceived Sound Quality of Sound-Reproducing Systems
- Toole (1986): Loudspeaker Measurements and Their Relationship to Listener Preferences
- Moore (2012): An Introduction to the Psychology of Hearing

Industry Benchmarks:
- Maag EQ4 (Famous "Air Band" @ 40 kHz downsampled)
- Aphex Aural Exciter (HF harmonic generator)
- BBE Sonic Maximizer (Phase compensation + HF boost)
- Pultec HLF-3C (High-frequency boost)
- Dangerous Music BAX EQ (Shelf filter mastering)

Quality Target: 0.82 → 0.93 (+13% improvement)
Performance Target: <0.08× realtime

Author: Aurik Development Team
Version: 2.0.0 Professional
"""

import logging
import threading
import time

import numpy as np
from scipy import signal

from backend.core.audio_utils import safe_to_mono, stereo_channel_view, stereo_like
from backend.core.defect_scanner import MaterialType

from .output_guard import evaluate_output_guard
from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# §2.46b Spectral-Tilt-Preservation: material-adaptive tolerance in dB/octave
_TILT_TOLERANCE_P39: dict[str, float] = {
    "digital": 1.5,
    "cd_digital": 1.5,
    "streaming": 1.5,
    "tape": 1.875,
    "reel_tape": 1.875,
    "vinyl": 2.25,
    "minidisc": 2.25,
    "shellac": 3.0,
    "wax_cylinder": 3.0,
    "wire_recording": 3.0,
}


def _est_tilt_p39(audio: np.ndarray, sr: int) -> float:
    """Quick spectral tilt estimate in dB/octave (§2.46b)."""
    mono = safe_to_mono(audio) if audio.ndim == 2 else audio
    n = min(len(mono), 8192)
    if n < 64:
        return 0.0
    spec = np.abs(np.fft.rfft(mono[:n] * np.hanning(n))) + 1e-12
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    valid = (freqs >= 100.0) & (freqs <= sr * 0.45)
    if np.sum(valid) < 8:
        return 0.0
    log_f = np.log2(freqs[valid] + 1e-12)
    log_m = 20.0 * np.log10(spec[valid])
    log_f_c = log_f - log_f.mean()
    log_m_c = log_m - log_m.mean()
    denom = float(np.dot(log_f_c, log_f_c))
    return float(np.dot(log_f_c, log_m_c) / denom) if denom > 1e-10 else 0.0


class AirBandEnhancement(PhaseInterface):
    """
    Professional Air Band Enhancement Engine.

    Key Features:
    - High-frequency shelving (12-20 kHz)
    - Harmonic excitation (add sparkle)
    - Material-adaptive intensity
    - Psychoacoustic optimization
    - Analog-style warmth

    Use Cases:
    - Restore missing treble from bandwidth-limited sources
    - Add "air" and "shimmer" to vocals and instruments
    - Enhance perceived detail and clarity
    - Modernize vintage recordings

    Performance: <0.08× realtime on modern CPU
    """

    # Air band frequency range
    AIR_BAND_HZ = (12000, 20000)

    # Enhancement parameters (material-adaptive)
    AIR_CONFIG = {
        MaterialType.SHELLAC: {
            "shelf_gain_db": 6.0,  # Strong (restore missing HF)
            "shelf_freq_hz": 10000,
            "exciter_mix": 0.40,
            "saturation_drive": 0.30,
        },
        MaterialType.VINYL: {
            "shelf_gain_db": 4.0,
            "shelf_freq_hz": 12000,
            "exciter_mix": 0.30,
            "saturation_drive": 0.20,
        },
        MaterialType.TAPE: {
            "shelf_gain_db": 5.0,  # v9.10.114: ↑3.0→5.0 — Tape-HF nach Rauschreduktion sicher erweiterbar
            "shelf_freq_hz": 13000,
            "exciter_mix": 0.28,  # v9.10.114: ↑0.20→0.28
            "saturation_drive": 0.20,  # v9.10.114: ↑0.15→0.20
        },
        MaterialType.CD_DIGITAL: {
            "shelf_gain_db": 5.0,  # v9.10.114: ↑3.5→5.0 — CD hat klare HF-Basis
            "shelf_freq_hz": 12000,
            "exciter_mix": 0.30,  # v9.10.114: ↑0.25→0.30
            "saturation_drive": 0.25,
        },
        MaterialType.STREAMING: {
            "shelf_gain_db": 4.5,  # v9.10.114: ↑4.0→4.5
            "shelf_freq_hz": 11000,
            "exciter_mix": 0.32,  # v9.10.114: ↑0.30→0.32
            "saturation_drive": 0.22,  # v9.10.114: ↑0.20→0.22
        },
    }

    def __init__(self):
        super().__init__()
        self.name = "Air Band Enhancement v2 Professional"
        self._sos_air_cache: dict[int, np.ndarray] = {}
        self._shelf_coeffs: dict[tuple, tuple] = {}
        self._cache_lock = threading.Lock()

    def get_metadata(self) -> PhaseMetadata:
        """Return phase metadata."""
        return PhaseMetadata(
            phase_id="phase_39_air_band_enhancement",
            name="Air Band Enhancement v2 Professional",
            category=PhaseCategory.ENHANCEMENT,
            priority=5,
            dependencies=[],
            estimated_time_factor=0.08,
            version="2.0.0",
            memory_requirement_mb=35,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.93,
            description="High-frequency shimmer and air enhancement (12-20 kHz)",
        )

    def process(
        self, audio: np.ndarray, sample_rate: int, material: MaterialType = MaterialType.CD_DIGITAL, **kwargs
    ) -> PhaseResult:
        """
        Apply air band enhancement to audio.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Material type for adaptive processing

        Returns:
            PhaseResult with enhanced audio
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()
        self.validate_input(audio)

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        if _effective_strength <= 0.0:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material.name,
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=[],
            )

        is_stereo = audio.ndim == 2
        config = dict(self.AIR_CONFIG.get(material, self.AIR_CONFIG[MaterialType.CD_DIGITAL]))

        quality_mode = str(kwargs.get("quality_mode", "balanced")).lower()
        if quality_mode in ("quality", "maximum", "studio2026"):
            hq_scale = 1.12 if quality_mode in ("maximum", "studio2026") else 1.06
        else:
            hq_scale = 1.0

        config["shelf_gain_db"] = float(config["shelf_gain_db"] * _effective_strength)
        config["exciter_mix"] = float(config["exciter_mix"] * _effective_strength)
        config["saturation_drive"] = float(config["saturation_drive"] * _effective_strength)
        config["shelf_gain_db"] = float(np.clip(config["shelf_gain_db"] * hq_scale, 0.0, 7.0))
        config["exciter_mix"] = float(np.clip(config["exciter_mix"] * hq_scale, 0.0, 0.55))
        config["saturation_drive"] = float(np.clip(config["saturation_drive"] * hq_scale, 0.0, 0.45))

        # ── Era/Genre-adaptive air band scaling (context injection §2.x) ──
        _brillanz = kwargs.get("brillanz_target")
        _decade = kwargs.get("decade")
        _genre = kwargs.get("genre_label", "")
        if _brillanz is not None:
            _b_scale = 0.30 + 0.90 * float(_brillanz)
            _b_scale = max(0.65, min(1.20, _b_scale))
            config["shelf_gain_db"] *= _b_scale
            config["exciter_mix"] *= _b_scale
            logger.debug("Phase 39: brillanz_target=%.2f → air scale=%.2f", float(_brillanz), _b_scale)
        if _decade is not None:
            _dec = int(_decade)
            if _dec <= 1950:
                # Vintage: cap air enhancement (Spec §5: Rolloff ≤ 7 kHz not expand)
                config["shelf_gain_db"] = min(config["shelf_gain_db"], 2.5)
                config["exciter_mix"] = min(config["exciter_mix"], 0.15)
                config["shelf_freq_hz"] = max(config["shelf_freq_hz"], 14000)
            elif _dec <= 1970:
                # Early stereo era: moderate air
                config["shelf_gain_db"] *= 0.80
                config["exciter_mix"] *= 0.80
            elif _dec >= 2000:
                # Modern: less exciter (already has HF content)
                config["exciter_mix"] *= 0.75
        _genre_lower_39 = str(_genre).lower()
        if _genre_lower_39 in ("klassik", "oper"):
            # Classical: minimal air exciter to preserve natural overtones
            config["exciter_mix"] *= 0.50
            config["saturation_drive"] *= 0.50
        elif _genre_lower_39 == "reggae":
            # Reggae: warm/dark production style \u2014 HF air excitation would strip character.
            config["exciter_mix"] *= 0.40
            config["saturation_drive"] *= 0.50
        elif _genre_lower_39 in ("blues", "folk"):
            # Blues/Folk: natural warmth and harmonic texture, minimal air processing.
            config["exciter_mix"] *= 0.65
            config["saturation_drive"] *= 0.70
        elif _genre_lower_39 in ("electronic", "hip-hop"):
            # Electronic/Hip-Hop: typically already has engineered HF presence.
            config["exciter_mix"] *= 0.70
            config["saturation_drive"] *= 0.75

        # §2.41 (v9.10.116) SOTA: Ära-bewusste Air-Band-Deckelung aus SourceFidelityTarget.
        # Physikalische Invariante (Klangtreue §2.41): Luft-Anhebung nie ÜBER der ären-
        # typischen Original-Bandbreite — was das Original nicht enthalten konnte,
        # darf nicht synthetisiert werden (falsche Helligkeit).
        # Aber: akkumulierter HF-Verlust (Generationsverlust) soll KOMPENSIERT werden.
        _sfr_cal_39 = kwargs.get("song_calibration_profile", {})
        _sfr_bw_39 = float(_sfr_cal_39.get("source_fidelity_bandwidth_target_hz", 0.0))
        _sfr_hf_loss_39 = float(_sfr_cal_39.get("source_fidelity_hf_loss_db", 0.0))
        _sfr_conf_39 = float(_sfr_cal_39.get("source_fidelity_confidence", 0.5))
        if _sfr_bw_39 > 0.0 and _sfr_conf_39 >= 0.30:
            # Shelf-Frequenz-Deckelung: nie höher als 85% der Original-Bandbreite
            _air_bw_ceil = float(np.clip(_sfr_bw_39 * 0.85, 6000.0, 20000.0))
            if _air_bw_ceil < config["shelf_freq_hz"]:
                config["shelf_freq_hz"] = _air_bw_ceil
                logger.debug(
                    "Phase 39: era bw_target=%.0f → shelf_freq capped at %.0f Hz (conf=%.2f)",
                    _sfr_bw_39,
                    _air_bw_ceil,
                    _sfr_conf_39,
                )
            # HF-Verlust-Kompensation: mehr Generationen verloren → exciter-Boost
            # (kompensiert nur was real verloren ging, nicht over-synthesis)
            if _sfr_hf_loss_39 > 0.5:
                _loss_scale = float(np.clip(1.0 + _sfr_hf_loss_39 / 18.0 * _sfr_conf_39, 1.0, 1.35))
                config["exciter_mix"] = float(np.clip(config["exciter_mix"] * _loss_scale, 0.0, 0.55))
                logger.debug(
                    "Phase 39: hf_loss=%.1f dB → exciter_mix×%.2f (scale=%.2f)",
                    _sfr_hf_loss_39,
                    config["exciter_mix"],
                    _loss_scale,
                )

        # Measure initial HF energy
        hf_energy_before = self._measure_hf_energy(audio, sample_rate)

        # §2.51 M/S-Domain: Air Band auf Mid; Side unver\u00e4ndert
        if is_stereo:
            _ch0, _ch1 = stereo_channel_view(audio)
            mid = (_ch0 + _ch1) / np.sqrt(2.0)
            side = (_ch0 - _ch1) / np.sqrt(2.0)
            mid_enhanced = self._enhance_channel(mid, sample_rate, config)
            out_l = (mid_enhanced + side) / np.sqrt(2.0)
            out_r = (mid_enhanced - side) / np.sqrt(2.0)
            enhanced_audio = stereo_like(out_l, out_r, audio)
        else:
            enhanced_audio = self._enhance_channel(audio, sample_rate, config)

        if 0.0 < _effective_strength < 1.0:
            enhanced_audio = audio + _effective_strength * (enhanced_audio - audio)

        # §2.46b Spectral-Tilt-Guard: cap Air-Band synthesis if tilt deviates beyond tolerance
        _tilt_capped_p39 = False
        try:
            _mat_k39 = str(getattr(material, "value", str(material))).lower().replace(" ", "_").replace("-", "_")
            _tol39 = _TILT_TOLERANCE_P39.get(_mat_k39, 2.0)
            _tb39 = _est_tilt_p39(audio, sample_rate)
            _ta39 = _est_tilt_p39(enhanced_audio, sample_rate)
            _dev39 = abs(_ta39 - _tb39)
            if _dev39 > _tol39:
                _cap39 = float(np.clip(1.0 - (_dev39 - _tol39) / (_tol39 * 2.0), 0.5, 1.0))
                enhanced_audio = _cap39 * enhanced_audio + (1.0 - _cap39) * audio
                enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)
                _tilt_capped_p39 = True
                logger.info(
                    "phase_39 §2.46b tilt-cap: before=%.2f after=%.2f dev=%.2f tol=%.2f cap=%.2f",
                    _tb39,
                    _ta39,
                    _dev39,
                    _tol39,
                    _cap39,
                )
        except Exception as _tc39:
            logger.debug("phase_39 §2.46b tilt-cap skipped (graceful): %s", _tc39)

        enhanced_audio_pre_guard = enhanced_audio.copy()

        # Measure final HF energy
        hf_energy_after = self._measure_hf_energy(enhanced_audio, sample_rate)
        hf_boost_db = 20 * np.log10((hf_energy_after + 1e-10) / (hf_energy_before + 1e-10))

        execution_time = time.time() - start_time
        rt_factor = execution_time / (len(audio) / sample_rate)

        # ── HF-Kumulativ-Limit (Spec §8.2: Presence + Air kumulativ ≤ +4 dB) ──
        # Listening-Fatigue-Schutz: Gesamtanhebung 2–20 kHz limitieren
        hf_cumul_db = float(kwargs.get("hf_cumulative_gain_db", 0.0))
        MAX_HF_CUMUL_DB = 4.0
        if hf_cumul_db > MAX_HF_CUMUL_DB:
            logger.warning(
                "Phase 39: HF-Kumulativ-Limit erreicht (%.1f dB > %.1f dB) — "
                "Air-Band-Gain reduziert (Listening-Fatigue-Schutz, Spec §8.2)",
                hf_cumul_db,
                MAX_HF_CUMUL_DB,
            )
            # Gain-Korrektur: Überschuss rückgängig machen
            excess_db = hf_cumul_db - MAX_HF_CUMUL_DB
            gain_correction = 10 ** (-excess_db / 20.0)
            if enhanced_audio.ndim == 1:
                enhanced_audio = enhanced_audio * gain_correction
            else:
                enhanced_audio = enhanced_audio * gain_correction
            enhanced_audio = np.nan_to_num(enhanced_audio, nan=0.0, posinf=0.0, neginf=0.0)
            enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)

        output_guard_enabled = quality_mode in ("quality", "maximum", "studio2026")
        guard = evaluate_output_guard(
            original=audio,
            candidate=enhanced_audio,
            enabled=output_guard_enabled,
            max_abs_rms_delta_db=1.2,
            stereo_side_ratio_min=0.60,
            stereo_side_ratio_max=1.40,
        )
        if guard.fallback:
            enhanced_audio = enhanced_audio_pre_guard
            enhanced_audio = np.nan_to_num(enhanced_audio, nan=0.0, posinf=0.0, neginf=0.0)
            enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)

        return PhaseResult(
            success=True,
            audio=enhanced_audio,
            execution_time_seconds=execution_time,
            metadata={
                "material": material.name,
                "hf_boost_db": float(hf_boost_db),
                "shelf_gain_db": float(config["shelf_gain_db"]),
                "shelf_freq_hz": float(config["shelf_freq_hz"]),
                "rt_factor": float(rt_factor),
                "hf_cumulative_db": float(hf_cumul_db),
                "quality_mode": quality_mode,
                "hq_scale": hq_scale,
                "output_guard_enabled": output_guard_enabled,
                "output_guard_fallback": guard.fallback,
                "output_guard_reason": guard.reason,
                "rms_delta_db": guard.rms_delta_db,
                "stereo_side_ratio": guard.stereo_side_ratio,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "spectral_tilt_capped": _tilt_capped_p39,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            warnings=[],
        )

    def _enhance_channel(self, audio: np.ndarray, sample_rate: int, config: dict[str, float]) -> np.ndarray:
        """
        Enhance air band in a single audio channel.

        v2.1 fix: Correct combination formula.
        - Old (WRONG): shelved * 0.7 + excited * 0.3
          → Scales down the dry signal, loses energy at non-HF bands
        - New (CORRECT): shelved + wet_hf * exciter_mix
          → Keeps the shelf as the primary output; adds only the exciter delta on top
          → exciter_mix controls how much of the HF exciter blend is added

        Also: Bandwidth guard — only add exciter energy if source has content above 8 kHz
        (avoids adding artificial HF air above source's actual bandwidth limit).
        """
        # 1. Bandwidth guard: measure HF energy [10k–18k Hz] vs broadband
        if sample_rate >= 22050 and len(audio) > 8:
            n_fft = 1024
            segment = audio[: min(len(audio), n_fft * 8)]
            spec = np.abs(np.fft.rfft(segment)) ** 2
            freqs = np.fft.rfftfreq(len(segment), d=1.0 / sample_rate)
            broadband = float(np.sum(spec) + 1e-12)
            hf_mask = freqs >= 8000.0
            hf_content = float(np.sum(spec[hf_mask]) + 1e-12)
            hf_fraction = hf_content / broadband
            # If source has almost no HF content (bandwidth-limited), reduce exciter
            if hf_fraction < 0.005:
                # Very limited bandwidth — don't add synthetic air
                bw_scale = min(1.0, hf_fraction / 0.005)
                config = dict(config)  # don't mutate the outer config
                config["exciter_mix"] = config["exciter_mix"] * bw_scale
                logger.debug("Phase 39 BW-guard: hf_frac=%.4f → exciter scale=%.2f", hf_fraction, bw_scale)

        # 2. Apply the shelving EQ (primary output)
        shelved = self._apply_high_shelf(audio, sample_rate, config["shelf_freq_hz"], config["shelf_gain_db"])

        # 3. Generate exciter delta (only the HF harmonic component added to dry signal)
        excited = self._apply_exciter(audio, sample_rate, config["exciter_mix"], config["saturation_drive"])

        # 4. Correct combination: shelved is the base; add additive exciter delta
        #    exciter_mix is already applied inside _apply_exciter (returns audio + hf_delta*mix)
        #    So: combine shelf + (excited - audio) * additional blend factor
        hf_delta = excited - audio  # just the harmonic addition
        enhanced = shelved + hf_delta  # proper additive combination

        return enhanced

    def _apply_high_shelf(self, audio: np.ndarray, sample_rate: int, freq_hz: float, gain_db: float) -> np.ndarray:
        """Apply high-frequency shelving filter (biquad coefficients cached per key)."""
        cache_key = (sample_rate, freq_hz, gain_db)
        with self._cache_lock:
            if cache_key not in self._shelf_coeffs:
                w0 = 2 * np.pi * freq_hz / sample_rate
                A = 10 ** (gain_db / 40)
                alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / 0.707 - 1) + 2)
                b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
                b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
                b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
                a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
                a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
                a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
                b = np.array([b0, b1, b2]) / a0
                a = np.array([1, a1 / a0, a2 / a0])
                self._shelf_coeffs[cache_key] = (b, a)
            b, a = self._shelf_coeffs[cache_key]
        # §2.51 Anti-Zeitversatz: filtfilt (Zero-Phase) statt lfilter — Shelf-EQ darf keine
        # Gruppenlatenz erzeugen (hörbar als Zeitversatz auf HF-Transienten/Vokaleinsätzen).
        if len(audio) >= 9:
            return signal.filtfilt(b, a, audio)
        return signal.lfilter(b, a, audio)

    def _apply_exciter(self, audio: np.ndarray, sample_rate: int, mix: float, drive: float) -> np.ndarray:
        """Apply harmonic exciter to HF region (SOS filter cached per sample_rate)."""
        with self._cache_lock:
            if sample_rate not in self._sos_air_cache:
                self._sos_air_cache[sample_rate] = signal.butter(
                    4, self.AIR_BAND_HZ, btype="band", fs=sample_rate, output="sos"
                )
            sos = self._sos_air_cache[sample_rate]
        # §2.51 Anti-Zeitversatz: sosfiltfilt (Zero-Phase) statt sosfilt — hf + audio werden gemischt.
        hf = signal.sosfiltfilt(sos, audio)
        excited_hf = np.tanh(hf * drive * 2) / (drive + 0.5)
        return audio + excited_hf * mix

    def _measure_hf_energy(self, audio: np.ndarray, sample_rate: int) -> float:
        """Measure high-frequency energy (12-20 kHz RMS, cached SOS filter)."""
        if audio.ndim == 2:
            audio = audio[:, 0]  # Use left channel
        with self._cache_lock:
            if sample_rate not in self._sos_air_cache:
                self._sos_air_cache[sample_rate] = signal.butter(
                    4, self.AIR_BAND_HZ, btype="band", fs=sample_rate, output="sos"
                )
            sos = self._sos_air_cache[sample_rate]
        hf = signal.sosfilt(sos, audio)
        return float(np.sqrt(np.mean(hf**2)))
