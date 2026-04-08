"""
Aurik 9 — HolisticPerceptualGate §2.44 [RELEASE_MUST]
======================================================
Last gate before export. Measures holistic perceptual improvement
instead of checking individual goals only.

HPI > 0 → Export | HPI ≤ 0 → Rollback

§2.44 Referenz-Anker-Strategie (Restorability-abhängig):
  - Restorability > 70  → timbral_fidelity gegen Input (gute Annäherung ans Original)
  - Restorability 50–70 → 60% Input + 40% MERT-Referenz aus GP-Memory
  - Restorability ≤ 50  → 70% MERT-Referenz + 30% Input (Input zu weit vom Original)

MERT-Referenz-Memory: EMA (α=0.15) pro (genre × material × era_bin).
Fallback-Kaskade (5 Stufen) wenn kein passender Referenz-Vektor.
Referenz-Update nur wenn: HPI > 0.5 AND artifact_freedom ≥ 0.95 AND P1/P2 bestanden.

Reference: Spec 02 §2.44, §2.49 (artifact_freedom)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ── EMA constant for reference-memory updates ──────────────────────────────
_EMA_ALPHA: float = 0.15
_MIN_OBS_CALIBRATED: int = 3  # < 3 obs → Bootstrap mit erhöhter Unsicherheit

# ── Singleton ──────────────────────────────────────────────────────────────
_instance: HolisticPerceptualGate | None = None
_lock = threading.Lock()


def get_holistic_gate() -> HolisticPerceptualGate:
    """Thread-safe Singleton accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = HolisticPerceptualGate()
    return _instance


@dataclass
class _RefEntry:
    """One entry in the reference memory (spectral embedding + EMA state)."""

    embedding: np.ndarray  # shape (n_mels,) — spectral prototype
    obs_count: int = 0  # number of successful updates
    calibrated: bool = False  # True once obs_count >= _MIN_OBS_CALIBRATED


@dataclass
class HPIResult:
    """Result of HPI evaluation."""

    hpi: float
    passed: bool
    mert_similarity: float = 1.0
    timbral_fidelity: float = 1.0
    artifact_freedom: float = 1.0
    emotional_arc_preservation: float = 1.0
    studio_quality_gain: float = 1.0
    pqs_improvement: float = 1.0
    is_studio_mode: bool = False
    detail: dict = field(default_factory=dict)


class HolisticPerceptualGate:
    """§2.44 Holistic Perceptual Gate — last gate before export."""

    def __init__(self) -> None:
        # §2.44 MERT-Reference-Memory: key = (genre, material, era_bin)
        self._ref_memory: dict[tuple[str, str, str], _RefEntry] = {}
        self._ref_lock = threading.Lock()
        # MERT similarity path is optional and must never break the gate.
        self._mert_path_disabled: bool = False

    def evaluate_restoration(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
        artifact_freedom: float = 1.0,
        emotional_arc_score: float = 1.0,
        restorability_score: float = 70.0,
        genre: str = "DEFAULT",
        material: str = "digital",
        era_bin: str = "post-1990",
    ) -> HPIResult:
        """Evaluate HPI for Restoration mode.

        HPI = MERT_similarity × timbral_fidelity × artifact_freedom × emotional_arc_preservation

        §2.44 Referenz-Anker-Strategie (Restorability-abhängig):
          - > 70: timbral_fidelity gegen Input
          - 50–70: 60% Input + 40% MERT-Referenz
          - ≤ 50: 30% Input + 70% MERT-Referenz
        """
        mert_sim = self._compute_mert_similarity(original, restored, sr)

        # §2.44 Restorability-dependent anchor weights
        if restorability_score > 70.0:
            input_weight, ref_weight = 1.0, 0.0
        elif restorability_score >= 50.0:
            input_weight, ref_weight = 0.6, 0.4
        else:
            input_weight, ref_weight = 0.3, 0.7

        timbral_input = self._compute_timbral_fidelity(original, restored, sr)
        if ref_weight > 0.0:
            ref_vec = self._get_reference_vector(genre, material, era_bin)
            if ref_vec is not None:
                rest_embed = self._compute_embedding(restored, sr)
                timbral_ref = float(np.clip(self._cosine_similarity(ref_vec, rest_embed), 0.0, 1.0))
            else:
                # Fallback-Kaskade §2.44 Stufe 5: kein Ref-Vektor → rein gegen Input
                timbral_ref = timbral_input
        else:
            timbral_ref = timbral_input

        timbral = input_weight * timbral_input + ref_weight * timbral_ref

        hpi = mert_sim * timbral * artifact_freedom * emotional_arc_score

        # §2.44: RestorabilityEstimator > 0.85 → stricter gate
        if restorability_score > 85.0:
            hpi = hpi * 0.95

        passed = hpi > 0.0 and artifact_freedom >= 0.95

        logger.info(
            "§2.44 HPI(Restoration)=%.4f passed=%s "
            "(mert=%.3f timbral=%.3f[in=%.3f ref=%.3f w=%.1f/%.1f] artifact=%.3f emotional=%.3f restorability=%.1f)",
            hpi,
            passed,
            mert_sim,
            timbral,
            timbral_input,
            timbral_ref,
            input_weight,
            ref_weight,
            artifact_freedom,
            emotional_arc_score,
            restorability_score,
        )

        return HPIResult(
            hpi=round(hpi, 4),
            passed=passed,
            mert_similarity=round(mert_sim, 4),
            timbral_fidelity=round(timbral, 4),
            artifact_freedom=round(artifact_freedom, 4),
            emotional_arc_preservation=round(emotional_arc_score, 4),
            is_studio_mode=False,
            detail={
                "restorability_score": restorability_score,
                "strict_gate": restorability_score > 85.0,
                "input_weight": input_weight,
                "ref_weight": ref_weight,
                "timbral_input": timbral_input,
                "timbral_ref": timbral_ref,
                "genre": genre,
                "material": material,
                "era_bin": era_bin,
            },
        )

    def update_reference_memory(
        self,
        restored: np.ndarray,
        sr: int,
        hpi: float,
        artifact_freedom: float,
        p1_p2_passed: bool,
        genre: str = "DEFAULT",
        material: str = "digital",
        era_bin: str = "post-1990",
    ) -> None:
        """§2.44 Update MERT reference memory after successful restoration.

        Quality-Gate: only HPI > 0.5 AND artifact_freedom ≥ 0.95 AND P1/P2 passed.
        Update via EMA (α=0.15) → prevents mediocre restorations from degrading reference.
        """
        if not (hpi > 0.5 and artifact_freedom >= 0.95 and p1_p2_passed):
            return

        embedding = self._compute_embedding(restored, sr)
        key = (genre, material, era_bin)

        with self._ref_lock:
            if key in self._ref_memory:
                entry = self._ref_memory[key]
                # §2.44 EMA: α=0.15 → new_ref = 0.85 * old + 0.15 * new_embedding
                entry.embedding = (1.0 - _EMA_ALPHA) * entry.embedding + _EMA_ALPHA * embedding
                entry.obs_count += 1
                entry.calibrated = entry.obs_count >= _MIN_OBS_CALIBRATED
            else:
                self._ref_memory[key] = _RefEntry(
                    embedding=embedding.copy(),
                    obs_count=1,
                    calibrated=False,
                )

        logger.info(
            "§2.44 ReferenceMemory updated key=%s obs=%d calibrated=%s",
            key,
            self._ref_memory[key].obs_count,
            self._ref_memory[key].calibrated,
        )

    def _get_reference_vector(self, genre: str, material: str, era_bin: str) -> np.ndarray | None:
        """§2.44 Fallback-Kaskade (5 Stufen).

        Stufe 1: Gleiche Genre-Familie + nächstliegende Ära → GP-Memory
        Stufe 2: Gleiche Ära + nächstliegendes Genre → GP-Memory
        Stufe 3: Bootstrap-Prototyp für Genre-Cluster (material-agnostic)
        Stufe 4: Genre-agnostischer Ära-Median
        Stufe 5: Kein Ref-Vektor → None → rein gegen Input
        """
        # Stufe 1: Exact match
        key = (genre, material, era_bin)
        entry = self._ref_memory.get(key)
        if entry is not None:
            return entry.embedding

        # Stufe 2: Same era, any material (nächstliegendes Genre)
        era_entries = [self._ref_memory[k] for k in self._ref_memory if k[2] == era_bin and k[0] == genre]
        if era_entries:
            embeddings = np.stack([e.embedding for e in era_entries])
            return np.mean(embeddings, axis=0)

        # Stufe 3: Same genre, any material, any era
        genre_entries = [self._ref_memory[k] for k in self._ref_memory if k[0] == genre]
        if genre_entries:
            embeddings = np.stack([e.embedding for e in genre_entries])
            return np.mean(embeddings, axis=0)

        # Stufe 4: Genre-agnostischer Ära-Median
        all_era = [self._ref_memory[k] for k in self._ref_memory if k[2] == era_bin]
        if all_era:
            embeddings = np.stack([e.embedding for e in all_era])
            return np.mean(embeddings, axis=0)

        # Stufe 5: Kein Referenz-Vektor
        return None

    def _compute_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute spectral embedding (mel-energy vector) as MERT-proxy."""
        mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)
        n_samples = len(mono)
        if n_samples < 2048:
            return np.ones(40, dtype=np.float32)

        n_fft = 2048
        hop = 512
        n_mels = 40
        n_frames = min(200, max(1, (n_samples - n_fft) // hop))
        win = np.hanning(n_fft).astype(np.float32)

        # Mel filterbank
        mel_freqs = np.linspace(0, 2595 * np.log10(1 + (sr / 2.0) / 700.0), n_mels + 2)
        hz_freqs = 700.0 * (10.0 ** (mel_freqs / 2595.0) - 1.0)
        bin_freqs = np.clip(np.floor((n_fft + 1) * hz_freqs / sr).astype(int), 0, n_fft // 2)
        filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for m in range(n_mels):
            f_s, f_c, f_e = bin_freqs[m], bin_freqs[m + 1], bin_freqs[m + 2]
            if f_c > f_s:
                filterbank[m, f_s:f_c] = np.linspace(0, 1, f_c - f_s)
            if f_e > f_c:
                filterbank[m, f_c:f_e] = np.linspace(1, 0, f_e - f_c)

        mel_frames = []
        for i in range(n_frames):
            s = i * hop
            e = s + n_fft
            if e > n_samples:
                break
            spec = np.abs(np.fft.rfft(mono[s:e] * win)) ** 2
            mel_frames.append(filterbank @ spec)

        if not mel_frames:
            return np.ones(n_mels, dtype=np.float32)

        embedding = np.log1p(np.mean(mel_frames, axis=0)).astype(np.float32)
        norm = float(np.linalg.norm(embedding) + 1e-12)
        return embedding / norm

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embedding vectors."""
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
        dot = float(np.dot(a, b))
        norm = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        return dot / norm

    def evaluate_studio(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
        pqs_improvement: float = 0.0,
        artifact_freedom: float = 1.0,
        emotional_arc_score: float = 1.0,
    ) -> HPIResult:
        """Evaluate HPI for Studio 2026 mode.

        HPI = studio_quality_gain × PQS_improvement × artifact_freedom × emotional_arc_preservation
        """
        studio_gain = self._compute_studio_quality_gain(original, restored, sr)
        # §2.44 [FIX] pqs_improvement als Vorzeichenträger — kein positives Clipping (max 0.0
        # entfernt). Negatives pqs_improvement → HPI < 0 → Rollback (§2.44: HPI ≤ 0 → Rollback).
        # Normierung: pqs_improvement ∈ [-1, 1] bleibt erhalten; Werte außerhalb werden geclippt.
        pqs_signed = float(np.clip(pqs_improvement, -1.0, 1.0))

        hpi = studio_gain * pqs_signed * artifact_freedom * emotional_arc_score

        passed = hpi > 0.0 and artifact_freedom >= 0.95

        logger.info(
            "§2.44 HPI(Studio2026)=%.4f passed=%s (studio_gain=%.3f pqs_signed=%.3f artifact=%.3f emotional=%.3f)",
            hpi,
            passed,
            studio_gain,
            pqs_signed,
            artifact_freedom,
            emotional_arc_score,
        )

        return HPIResult(
            hpi=round(hpi, 4),
            passed=passed,
            studio_quality_gain=round(studio_gain, 4),
            pqs_improvement=round(pqs_improvement, 4),
            artifact_freedom=round(artifact_freedom, 4),
            emotional_arc_preservation=round(emotional_arc_score, 4),
            is_studio_mode=True,
        )

    # ── Component computations ─────────────────────────────────────────────

    def _compute_mert_similarity(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
    ) -> float:
        """Compute MERT-based musical similarity (melody, harmony, rhythm).

        Primary path: MERT plugin analysis similarity.
        Fallback path: spectral correlation proxy (artifact-safe).
        """
        orig_clean = np.nan_to_num(original.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        rest_clean = np.nan_to_num(restored.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # Keep legacy short-signal behavior deterministic for tests and edge-cases.
        orig_mono = orig_clean if orig_clean.ndim == 1 else np.mean(orig_clean, axis=0)
        rest_mono = rest_clean if rest_clean.ndim == 1 else np.mean(rest_clean, axis=0)
        if min(len(orig_mono), len(rest_mono)) < 1024:
            return 1.0

        # MERT-Plugin-First path (optional, failure-safe)
        if not self._mert_path_disabled:
            try:
                # Imported lazily to avoid mandatory ML initialization on module import.
                from plugins.mert_plugin import get_loaded_mert_plugin, get_mert_plugin

                plugin = get_loaded_mert_plugin()
                if plugin is None:
                    plugin = get_mert_plugin()

                a1 = plugin.analyze(orig_clean, sr)
                a2 = plugin.analyze(rest_clean, sr)

                # Normalize F0 proximity to [0,1] via log2 octave distance.
                f0_1 = float(max(0.0, getattr(a1, "estimated_f0_hz", 0.0)))
                f0_2 = float(max(0.0, getattr(a2, "estimated_f0_hz", 0.0)))
                if f0_1 > 0.0 and f0_2 > 0.0:
                    oct_dist = abs(np.log2((f0_1 + 1e-9) / (f0_2 + 1e-9)))
                    f0_sim = float(np.exp(-oct_dist / 0.5))
                else:
                    f0_sim = 1.0

                h1 = float(np.clip(getattr(a1, "harmonicity", 0.0), 0.0, 1.0))
                h2 = float(np.clip(getattr(a2, "harmonicity", 0.0), 0.0, 1.0))
                t1 = float(np.clip(getattr(a1, "tonal_consistency", 0.0), 0.0, 1.0))
                t2 = float(np.clip(getattr(a2, "tonal_consistency", 0.0), 0.0, 1.0))
                f1 = float(np.clip(getattr(a1, "spectral_flux_coherence", 0.0), 0.0, 1.0))
                f2 = float(np.clip(getattr(a2, "spectral_flux_coherence", 0.0), 0.0, 1.0))

                harm_sim = 1.0 - abs(h1 - h2)
                tonal_sim = 1.0 - abs(t1 - t2)
                flux_sim = 1.0 - abs(f1 - f2)

                plugin_sim = 0.35 * harm_sim + 0.35 * tonal_sim + 0.20 * flux_sim + 0.10 * f0_sim
                # §2.44 Blend: 65% Plugin-Score + 35% Spektral-Proxy.
                # min() war zu konservativ und zog valide Ergebnisse systematisch nach unten
                # (proxy ~0.7 bei Breitband-Änderungen → false rollback auch bei plugin=0.95).
                proxy_sim = self._compute_mert_similarity_spectral_proxy(orig_clean, rest_clean, sr)
                sim = 0.65 * float(plugin_sim) + 0.35 * float(proxy_sim)
                return float(np.clip(sim, 0.0, 1.0))
            except Exception as exc:
                logger.debug("§2.44 MERT similarity fallback to spectral proxy: %s", exc)
                # Disable repeated failing plugin initialization attempts in this process.
                self._mert_path_disabled = True

        # Failure-safe spectral proxy fallback.
        return self._compute_mert_similarity_spectral_proxy(orig_clean, rest_clean, sr)

    def _compute_mert_similarity_spectral_proxy(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
    ) -> float:
        """Spectral proxy for musical similarity when MERT plugin is unavailable."""
        orig_mono = original if original.ndim == 1 else np.mean(original, axis=0)
        rest_mono = restored if restored.ndim == 1 else np.mean(restored, axis=0)
        min_len = min(len(orig_mono), len(rest_mono))
        if min_len < 1024:
            return 1.0

        orig_mono = orig_mono[:min_len]
        rest_mono = rest_mono[:min_len]

        # Multi-scale spectral correlation
        n_fft = min(2048, min_len)
        hop = n_fft // 4
        n_frames = max(1, (min_len - n_fft) // hop)
        n_frames = min(n_frames, 100)

        correlations = []
        win = np.hanning(n_fft).astype(np.float32)

        for i in range(n_frames):
            s = i * hop
            e = s + n_fft
            if e > min_len:
                break

            orig_spec = np.abs(np.fft.rfft(orig_mono[s:e] * win))
            rest_spec = np.abs(np.fft.rfft(rest_mono[s:e] * win))

            # Log-magnitude correlation (perceptually meaningful)
            orig_log = np.log1p(orig_spec)
            rest_log = np.log1p(rest_spec)

            orig_norm = orig_log - np.mean(orig_log)
            rest_norm = rest_log - np.mean(rest_log)

            denom = float(np.sqrt(np.sum(orig_norm**2) * np.sum(rest_norm**2)) + 1e-12)
            if denom > 1e-12:
                corr = float(np.sum(orig_norm * rest_norm) / denom)
                correlations.append(max(0.0, corr))

        if not correlations:
            return 1.0

        return float(np.clip(np.mean(correlations), 0.0, 1.0))

    def _compute_timbral_fidelity(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
    ) -> float:
        """Structural timbral coherence — not mere input similarity.

        Uses mel-embedding cosine similarity (timbral perceptual features).
        Delegates to _compute_embedding for shared mel-filterbank computation.
        """
        min_len = min(
            len(original) if original.ndim == 1 else original.shape[-1],
            len(restored) if restored.ndim == 1 else restored.shape[-1],
        )
        if min_len < 2048:
            return 1.0
        orig_embed = self._compute_embedding(original, sr)
        rest_embed = self._compute_embedding(restored, sr)
        return float(np.clip(self._cosine_similarity(orig_embed, rest_embed), 0.0, 1.0))

    def _compute_studio_quality_gain(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
    ) -> float:
        """Studio 2026: improvement in studio quality relative to input.

        Compares how much closer the *restored* signal is to the studio reference
        (−14 LUFS, noise ≤ −72 dBFS) compared to the *original* input.
        A restored signal that is closer → gain > 0.5; same or worse → gain ≤ 0.5.
        Always returns ≥ 0.1 to avoid killing HPI when improvement is ambiguous.
        """

        def _score(audio: np.ndarray) -> float:
            mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)
            if len(mono) < 1024:
                return 0.5
            rms = float(np.sqrt(np.mean(mono**2) + 1e-12))
            lufs_approx = 20.0 * np.log10(rms + 1e-12)
            lufs_error = abs(lufs_approx - (-14.0))
            lufs_score = max(0.0, 1.0 - lufs_error / 30.0)

            frame_len = int(0.03 * sr)
            hop = frame_len // 2
            n_frames = max(1, (len(mono) - frame_len) // hop)
            energies = [
                float(np.mean(mono[i * hop : i * hop + frame_len] ** 2) + 1e-12) for i in range(min(n_frames, 500))
            ]
            noise_floor = 10.0 * np.log10(np.percentile(energies, 5)) if energies else -72.0
            noise_score = 1.0 if noise_floor <= -72.0 else max(0.0, 1.0 - (noise_floor + 72.0) / 30.0)
            return float((lufs_score + noise_score) / 2.0)

        in_score = _score(original)
        out_score = _score(restored)

        # Improvement ratio mapped to [0.1, 1.0].
        # out/in > 1 → improved → gain → 1.0; equal → 0.5; worse → down to 0.1.
        ratio = out_score / max(in_score, 1e-4)
        gain = float(np.clip(0.5 * ratio, 0.1, 1.0))
        return gain
