"""
Aurik 9 — HolisticPerceptualGate §2.44 [RELEASE_MUST]
======================================================
Last gate before export. Measures holistic perceptual improvement
instead of checking individual goals only.

HPI > 0 → Export | HPI ≤ 0 → Rollback

§2.44 FIX v9.11.2 — Referenz-Paradox beseitigt:
  Für ALLE Restorability-Bereiche: Referenz-Vektor aus GP-Memory primär;
  kein Ref-Vektor → direktionale Verbesserungsmessung (_compute_directional_restoration_quality).
  Input-Ähnlichkeit als prim. Maß entfernt (bestrafte erfolgreiche Restaurierung).

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
    fail_reason: object | None = None  # §1.4a FailReason when passed=False


class HolisticPerceptualGate:
    """§2.44 Holistic Perceptual Gate — last gate before export."""

    def __init__(self) -> None:
        # §2.44 MERT-Reference-Memory: key = (genre, material, era_bin)
        self._ref_memory: dict[tuple[str, str, str], _RefEntry] = {}
        self._ref_lock = threading.Lock()
        # MERT similarity path is optional and must never break the gate.
        self._mert_path_disabled: bool = False
        # §2.44 VERBOTEN: MERT darf nicht primary sein wenn VERSA verfügbar.
        # _mert_proxy_used = True → VERSA fehlgeschlagen, MERT als Fallback aktiv.
        self._mert_proxy_used: bool = False

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

        §2.44 FIX v9.11.2 — Referenz-Paradox beseitigt:
          Strukturelle Klangkohärenz bedeutet NICHT Ähnlichkeit zum degradierten Input.
          Ein erfolgreich restauriertes Signal weicht vom degradierten Input ab —
          der alte Ansatz (input_weight=1.0 bei restorability > 70) hat gute
          Restaurierung aktiv bestraft.

          Korrekte Strategie für alle Restorability-Bereiche:
            1. Referenz-Vektor aus GP-Memory (genre × material × era_bin) → primär
            2. Kein Referenz-Vektor → direktionale Verbesserungsmessung:
               misst ob Signal in Richtung "sauber + musikalisch" verbessert wurde
          Input-Ähnlichkeit dient nur als Content-Integrity-Anteil (klein).
        """
        self._mert_proxy_used = False  # reset per evaluation
        mert_sim = self._compute_mert_similarity(original, restored, sr)

        # §2.44 FIX v9.11.2: Referenz-Vektor bevorzugen für ALLE Restorability-Bereiche.
        # Kein Ref-Vektor → direktionale Qualitätsmessung statt Input-Ähnlichkeit.
        ref_vec = self._get_reference_vector(genre, material, era_bin)
        if ref_vec is not None:
            rest_embed = self._compute_embedding(restored, sr)
            timbral_ref = float(np.clip(self._cosine_similarity(ref_vec, rest_embed), 0.0, 1.0))
        else:
            # Kein Referenz-Vektor: misst ob das Signal in Richtung "sauber" verbessert wurde
            timbral_ref = self._compute_directional_restoration_quality(original, restored, sr)

        # timbral_input als Content-Integrity-Anteil (für Logging und niedrige Restorability)
        timbral_input = self._compute_timbral_fidelity(original, restored, sr)

        # §2.44 Restorability-dependent weights — Referenz/Direktional dominiert stets
        if restorability_score > 70.0:
            # Hohe Restorability: Signal bewegt sich weg vom Defekt, hin zur Referenz.
            # Input-Ähnlichkeit ist hier KEIN Qualitätsmaß (Referenz-Paradox).
            input_weight, ref_weight = 0.0, 1.0
        elif restorability_score >= 50.0:
            # Mittlere Restorability: minimaler Input-Anteil als Ankerpunkt
            input_weight, ref_weight = 0.35, 0.65
        else:
            # Niedrige Restorability: kleiner Content-Integrity-Anteil
            input_weight, ref_weight = 0.2, 0.8

        timbral = input_weight * timbral_input + ref_weight * timbral_ref

        hpi = mert_sim * timbral * artifact_freedom * emotional_arc_score

        # §B3 NORESQA integration: Non-intrusive MOS proxy modulates HPI weakly
        # (Manocha & Kumar 2022, INTERSPEECH). Acts as a soft quality sanity check.
        # Weight 0.15 keeps it advisory: noresqa_ensemble ∈ [0.85, 1.00]
        noresqa_score = self._compute_noresqa_score(restored, sr)
        noresqa_ensemble = 0.85 + 0.15 * noresqa_score
        hpi = hpi * noresqa_ensemble

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

        # §1.4a FailReason for failed gate
        _fr = None
        if not passed:
            from backend.core.pipeline_health_state import make_fail_reason

            if artifact_freedom < 0.95:
                _fr = make_fail_reason(
                    "HolisticPerceptualGate",
                    "ARTIFACT_VETO",
                    severity="failed",
                    action="rollback",
                    details=f"artifact_freedom={artifact_freedom:.3f} < 0.95",
                )
            else:
                _fr = make_fail_reason(
                    "HolisticPerceptualGate",
                    "HPI_BELOW_ZERO",
                    severity="failed",
                    action="rollback",
                    details=f"HPI={hpi:.4f} <= 0",
                )

        # §2.44-lit PEAQ/MUSHRA-inspired additive diagnostic (ISO 16832 + ITU-R BS.1387 + BS.1534).
        # MUSHRA and PEAQ use weighted linear combination of quality factors.
        # The product formula here treats each factor as independently necessary
        # (Lagrange-multiplier semantics: zero in any dimension = complete failure).
        # The additive alternative is computed ONLY for comparative diagnostics —
        # if product HPI fails while PEAQ-additive passes, a single factor collapse may
        # indicate a false rollback worth inspecting in logs.
        _peaq_additive = float(np.clip(0.40 * mert_sim + 0.35 * timbral + 0.25 * float(emotional_arc_score), 0.0, 1.0))
        _peaq_hpi_val = float(np.clip(_peaq_additive * artifact_freedom, 0.0, 1.0))
        if not passed and _peaq_hpi_val > 0.30 and artifact_freedom >= 0.95:
            logger.warning(
                "§2.44 HPI-Diskrepanz (PEAQ-Lit-Vergleich): product=%.4f FAIL aber PEAQ-additiv=%.4f PASS "
                "(mert=%.3f timbral=%.3f emotional=%.3f) — Single-Factor-Kollaps prüfen "
                "[ISO 16832 / ITU-R BS.1387]",
                hpi,
                _peaq_hpi_val,
                mert_sim,
                timbral,
                float(emotional_arc_score),
            )

        return HPIResult(
            hpi=round(hpi, 4),
            passed=passed,
            mert_similarity=round(mert_sim, 4),
            timbral_fidelity=round(timbral, 4),
            artifact_freedom=round(artifact_freedom, 4),
            emotional_arc_preservation=round(emotional_arc_score, 4),
            is_studio_mode=False,
            fail_reason=_fr,
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
                "mert_proxy_used": self._mert_proxy_used,
                "noresqa_score": round(noresqa_score, 4),
                "noresqa_ensemble": round(noresqa_ensemble, 4),
                # §2.44-lit: PEAQ/MUSHRA additive metric for comparative diagnostics
                "peaq_additive_hpi": round(_peaq_hpi_val, 4),
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

        # §1.4a FailReason for failed Studio gate
        _fr_s = None
        if not passed:
            from backend.core.pipeline_health_state import make_fail_reason

            if artifact_freedom < 0.95:
                _fr_s = make_fail_reason(
                    "HolisticPerceptualGate",
                    "ARTIFACT_VETO",
                    severity="failed",
                    action="rollback",
                    details=f"artifact_freedom={artifact_freedom:.3f} < 0.95",
                )
            else:
                _fr_s = make_fail_reason(
                    "HolisticPerceptualGate",
                    "HPI_BELOW_ZERO",
                    severity="failed",
                    action="rollback",
                    details=f"Studio HPI={hpi:.4f} <= 0",
                )

        return HPIResult(
            hpi=round(hpi, 4),
            passed=passed,
            studio_quality_gain=round(studio_gain, 4),
            pqs_improvement=round(pqs_improvement, 4),
            artifact_freedom=round(artifact_freedom, 4),
            emotional_arc_preservation=round(emotional_arc_score, 4),
            is_studio_mode=True,
            fail_reason=_fr_s,
        )

    # ── Component computations ─────────────────────────────────────────────

    def _compute_mert_similarity(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
    ) -> float:
        """Compute musical quality coefficient for HPI.

        §2.44 VERBOTEN: MERT darf NICHT primary sein wenn VERSA verfügbar.
        Primary path: VERSA MOS auf restoreriertem Audio (referenzfrei, kein Referenz-Paradoxon).
        Fallback path 1: MERT plugin similarity (proxy, setzt self._mert_proxy_used=True).
        Fallback path 2: spectral correlation proxy (artifact-safe).
        """
        orig_clean = np.nan_to_num(original.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        rest_clean = np.nan_to_num(restored.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # Keep legacy short-signal behavior deterministic for tests and edge-cases.
        orig_mono = orig_clean if orig_clean.ndim == 1 else np.mean(orig_clean, axis=0)
        rest_mono = rest_clean if rest_clean.ndim == 1 else np.mean(rest_clean, axis=0)
        if min(len(orig_mono), len(rest_mono)) < 1024:
            return 1.0

        # ─── PRIMARY PATH: VERSA MOS ───────────────────────────────────────────
        # §2.44 VERBOTEN: MERT darf nicht primary sein wenn VERSA verfügbar.
        # VERSA MOS (1–5) → normalisiert [0,1] via (mos-1)/4.
        # Referenzfrei → kein Referenz-Paradoxon, kein Input-Similarity-Bias.
        try:
            from plugins.versa_plugin import get_versa_plugin as _get_versa

            _versa = _get_versa()
            _versa_result = _versa.score(rest_clean, sr)
            _versa_mos = float(np.clip(_versa_result.mos, 1.0, 5.0))
            # Nonnalisierung: MOS 1→0.0, MOS 3→0.5, MOS 5→1.0
            _versa_sim = float(np.clip((_versa_mos - 1.0) / 4.0, 0.0, 1.0))
            # Blend: 70% VERSA (referenzfrei) + 30% klassische spektrale Kohärenz
            _spectral_coh = self._compute_mert_similarity_spectral_proxy(orig_clean, rest_clean, sr)
            _blended = float(np.clip(0.70 * _versa_sim + 0.30 * _spectral_coh, 0.0, 1.0))
            logger.debug(
                "§2.44 VERSA-primary: mos=%.2f → versa_sim=%.3f spectral_coh=%.3f blended=%.3f",
                _versa_mos,
                _versa_sim,
                _spectral_coh,
                _blended,
            )
            self._mert_proxy_used = False  # VERSA succeeded
            return _blended
        except Exception as _versa_exc:
            logger.debug("§2.44 VERSA primary failed → MERT proxy fallback: %s", _versa_exc)
            self._mert_proxy_used = True  # VERSA failed, MERT is proxy

        # ─── FALLBACK PATH 1: MERT plugin ─────────────────────────────────────
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

        if correlations:
            return float(np.clip(float(np.mean(correlations)), 0.0, 1.0))
        return 0.8

    def _compute_timbral_fidelity(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
    ) -> float:
        """Content-integrity check via mel-embedding cosine similarity.

        Used as small content-preservation anchor in evaluate_restoration().
        NOT used as primary timbral_fidelity measure (see §2.44 FIX v9.11.2).
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

    def _compute_directional_restoration_quality(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
    ) -> float:
        """§2.44 FIX v9.11.2 — Direktionale Verbesserungsmessung als Fallback.

        Misst ob die Restaurierung das Signal in Richtung "sauber und musikalisch"
        verbessert hat. Wird verwendet wenn kein Referenz-Vektor im GP-Memory vorliegt.

        Drei Komponenten:
          A) Noise-Floor-Delta: tieferer Rauschboden nach Restaurierung → Wert steigt
          B) Spektrale Klarheit (HF-Crest): höhere Klarheit nach Denoising → Wert steigt
          C) Content-Integrity-Guard: verhindert, dass zerstörter Inhalt besteht

        Returns:
            0.5 = keine Veränderung (Bypass)
            > 0.5 = Signal wurde verbessert (Rauschen reduziert, Klarheit erhöht)
            < 0.5 = Signal wurde verschlechtert
        """
        orig_mono = (original if original.ndim == 1 else np.mean(original, axis=0)).astype(np.float32)
        rest_mono = (restored if restored.ndim == 1 else np.mean(restored, axis=0)).astype(np.float32)
        min_len = min(len(orig_mono), len(rest_mono))
        if min_len < 1024:
            return 0.75  # short signal: neutral-good

        orig_mono = np.nan_to_num(orig_mono[:min_len], nan=0.0)
        rest_mono = np.nan_to_num(rest_mono[:min_len], nan=0.0)

        # C) Content-Integrity-Guard: spectral correlation (log-magnitude)
        n_fft_c = min(4096, min_len)
        orig_spec = np.abs(np.fft.rfft(orig_mono[:n_fft_c] * np.hanning(n_fft_c)))
        rest_spec = np.abs(np.fft.rfft(rest_mono[:n_fft_c] * np.hanning(n_fft_c)))
        orig_log = np.log1p(orig_spec)
        rest_log = np.log1p(rest_spec)
        orig_n = orig_log - np.mean(orig_log)
        rest_n = rest_log - np.mean(rest_log)
        denom_c = float(np.sqrt(np.sum(orig_n**2) * np.sum(rest_n**2)) + 1e-12)
        content_corr = float(np.sum(orig_n * rest_n) / denom_c) if denom_c > 1e-12 else 0.0
        if content_corr < 0.3:
            # Musikalischer Inhalt schwer verändert → frühere Rückkehr
            return float(np.clip(0.3 + 0.2 * max(0.0, content_corr), 0.0, 1.0))

        # A) Noise-Floor-Delta (5. Perzentil der Frame-Energien)
        frame_len = max(1, int(0.03 * sr))
        hop = max(1, frame_len // 2)
        n_frames = min(200, max(1, (min_len - frame_len) // hop))
        orig_e: list[float] = []
        rest_e: list[float] = []
        for i in range(n_frames):
            s = i * hop
            e = s + frame_len
            if e > min_len:
                break
            orig_e.append(float(np.mean(orig_mono[s:e] ** 2) + 1e-12))
            rest_e.append(float(np.mean(rest_mono[s:e] ** 2) + 1e-12))
        if orig_e and rest_e:
            orig_nf_db = 10.0 * float(np.log10(float(np.percentile(orig_e, 5))))
            rest_nf_db = 10.0 * float(np.log10(float(np.percentile(rest_e, 5))))
            noise_delta_db = orig_nf_db - rest_nf_db  # > 0 wenn Rauschen reduziert
        else:
            noise_delta_db = 0.0
        noise_score = float(np.clip(0.5 + noise_delta_db / 40.0, 0.0, 1.0))

        # B) Spektrale Klarheit (HF Crest-Factor 2–16 kHz)
        freqs = np.fft.rfftfreq(n_fft_c, d=1.0 / sr)
        orig_fft_full = np.abs(np.fft.rfft(orig_mono[:n_fft_c] * np.hanning(n_fft_c)))
        rest_fft_full = np.abs(np.fft.rfft(rest_mono[:n_fft_c] * np.hanning(n_fft_c)))
        hf_mask = (freqs >= 2000) & (freqs <= 16000)
        hf_bins_o = orig_fft_full[hf_mask]
        hf_bins_r = rest_fft_full[hf_mask]
        if len(hf_bins_o) >= 10 and len(hf_bins_r) >= 10:
            crest_o = float(np.percentile(hf_bins_o, 95)) / (float(np.median(hf_bins_o)) + 1e-9)
            crest_r = float(np.percentile(hf_bins_r, 95)) / (float(np.median(hf_bins_r)) + 1e-9)
            max_crest = max(crest_o, crest_r, 1e-9)
            crest_improvement = (crest_r - crest_o) / max_crest  # [-1, 1]
        else:
            crest_improvement = 0.0
        clarity_score = float(np.clip(0.5 + crest_improvement * 0.5, 0.0, 1.0))

        return float(np.clip(0.6 * noise_score + 0.4 * clarity_score, 0.0, 1.0))

    def _compute_noresqa_score(self, audio: np.ndarray, sr: int) -> float:
        """§B3 NORESQA: Non-intrusive quality estimation (Manocha & Kumar, INTERSPEECH 2022).

        Attempts to use NoresqaPlugin if available; falls back to a DSP proxy that
        combines spectral flatness, SNR estimate, and harmonic coherence — all
        reference-free indicators of audio quality aligned with MOS correlates.

        Returns a score in [0, 1] (1.0 = highest quality). Non-blocking.
        """
        try:
            from plugins.noresqa_plugin import get_noresqa_plugin  # type: ignore

            _plg = get_noresqa_plugin()
            if _plg is not None:
                mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)
                score = float(_plg.score(mono.astype(np.float32), sr))
                return float(np.clip(score, 0.0, 1.0))
        except Exception:  # plugin not installed → DSP fallback
            pass

        # DSP proxy: three reference-free quality correlates
        try:
            mono = np.asarray(audio if audio.ndim == 1 else np.mean(audio, axis=0), dtype=np.float32)
            mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
            if len(mono) < 1024:
                return 1.0

            n_fft = min(4096, len(mono))
            win = np.hanning(n_fft)
            spec = np.abs(np.fft.rfft(mono[:n_fft] * win)) + 1e-12

            # a) Spectral Flatness (Wiener entropy) — lower = more tonal = higher quality
            geo_mean = float(np.exp(np.mean(np.log(spec))))
            arith_mean = float(np.mean(spec))
            sfm = float(np.clip(geo_mean / (arith_mean + 1e-12), 0.0, 1.0))
            # SFM near 0 = very tonal (music), near 1 = noise-like
            # Quality proxy: music should be 0.05–0.40 → map via gaussIan around 0.15
            sfm_score = float(np.clip(np.exp(-((sfm - 0.15) ** 2) / 0.08), 0.0, 1.0))

            # b) Noise Floor Estimate via 5th percentile (low = cleaner signal)
            frame_len = max(512, n_fft // 8)
            hop = frame_len // 2
            n_frames = max(1, (len(mono) - frame_len) // hop)
            frame_rmss = [
                float(np.sqrt(np.mean(mono[i * hop : i * hop + frame_len] ** 2) + 1e-12))
                for i in range(min(n_frames, 200))
            ]
            if frame_rmss:
                noise_floor_db = 20.0 * float(np.log10(float(np.percentile(frame_rmss, 5)) + 1e-12))
            else:
                noise_floor_db = -60.0
            # Map [-80, -20] dBFS → [1, 0]
            snr_score = float(np.clip((-noise_floor_db - 20.0) / 60.0, 0.0, 1.0))

            # c) Harmonic coherence: autocorrelation peak ratio at fundamental period
            # Use 50 ms window in the most energetic segment
            win_len = int(0.05 * sr)
            if win_len >= 64 and len(mono) >= win_len:
                # Find most energetic 50 ms segment
                n_seg = max(1, (len(mono) - win_len) // (win_len // 2))
                energies_seg = [
                    float(np.mean(mono[i * (win_len // 2) : i * (win_len // 2) + win_len] ** 2))
                    for i in range(min(n_seg, 100))
                ]
                best_seg = int(np.argmax(energies_seg)) * (win_len // 2)
                segment = mono[best_seg : best_seg + win_len]
                from backend.core.core_utils import fft_autocorr

                ac = fft_autocorr(segment)
                ac = ac / (ac[0] + 1e-12)
                # Look for AC peak in F0 range 80–800 Hz → lags [sr//800, sr//80]
                lag_min = max(1, int(sr / 800))
                lag_max = min(len(ac) - 1, int(sr / 80))
                if lag_max > lag_min:
                    peak_lag = int(np.argmax(ac[lag_min : lag_max + 1])) + lag_min
                    harmonic_coh = float(np.clip(ac[peak_lag], 0.0, 1.0))
                else:
                    harmonic_coh = 0.5
            else:
                harmonic_coh = 0.5

            # Weighted combo (balanced: SFM captures tonal structure, SNR cleanness, HC harmonicity)
            proxy = 0.35 * sfm_score + 0.40 * snr_score + 0.25 * harmonic_coh
            return float(np.clip(proxy, 0.0, 1.0))
        except Exception as _exc:
            logger.debug("NORESQA DSP-proxy error (non-blocking): %s", _exc)
            return 1.0  # neutral: don't penalise when guard fails

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
