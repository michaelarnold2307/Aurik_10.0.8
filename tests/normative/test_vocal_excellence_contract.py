"""[RELEASE_MUST] Normativer Gesangsexzellenz-Contract.

Dieser CI-Test bündelt die wichtigsten Gesangs-Invarianten aus Spec 01/§2.35d,
§2.35c, §2.36 und §0a in einem kompakten Guard:

1. VQI-Schwellen und Gewichtung bleiben stabil.
2. Singer-Identity-Rollback-Schutz bleibt dokumentiert.
3. Die Vocal-Chain-Reihenfolge bleibt kanonisch.
4. Restoration trennt Stem-Enhancement strikt von phonem-bewusster Steuerung.

Der Test prüft bewusst Code- und Spec-Artefakte gemeinsam, um Drift zwischen
normativer Aussage und tatsächlicher Pipeline-Verdrahtung früh zu erkennen.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC_01 = _ROOT / ".github" / "specs" / "01_musical_goals.md"
_VQI_CODE = _ROOT / "backend" / "core" / "musical_goals" / "vocal_quality_index.py"
_UV3 = _ROOT / "backend" / "core" / "unified_restorer_v3.py"
_MUSICAL_GOALS_INSTR = _ROOT / ".github" / "instructions" / "musical_goals.instructions.md"
_SECTION_0A_GUARD = _ROOT / "tests" / "normative" / "test_section_0a_restoration_guard.py"
_PANNS = _ROOT / "plugins" / "panns_plugin.py"


@pytest.mark.normative
@pytest.mark.timeout(10)
class TestVocalExcellenceSpec:
    def test_spec_declares_vocal_excellence_contract(self) -> None:
        content = _SPEC_01.read_text(encoding="utf-8")

        assert "§2.35d [RELEASE_MUST] Gesangsexzellenz-Contract" in content
        assert (
            "phase_19_de_esser -> phase_42_vocal_enhancement -> phase_43_ml_deesser -> phase_58_lyrics_guided_enhancement"
            in content
        )
        assert "phase_42_vocal_enhancement` bleibt in `restoration` verboten" in content
        assert "phase_58_lyrics_guided_enhancement` bleibt bei erkannter Stimme dennoch Pflicht" in content


@pytest.mark.normative
@pytest.mark.timeout(10)
class TestVocalExcellenceThresholds:
    def test_vqi_constants_remain_at_documented_contract_values(self) -> None:
        content = _VQI_CODE.read_text(encoding="utf-8")

        assert "VQI_WORLD_CLASS = 0.88" in content
        assert "VQI_PROFESSIONAL = 0.82" in content
        assert "VQI_THRESHOLD = 0.72" in content

    def test_vqi_weights_still_sum_to_one(self) -> None:
        from backend.core.musical_goals import vocal_quality_index as vqi

        total = vqi._W_SINGER_ID + vqi._W_FORMANT + vqi._W_ARTICULATION + vqi._W_PROXIMITY + vqi._W_SIBILANCE
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_singer_identity_rollback_threshold_remains_documented(self) -> None:
        content = _MUSICAL_GOALS_INSTR.read_text(encoding="utf-8")

        assert "cos_sim < 0.92 → Rollback letzter Vokal-Phase" in content

    def test_vocal_metric_activation_thresholds_remain_conservative(self) -> None:
        panns = _PANNS.read_text(encoding="utf-8")
        spec = _SPEC_01.read_text(encoding="utf-8")

        assert "threshold ≥ 0.40" in panns
        assert "panns_singing_confidence >= 0.35" in spec


@pytest.mark.normative
@pytest.mark.timeout(10)
class TestVocalExcellencePipelineWiring:
    def test_uv3_passes_song_structure_vocal_segments_into_vqi(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert "for seg in (self._ssa_segments or [])" in content
        assert 'if bool(getattr(seg, "has_vocals", False)) and float(seg.end_s) > float(seg.start_s)' in content
        assert "vocal_segments=_vqi_segments or None" in content

    def test_uv3_accumulates_vqi_fields_before_final_export(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert 'self._phase_metadata_accumulator["vqi"] = _vqi_score' in content
        assert 'self._phase_metadata_accumulator["singer_identity_cosine"] = float(' in content
        assert 'self._phase_metadata_accumulator["singer_id_dsp_fallback"] = bool(' in content
        assert 'self._phase_metadata_accumulator["vqi_tier"] = _vqi_result.get("vqi_tier", "unknown")' in content

    def test_uv3_exports_final_vocal_metadata_fields(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert '"vqi": (self._phase_metadata_accumulator or {}).get("vqi")' in content
        assert (
            '"singer_identity_cosine": (self._phase_metadata_accumulator or {}).get("singer_identity_cosine")'
            in content
        )
        assert (
            '"singer_id_dsp_fallback": (self._phase_metadata_accumulator or {}).get("singer_id_dsp_fallback")'
            in content
        )
        assert '"vqi_tier": (self._phase_metadata_accumulator or {}).get("vqi_tier")' in content

    def test_uv3_keeps_canonical_vocal_chain_order(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert '_move_before("phase_19_de_esser", "phase_42_vocal_enhancement")' in content
        assert '_move_before("phase_19_de_esser", "phase_43_ml_deesser")' in content
        assert '_move_before("phase_42_vocal_enhancement", "phase_58_lyrics_guided_enhancement")' in content
        assert '_move_before("phase_43_ml_deesser", "phase_58_lyrics_guided_enhancement")' in content

    def test_uv3_keeps_lyrics_guided_enhancement_for_detected_vocals(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert "if vocals_detected:" in content
        assert 'selected.append("phase_58_lyrics_guided_enhancement")' in content

    def test_restoration_guard_keeps_mode_split_between_phase_42_and_phase_58(self) -> None:
        content = _SECTION_0A_GUARD.read_text(encoding="utf-8")

        assert '"phase_42_vocal_enhancement"' in content
        assert '"phase_58_lyrics_guided_enhancement": "Lyrics-Guided: §2.36 PFLICHT auch in Restoration' in content
