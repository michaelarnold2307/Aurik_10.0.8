from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import scripts.continuous_deep_analysis as cda_module
from scripts.continuous_deep_analysis import ContinuousDeepAnalyzer, PhaseCheckpoint


def test_collect_checkpoints_prefers_phase_local_hpi_and_afg() -> None:
    analyzer = ContinuousDeepAnalyzer(realtime=False)

    metadata = {
        "holistic_perceptual_gate": {"hpi": 0.88, "artifact_freedom": 0.99},
        "artifact_freedom": {"score": 0.99},
        "carrier_chain_recovery_ratio": 0.12,
        "pmgg_log_entries": [
            {
                "phase_id": "phase_03_denoise",
                "timestamp": 1.0,
                "scores_before": {"natuerlichkeit": 0.92},
                "scores_after": {"natuerlichkeit": 0.91},
                "hpi_score": 0.41,
                "artifact_freedom_score": 0.96,
            },
            {
                "phase_id": "phase_04_eq_correction",
                "timestamp": 2.0,
                "scores_before": {"natuerlichkeit": 0.91},
                "scores_after": {"natuerlichkeit": 0.92},
                "holistic_perceptual_gate": {"hpi": 0.73, "artifact_freedom": 0.98},
            },
        ],
    }
    restoration_result = SimpleNamespace(metadata=metadata)

    analyzer._collect_checkpoints_from_restore_result(restoration_result, pre_result=None)

    assert len(analyzer.checkpoints) == 2
    assert analyzer.checkpoints[0].hpi_score == 0.41
    assert analyzer.checkpoints[0].artifact_freedom == 0.96
    assert analyzer.checkpoints[1].hpi_score == 0.73
    assert analyzer.checkpoints[1].artifact_freedom == 0.98


def test_collect_checkpoints_reads_phase_proxies_from_metadata() -> None:
    analyzer = ContinuousDeepAnalyzer(realtime=False)

    metadata = {
        "holistic_perceptual_gate": {"hpi": 0.90, "artifact_freedom": 0.99},
        "artifact_freedom": {"score": 0.99},
        "carrier_chain_recovery_ratio": 0.10,
        "pmgg_log_entries": [
            {
                "phase_id": "phase_01_click_removal",
                "timestamp": 1.0,
                "scores_before": {"natuerlichkeit": 0.90},
                "scores_after": {"natuerlichkeit": 0.91},
                "metadata": {
                    "phase_hpi_proxy": 0.64,
                    "phase_artifact_freedom_proxy": 0.97,
                },
            },
            {
                "phase_id": "phase_02_hum_removal",
                "timestamp": 2.0,
                "scores_before": {"natuerlichkeit": 0.91},
                "scores_after": {"natuerlichkeit": 0.89},
                "metadata": {
                    "phase_hpi_proxy": 0.58,
                    "phase_artifact_freedom_proxy": 0.94,
                },
            },
        ],
    }
    restoration_result = SimpleNamespace(metadata=metadata)

    analyzer._collect_checkpoints_from_restore_result(restoration_result, pre_result=None)

    assert len(analyzer.checkpoints) == 2
    assert analyzer.checkpoints[0].hpi_score == 0.64
    assert analyzer.checkpoints[0].artifact_freedom == 0.97
    assert analyzer.checkpoints[1].hpi_score == 0.58
    assert analyzer.checkpoints[1].artifact_freedom == 0.94


def test_generate_summary_requires_hpi_gate_for_excellent() -> None:
    analyzer = ContinuousDeepAnalyzer(realtime=False)
    analyzer._final_musical_goals = {
        "natuerlichkeit": 0.95,
        "authentizitaet": 0.93,
    }
    analyzer.checkpoints = [
        PhaseCheckpoint(
            phase_id="phase_65_vocal_naturalness_restoration",
            wall_time_s=0.0,
            musical_goals={"natuerlichkeit": 0.95, "authentizitaet": 0.93},
            hpi_score=0.46,
            artifact_freedom=0.99,
            carrier_recovery_ratio=0.0,
            noise_floor_db=None,
            defects_remaining=None,
            anomalies=[],
        )
    ]

    summary = analyzer._generate_summary()

    assert summary["quality_status"] == "NEEDS_REVIEW"
    assert "hpi<0.60" in summary["quality_gate_reasons"]


def test_extract_vocal_metrics_from_metadata_reads_vqi_and_identity() -> None:
    analyzer = ContinuousDeepAnalyzer(realtime=False)

    metrics = analyzer._extract_vocal_metrics_from_metadata(
        {
            "vqi": 0.83,
            "singer_identity_cosine": 0.94,
            "holistic_perceptual_gate": {"mert_similarity": 0.91},
        }
    )

    assert metrics["vqi"] == 0.83
    assert metrics["singer_identity_cosine"] == 0.94
    assert metrics["mert_similarity"] == 0.91


def test_generate_summary_includes_final_vocal_metrics() -> None:
    analyzer = ContinuousDeepAnalyzer(realtime=False)
    analyzer._final_musical_goals = {
        "natuerlichkeit": 0.92,
        "authentizitaet": 0.91,
    }
    analyzer._final_vocal_metrics = {
        "vqi": 0.86,
        "singer_identity_cosine": 0.95,
    }
    analyzer.checkpoints = [
        PhaseCheckpoint(
            phase_id="phase_03_denoise",
            wall_time_s=0.0,
            musical_goals={"natuerlichkeit": 0.92, "authentizitaet": 0.91},
            hpi_score=0.72,
            artifact_freedom=0.97,
            carrier_recovery_ratio=0.0,
            noise_floor_db=None,
            defects_remaining=None,
            anomalies=[],
        )
    ]

    summary = analyzer._generate_summary()

    assert summary["final_vqi"] == 0.86
    assert summary["final_singer_identity_cosine"] == 0.95


def test_run_analysis_resets_state_between_runs(monkeypatch, tmp_path) -> None:
    analyzer = ContinuousDeepAnalyzer(realtime=False)
    analyzer.checkpoints = [
        PhaseCheckpoint(
            phase_id="phase_old",
            wall_time_s=0.0,
            musical_goals={"natuerlichkeit": 0.5},
            hpi_score=0.2,
            artifact_freedom=0.8,
            carrier_recovery_ratio=0.0,
            noise_floor_db=None,
            defects_remaining=None,
            anomalies=[],
        )
    ]
    analyzer._final_musical_goals = {"natuerlichkeit": 0.99}
    analyzer._final_vocal_metrics = {"vqi": 0.99}

    monkeypatch.setattr(
        cda_module,
        "load_audio_file",
        lambda _path: {"audio": np.zeros(480, dtype=np.float32), "sr": 48000},
    )
    monkeypatch.setattr(
        cda_module,
        "run_pre_analysis",
        lambda *_args, **_kwargs: SimpleNamespace(
            medium=SimpleNamespace(primary_material="vinyl"),
            era=SimpleNamespace(decade=1970),
            restorability=SimpleNamespace(restorability_score=60.0),
            defects=SimpleNamespace(scores={}),
        ),
    )

    class _FakeRestorer:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.config = SimpleNamespace(mode=SimpleNamespace(value="quality"))

        def restore(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(
                musical_goals={},
                metadata={
                    "pmgg_log_entries": [],
                    "holistic_perceptual_gate": {"hpi": 0.75},
                    "artifact_freedom": {"score": 0.98},
                },
            )

        def is_studio_mode(self):
            return False

    monkeypatch.setattr(cda_module, "UnifiedRestorerV3", _FakeRestorer)

    result = analyzer.run_analysis(
        audio_path="dummy.wav",
        output_dir=str(tmp_path),
    )

    assert result["checkpoints"] == []
    assert result["final_musical_goals"] == {}
    assert result["final_vocal_metrics"] == {}
    assert result["summary"] == {"status": "no_checkpoints"}
