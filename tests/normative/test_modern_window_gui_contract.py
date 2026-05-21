"""Normative GUI contract checks for ModernMainWindow.

This test suite enforces a subset of hard GUI invariants defined in
copilot-instructions/specs for the modern frontend path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

GUI_FILE = Path("Aurik910/ui/modern_window.py")


def _read_gui_source() -> str:
    assert GUI_FILE.exists(), f"Missing GUI file: {GUI_FILE}"
    return GUI_FILE.read_text(encoding="utf-8")


@pytest.mark.normative
def test_progress_bar_uses_10000_scale_in_modern_window() -> None:
    src = _read_gui_source()
    assert "self.progress_bar.setRange(0, 10000)" in src
    assert "self.phase_progress_bar.setRange(0, 10000)" in src
    assert "self.refinement_progress_bar.setRange(0, 10000)" in src
    assert "setRange(0, 100)" not in src


@pytest.mark.normative
def test_preanalysis_gate_controls_recommendation_visibility() -> None:
    src = _read_gui_source()
    assert "_preanalysis_finalized_for" in src
    # Code uses _ready instead of _show_recommendation (same logic)
    assert "_ready = bool(_cfp) and (_finalized_for == _cfp)" in src
    assert "self._apply_mode_recommendation_visuals()" in src


@pytest.mark.normative
def test_magic_button_sync_gate_hooks_present() -> None:
    src = _read_gui_source()
    assert "def _try_signal_preanalysis_done(flag: str) -> None:" in src
    assert '_try_signal_preanalysis_done("era_genre")' in src
    assert '_try_signal_preanalysis_done("defect_scan")' in src
    assert "QTimer.singleShot(15_000, _preanalysis_timeout)" in src


@pytest.mark.normative
def test_preanalysis_hard_timeout_does_not_bypass_defect_scan_gate() -> None:
    src = _read_gui_source()
    assert "def _preanalysis_hard_timeout() -> None:" in src
    assert "QTimer.singleShot(180_000, _preanalysis_hard_timeout)" in src
    assert 'if "defect_scan" in self._preanalysis_flags:' in src
    assert "Pre-analysis hard-timeout reached, waiting for defect_scan before finalization" in src


@pytest.mark.normative
def test_era_chip_not_rendered_as_separate_chip() -> None:
    src = _read_gui_source()
    assert "chip_era wird NICHT gesetzt" in src
    assert "_show_chip(self.chip_era" not in src


@pytest.mark.normative
def test_bridge_fallback_contract_and_export_guard_present() -> None:
    src = _read_gui_source()
    assert "_BRIDGE_AVAILABLE = False" in src
    assert "def _export_guard(audio):" in src
    assert "np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)" in src
    assert "return np.clip(audio, -1.0, 1.0)" in src

    # Required bridge fallback stubs from the frontend contract.
    required_none_stubs = (
        "def _bridge_get_audio_file_validator() -> object | None:",
        "def _bridge_get_defect_scanner() -> type | None:",
        "def _bridge_get_defect_type() -> type | None:",
        "def _bridge_get_quality_mode() -> object | None:",
        "def _bridge_get_restorer_classes() -> object | None:",
        "def _bridge_get_medium_classifier_fn() -> object | None:",
        "def _bridge_get_era_classifier_fn() -> object | None:",
        "def _bridge_get_genre_classifier_fn() -> object | None:",
        "def _bridge_get_carrier_forensics_fn() -> object | None:",
        "def _bridge_get_audio_exporter_class() -> type | None:",
    )
    for stub in required_none_stubs:
        assert stub in src


@pytest.mark.normative
def test_shortcuts_complete() -> None:
    src = _read_gui_source()
    assert "_setup_shortcuts" in src
    # Required shortcuts from §11.4
    for seq in ("Key_Space", "Key_A", "Key_B", "Key_Escape", "Ctrl+R", "Ctrl+Shift+R", "Key_L"):
        assert seq in src, f"Missing shortcut: {seq}"
    # StandardKey for Ctrl+O / Ctrl+S
    assert "StandardKey.Open" in src
    assert "StandardKey.Save" in src


@pytest.mark.normative
def test_kmv_signals_and_refinement_bar_present() -> None:
    src = _read_gui_source()
    # KMV §2.38 signal connections
    for sig in ("refinement_started", "refinement_progress", "refinement_complete", "refinement_cancelled"):
        assert f".{sig}.connect(" in src, f"Missing KMV signal connect: {sig}"
    # Refinement bar: türkis, 3 px, range 0-10000, hidden until Stage 2
    assert "refinement_progress_bar" in src
    assert "refinement_progress_bar.setRange(0, 10000)" in src
    assert "int(pct) * 100" in src  # correct scaling


@pytest.mark.normative
def test_watchdog_formula_correct() -> None:
    src = _read_gui_source()
    # §11.4 Watchdog-Timer Formel: max(5_400_000, dur * 32_000 + 1_800_000)
    assert "32_000" in src
    assert "5_400_000" in src
    assert "1_800_000" in src


@pytest.mark.normative
def test_no_direct_magic_button_enable_in_preanalysis_emit_paths() -> None:
    src = _read_gui_source()
    # Contract: enabling should happen through _finalize_preanalysis(),
    # not directly in era/genre or defect pre-analysis update hooks.
    _forbidden_snippets = (
        "_run_defect_scan_bg._apply():\n",
        "_detect_era_genre_bg._upd():\n",
    )
    for marker in _forbidden_snippets:
        assert marker not in src

    _finalize_anchor = src.find("def _finalize_preanalysis() -> None:")
    _signal_anchor = src.find("def _try_signal_preanalysis_done(flag: str) -> None:")
    assert _finalize_anchor >= 0 and _signal_anchor > _finalize_anchor
    _finalize_block = src[_finalize_anchor:_signal_anchor]
    assert "_set_magic_buttons_enabled(True)" in _finalize_block


@pytest.mark.normative
def test_preventive_metadata_visibility_contract_in_quality_banner() -> None:
    src = _read_gui_source()

    # Frontend must keep phase-local prevention telemetry visible in post-run UI.
    assert '"preventive_actions": []' in src
    assert "phase31_damage_shield_applied" in src
    assert "phase31_stereo_delay_corrected" in src
    assert "loudness_makeup_db" in src
    assert "def _build_quality_banner_sections(" in src
    assert "preventive_actions: list[str]" in src
    assert "🛡️  Präventionsschutz:" in src


@pytest.mark.normative
def test_preventive_actions_callsite_contract_in_quality_pipeline() -> None:
    src = _read_gui_source()

    # Separate contract: extraction + callsite forwarding must stay intact.
    assert 'preventive_actions: list[str] = _ctx["preventive_actions"]' in src
    assert "self.prognose_widget.set_preventive_actions(preventive_actions)" in src
    assert "preventive_actions=preventive_actions" in src


@pytest.mark.normative
def test_bridge_unavailable_warning_is_one_shot_and_sets_runtime_health_state() -> None:
    src = _read_gui_source()
    assert 'if bool(getattr(self, "_bridge_unavailable_warning_shown", False)):' in src
    assert "self._bridge_unavailable_warning_shown = True" in src
    assert 'self._runtime_health_state = "bridge_unavailable"' in src


@pytest.mark.normative
def test_runtime_original_fallback_detection_uses_structured_metadata_signals() -> None:
    src = _read_gui_source()
    assert "def _detect_runtime_original_fallback_reason(restoration_result) -> str:" in src
    assert 'metadata = getattr(restoration_result, "metadata", {}) or {}' in src
    assert 'stage_notes = getattr(restoration_result, "stage_notes", {}) or {}' in src
    assert "export_quality_gate_failed" in src
    assert "export_blocked_by_quality_gate" in src
    assert "RUNTIME_ORIGINAL_FALLBACK" in src


@pytest.mark.normative
def test_warnings_are_gated_by_non_sota_status_in_quality_ui() -> None:
    src = _read_gui_source()
    assert '"is_sota_run": True' in src
    assert '"sota_warning_reason": ""' in src
    assert "if not is_sota_run:" in src
    assert "Nicht-SOTA-Ausführung" in src
    assert (
        'has_problem = (not is_sota_run) and (degradation_status in {"blocked", "critical_degraded", "degraded"})'
        in src
    )


@pytest.mark.normative
def test_musiclover_sota_metadata_forwarding_in_export_path() -> None:
    src = _read_gui_source()
    assert "quality_gate_musiclover_all_sota_real" in src
    assert "quality_gate_musiclover_sota_reason" in src


@pytest.mark.normative
def test_worldclass_gate_and_threshold_evidence_are_visible_in_quality_banner() -> None:
    src = _read_gui_source()
    assert '"_xp_threshold_evidence": {}' in src
    assert 'ctx["_xp_threshold_evidence"] = dict(_te_raw)' in src
    assert "xp_threshold_evidence: dict" in src
    assert 'qg_wcs = xp_quality_gate.get("worldclass_composite_gate", {})' in src
    assert "🏁  Worldclass-Gate:" in src
    assert "📚  Gate-Evidenz:" in src


@pytest.mark.normative
def test_worldclass_and_evidence_are_forwarded_in_export_metadata() -> None:
    src = _read_gui_source()
    assert "quality_gate_worldclass_score" in src
    assert "quality_gate_worldclass_threshold" in src
    assert "quality_gate_worldclass_passed" in src
    assert "quality_gate_worldclass_profile" in src
    assert "quality_gate_worldclass_artifact_veto" in src
    assert "quality_gate_evidence_worldclass_source_class" in src
    assert "quality_gate_evidence_worldclass_revalidate_by" in src


@pytest.mark.normative
def test_waveform_phase_animation_has_generic_fallback_and_progress_binding() -> None:
    src = _read_gui_source()
    assert "generic fallback" in src.lower()
    assert '_generic_key = f"generic:phase_' in src
    assert (
        "self.batch_thread.phase_progress.connect(lambda v: self.waveform_widget.set_stage_progress(v / 10000.0))"
        in src
    )
    assert "self.batch_thread.phase_progress.connect(" in src
    assert "self.waveform_widget_rest_ab.set_stage_progress(v / 10000.0)" in src


@pytest.mark.normative
def test_phase_step_label_has_no_audio_callback_fallback() -> None:
    src = _read_gui_source()
    assert "if _cur_pct >= 20:" in src
    assert "self._pending_step_info = label" in src
    assert "Fallback: some phases do not emit an audio snapshot callback." in src
    assert "self._phase_step_label.setText(label)" in src
    assert "self._phase_step_label.setVisible(True)" in src


@pytest.mark.normative
def test_waveform_stage_and_scan_are_mirrored_to_rest_ab_widget() -> None:
    src = _read_gui_source()
    assert "self.waveform_widget_rest_ab.set_scan_pos(frac)" in src
    assert "self.waveform_widget_rest_ab.set_active_stage(phase_text)" in src
    assert "self.waveform_widget_rest_ab.set_scan_pos(-1.0)" in src
    assert "self.waveform_widget_rest_ab.clear_stage()" in src
