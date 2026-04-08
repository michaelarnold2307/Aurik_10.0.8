from pathlib import Path

GUI_FILE = Path("Aurik910/ui/modern_window.py")
I18N_FILE = Path("Aurik910/i18n/__init__.py")
DEFECT_SCANNER_FILE = Path("backend/core/defect_scanner.py")


def test_ui_heartbeat_reassures_during_long_processing_phases() -> None:
    src = GUI_FILE.read_text(encoding="utf-8")
    assert 't("status.processing_reassure_analysis")' in src
    assert 't("status.processing_reassure_long_phase")' in src
    assert 't("status.processing_reassure_finalize")' in src
    assert "_time_since_cb >= 8.0" in src
    assert '_user_phase_text = _expl.lstrip(" ·").strip() if _expl else ""' in src
    assert '"Jetzt: {_base}' in src
    assert "Danach: {_next_hint}" in src


def test_ui_status_and_phase_labels_use_eliding_labels() -> None:
    src = GUI_FILE.read_text(encoding="utf-8")
    assert "class _ElidingLabel(QLabel):" in src
    assert 'self._phase_step_label = _ElidingLabel("")' in src
    assert 'self.status_text = _ElidingLabel(t("status.ready"))' in src
    assert 'self.stats_label = _ElidingLabel(t("status.stats", pending=0, completed=0, failed=0))' in src


def test_ui_top_row_refresh_prevents_collisions_on_narrow_widths() -> None:
    src = GUI_FILE.read_text(encoding="utf-8")
    assert "def _refresh_top_info_row(self, window_width: int | None = None) -> None:" in src
    assert '_btn_cancel.setText("■ Stoppen" if _compact else "■  Restaurierung stoppen")' in src
    assert "_stats_should_hide = _very_compact and _running" in src
    assert "self._refresh_top_info_row(w)" in src
    assert "self._refresh_top_info_row()" in src


def test_ui_load_progress_switches_to_finalize_mode_near_98_percent() -> None:
    src = GUI_FILE.read_text(encoding="utf-8")
    assert "_load_progress = pyqtSignal(float)" in src
    assert "def _on_load_progress_update(self, pct: float) -> None:" in src
    assert "if _preanalysis_pending and _p >= 99.8:" in src
    assert "_bar.setRange(0, 0)" in src
    assert '"⏳ Analyse wird finalisiert …"' in src


def test_defect_scanner_emits_fine_grained_tail_progress() -> None:
    src = DEFECT_SCANNER_FILE.read_text(encoding="utf-8")
    assert 'progress_callback: Optional["Callable[[float, str], None]"]' in src
    assert 'def _prog(pct: float, name: str = "") -> None:' in src
    assert 'def _tail_tick(name: str = "") -> None:' in src
    assert "_DIGITAL_FAST_TAIL" in src
    assert "_ANALOG_DEEP_TAIL" in src
    assert "_tail_start = 93.0" in src
    assert "_tail_start = 88.0" in src
    assert "_prog(_tail_start + _tail_span * _frac, name)" in src


def test_ui_i18n_contains_reassuring_processing_messages() -> None:
    src = I18N_FILE.read_text(encoding="utf-8")
    assert '"status.processing_reassure_analysis": "Aurik prüft die Aufnahme weiter sorgfältig"' in src
    assert (
        '"status.processing_reassure_long_phase": "Rechenintensive Phase aktiv - Fortschritt läuft stabil weiter"'
        in src
    )
    assert (
        '"status.processing_reassure_finalize": "Aurik finalisiert das Ergebnis und sichert alle Qualitätsprüfungen"'
        in src
    )
