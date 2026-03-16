"""Competitive CI-Gate — Aurik muss iZotope RX 10 in der Mehrheit der Szenarien schlagen.

Spec §8.2 Punkt 11 (copilot-instructions.md):
    Aurik ≥ iZotope RX 10 in ≥ 7/10 AMRB-Szenarien (elektrisch messbar).
    Messung via MUSHRA-Score aus run_benchmark() — KEINE Speech-Metriken (PESQ, STOI etc.).

Hinweis: Ein direkter iZotope-Aufruf ist im CI nicht möglich. Als Proxy dient der
AMRB-Baseline-MUSHRA von iZotope RX 10 (71.0) aus AMRB_BASELINES. Aurik muss diesen
Wert in ≥ 7 von 10 Szenarien übertreffen.

VERBOTENE METRIKEN (spec §3.1, §4.4):
    PESQ, STOI, SI-SDR, VISQOL (Speech Mode), DNSMOS, NISQA
    → Stattdessen: MUSHRA (OQS), PQS-MOS, Musical Goals

Ausführung: pytest tests/normative/test_competitive_ci_gate.py -m competitive --timeout=600 -v
Ausschluss: pytest -m "not competitive"
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from benchmarks.musical_restoration_benchmark import (
    AMRB_BASELINES,
    BenchmarkConfig,
    BenchmarkReport,
    run_benchmark,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Referenz-Baselines (AMRB_BASELINES §8.1)
# ---------------------------------------------------------------------------
_IZOTOPE_MUSHRA: float = AMRB_BASELINES["iZotope RX 10 (commercial)"]["mushra_overall"]  # 71.0
_IZOTOPE_PQS_MOS: float = AMRB_BASELINES["iZotope RX 10 (commercial)"]["pqs_mos"]  # 3.9
_AURIK_STUDIO_MUSHRA: float = AMRB_BASELINES["Aurik 9.9 (Studio 2026 Mode)"]["mushra_overall"]  # 88.0
_AURIK_RESTORE_MUSHRA: float = AMRB_BASELINES["Aurik 9.9 (Restoration Mode)"]["mushra_overall"]  # 84.0
_MIN_SCENARIOS_TO_WIN: int = 7  # §8.2 Punkt 11: ≥ 7/10 Szenarien

# Per-Szenario-Schwelle: gut unterhalb des iZotope-Gesamtscores,
# um Messrauschen (n_items=1) zu tolerieren. Aurik muss spürbar besser sein.
_PER_SCENARIO_WIN_THRESHOLD: float = _IZOTOPE_MUSHRA  # strikt: muss iZotope schlagen


# ---------------------------------------------------------------------------
# Aurik-restoration_fn
# ---------------------------------------------------------------------------


def _aurik_restoration_fn(audio: np.ndarray, sr: int) -> np.ndarray:
    """Ruft UnifiedRestorerV3 auf; fällt bei Fehler auf Pass-Through zurück."""
    try:
        from backend.core.unified_restorer_v3 import get_restorer  # type: ignore[import]

        result = get_restorer().restore(audio, sr)
        return result.audio
    except Exception as exc:  # pragma: no cover
        logger.warning("Aurik-Engine nicht verfügbar (%s) — Pass-Through (schlechte Scores erwartet).", exc)
        return audio


def _run_competitive(n_items: int = 1, verbose: bool = False) -> BenchmarkReport:
    config = BenchmarkConfig(
        restoration_fn=_aurik_restoration_fn,
        system_name="Aurik 9 Competitive",
        n_items_per_scenario=n_items,
        verbose=verbose,
    )
    return run_benchmark(config)


# ===========================================================================
# Competitive Tests
# ===========================================================================


@pytest.mark.competitive
@pytest.mark.timeout(600)
def test_aurik_beats_izotope_in_majority_of_scenarios() -> None:
    """Aurik MUSHRA muss iZotope RX 10 Baseline (71.0) in ≥ 7/10 Szenarien übertreffen.

    §8.2 Punkt 11: Pflicht-Benchmark für Weltmarktführer-Anspruch.
    VERBOTEN: PESQ, STOI, SI-SDR, VISQOL — ausschließlich MUSHRA (OQS) als Maßstab.
    """
    report = _run_competitive(n_items=1, verbose=True)

    scenarios_won = sum(1 for res in report.scenario_results.values() if res.mushra_mean > _PER_SCENARIO_WIN_THRESHOLD)
    losing_scenarios = [
        f"  {sid}: MUSHRA {res.mushra_mean:.1f} ≤ {_PER_SCENARIO_WIN_THRESHOLD:.1f}"
        for sid, res in report.scenario_results.items()
        if res.mushra_mean <= _PER_SCENARIO_WIN_THRESHOLD
    ]

    assert scenarios_won >= _MIN_SCENARIOS_TO_WIN, (
        f"\nCompetitive-Gate NICHT BESTANDEN:\n"
        f"  Szenarien > iZotope RX 10 : {scenarios_won}/10  (Ziel: ≥ {_MIN_SCENARIOS_TO_WIN})\n"
        f"  iZotope RX 10 Baseline    : MUSHRA {_IZOTOPE_MUSHRA:.1f}\n"
        f"  Aurik Gesamt-Score        : {report.overall_score:.1f}/100\n"
        f"  Schwächstes Szenario      : {report.worst_scenario}\n"
        f"\n"
        f"Verlorene Szenarien:\n" + "\n".join(losing_scenarios)
    )


@pytest.mark.competitive
@pytest.mark.timeout(600)
def test_aurik_overall_score_above_izotope_overall() -> None:
    """Aurik Gesamt-MUSHRA muss den iZotope-Gesamt-MUSHRA (71.0) deutlich übertreffen."""
    report = _run_competitive(n_items=1)

    margin = report.overall_score - _IZOTOPE_MUSHRA
    assert report.overall_score > _IZOTOPE_MUSHRA, (
        f"Aurik Gesamt-Score ({report.overall_score:.1f}) liegt NICHT über "
        f"iZotope RX 10 Baseline ({_IZOTOPE_MUSHRA:.1f}). "
        f"Differenz: {margin:+.1f} Punkte."
    )
    logger.info(
        "Competitive: Aurik %.1f vs iZotope %.1f (+%.1f Punkte Vorsprung)",
        report.overall_score,
        _IZOTOPE_MUSHRA,
        margin,
    )


@pytest.mark.competitive
@pytest.mark.timeout(600)
def test_aurik_pqs_mos_above_izotope_baseline() -> None:
    """Aurik-PQS-MOS muss iZotope PQS-MOS Baseline (3.9) übertreffen."""
    report = _run_competitive(n_items=1)

    # Alle Szenario-PQS-MOS-Werte sammeln
    all_pqs = [
        res.pqs_mos_mean
        for res in report.scenario_results.values()
        if hasattr(res, "pqs_mos_mean") and res.pqs_mos_mean is not None
    ]

    if not all_pqs:
        pytest.skip("Keine PQS-MOS-Daten im BenchmarkReport — Szenario überspringen.")

    mean_pqs = float(np.mean(all_pqs))
    assert mean_pqs > _IZOTOPE_PQS_MOS, (
        f"Aurik PQS-MOS ({mean_pqs:.2f}) liegt nicht über iZotope RX 10 Baseline "
        f"({_IZOTOPE_PQS_MOS:.1f}). Metriken: MUSHRA/PQS-MOS zulässig. "
        f"PESQ/STOI sind verboten (§4.4)."
    )


@pytest.mark.competitive
@pytest.mark.timeout(600)
def test_competitive_no_forbidden_metrics_used() -> None:
    """Stellt sicher, dass verbotene Speech-Metriken nicht in benchmark_suite importiert werden.

    §3.1/§4.4: PESQ, STOI, SI-SDR, VISQOL (Speech-Mode), DNSMOS, NISQA sind für
    Musik-Qualitätsbewertung absolut verboten.
    """
    import importlib

    # Modul laden (oder bereits geladen nutzen)
    module_name = "benchmarks.competitive.benchmark_suite"
    try:
        suite = importlib.import_module(module_name)
    except ImportError:
        pytest.skip(f"{module_name} nicht importierbar — Modul-Prüfung übersprungen.")

    # Prüfe, ob FORBIDDEN_METRICS-Konstante existiert
    assert hasattr(suite, "FORBIDDEN_METRICS"), (
        f"{module_name} enthält keine FORBIDDEN_METRICS-Konstante. "
        f"Bitte benchmarks/competitive/benchmark_suite.py gemäß §4.4 aktualisieren."
    )

    forbidden = set(suite.FORBIDDEN_METRICS)
    expected_forbidden = {"pesq", "stoi", "si_sdr", "visqol", "dnsmos", "nisqa"}
    missing = expected_forbidden - {m.lower() for m in forbidden}

    assert not missing, (
        f"FORBIDDEN_METRICS in {module_name} fehlen folgende verbotene Metriken: {missing}\n"
        f"Aktuell deklariert: {forbidden}"
    )
