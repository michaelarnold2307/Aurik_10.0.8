"""AMRB CI-Gate — blockiert Merge wenn OS-Führerschaft-Schwelle nicht erfüllt.

Spec §8.1 (copilot-instructions.md):
    AMRB-Gesamt-Score ≥ 84.0 UND ≥ 8/10 Szenarien bestanden.
    Baselines: iZotope RX 10 ≈ 71.0, Aurik 9.9 Restoration ≈ 84.0.

Laufzeit:  ~60–180 s (n_items_per_scenario=1, synthetische Signale intern erzeugt).
Ausführung: pytest tests/normative/test_amrb_ci_gate.py -m amrb --timeout=600 -v
Ausschluss: pytest -m "not amrb"  (für schnelle Unit-Test-Läufe)
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
# Referenzwerte aus AMRB_BASELINES (§8.1, AMRB v1.0)
# ---------------------------------------------------------------------------
_UNPROCESSED_MUSHRA: float = AMRB_BASELINES["Unbearbeitet (degradiert)"]["mushra_overall"]  # 32.0
_IZOTOPE_MUSHRA: float = AMRB_BASELINES["iZotope RX 10 (commercial)"]["mushra_overall"]  # 71.0
_AURIK_TARGET: float = 84.0  # OS-Führerschaft-Schwelle (§8.1)
_SCENARIOS_REQUIRED: int = 8  # von 10 Szenarien müssen bestanden sein


# ---------------------------------------------------------------------------
# Hilfsfunktion: Aurik-Pipeline als AMRB restoration_fn
# ---------------------------------------------------------------------------


def _aurik_restoration_fn(audio: np.ndarray, sr: int) -> np.ndarray:
    """Ruft UnifiedRestorerV3 auf; fällt bei Fehler auf Pass-Through zurück."""
    try:
        from backend.core.unified_restorer_v3 import get_restorer  # type: ignore[import]

        restorer = get_restorer()
        result = restorer.restore(audio, sr)
        out: np.ndarray = result.audio
        return out
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Aurik-Engine nicht verfügbar (%s) — Pass-Through (schlechte AMRB-Scores erwartet).",
            exc,
        )
        return audio


# ---------------------------------------------------------------------------
# Hilfsfunktion: Benchmark ausführen und Bericht zurückgeben
# ---------------------------------------------------------------------------


def _run_amrb(n_items: int = 1, verbose: bool = False) -> BenchmarkReport:
    config = BenchmarkConfig(
        restoration_fn=_aurik_restoration_fn,
        system_name="Aurik 9 CI",
        n_items_per_scenario=n_items,
        verbose=verbose,
    )
    return run_benchmark(config)


# ===========================================================================
# Normative Tests
# ===========================================================================


@pytest.mark.amrb
@pytest.mark.timeout(600)
def test_amrb_os_leadership_threshold() -> None:
    """Aurik muss AMRB overall_score ≥ 84.0 UND n_passed ≥ 8/10 erreichen (§8.1).

    Dieser Test blockiert einen Merge, wenn Aurik die OS-Führerschaft-Schwelle
    unterschreitet. Laufzeit ca. 60–180 s (synthetische Signale, n=1 pro Szenario).
    """
    report = _run_amrb(n_items=1, verbose=True)

    assert report.passes_os_leadership_threshold(), (
        f"\nAMRB OS-Führerschaft NICHT ERREICHT:\n"
        f"  Gesamt-Score : {report.overall_score:.1f}/100  (Ziel: ≥ {_AURIK_TARGET:.1f})\n"
        f"  Bestanden    : {report.n_passed}/10          (Ziel: ≥ {_SCENARIOS_REQUIRED})\n"
        f"  Schwächstes  : {report.worst_scenario}\n"
        f"\n"
        f"  Referenz iZotope RX 10 : {_IZOTOPE_MUSHRA:.1f}\n"
        f"  Referenz Unbearbeitet  : {_UNPROCESSED_MUSHRA:.1f}\n"
        f"\n"
        f"Maßnahme: Restaurierungslogik für Szenario '{report.worst_scenario}' prüfen.\n"
        f"Details: pytest -v --tb=long -m amrb"
    )


@pytest.mark.amrb
@pytest.mark.timeout(600)
def test_amrb_score_exceeds_izotope_baseline() -> None:
    """Aurik-Score muss über iZotope RX 10 Baseline (71.0) liegen (§8.2 Punkt 11)."""
    report = _run_amrb(n_items=1)

    assert report.overall_score > _IZOTOPE_MUSHRA, (
        f"Aurik ({report.overall_score:.1f}) liegt UNTER iZotope RX 10 Baseline "
        f"({_IZOTOPE_MUSHRA:.1f}). Kein Weltmarktführer-Anspruch."
    )


@pytest.mark.amrb
@pytest.mark.timeout(600)
def test_amrb_score_far_above_unprocessed() -> None:
    """Aurik-Score muss mindestens 40 MUSHRA-Punkte über Unbearbeitet liegen."""
    report = _run_amrb(n_items=1)
    min_required = _UNPROCESSED_MUSHRA + 40.0  # 32 + 40 = 72

    assert report.overall_score >= min_required, (
        f"Aurik-Score {report.overall_score:.1f} ist zu nah an 'Unbearbeitet' "
        f"({_UNPROCESSED_MUSHRA:.1f}). Mindestens {min_required:.1f} erwartet."
    )


@pytest.mark.amrb
@pytest.mark.timeout(600)
def test_amrb_at_least_8_scenarios_passed() -> None:
    """Genau ≥ 8/10 Szenarien müssen bestanden sein (MUSHRA ≥ 80 pro Szenario)."""
    report = _run_amrb(n_items=1)

    assert report.n_passed >= _SCENARIOS_REQUIRED, (
        f"Nur {report.n_passed}/10 Szenarien bestanden (Ziel: ≥ {_SCENARIOS_REQUIRED}).\n"
        f"Schwächstes Szenario: {report.worst_scenario}\n"
        f"Gesamt-Score: {report.overall_score:.1f}/100"
    )


@pytest.mark.amrb
@pytest.mark.timeout(600)
def test_amrb_report_fields_complete() -> None:
    """BenchmarkReport enthält alle Pflichtfelder mit sinnvollen Werten."""
    report = _run_amrb(n_items=1)

    # Numerische Grenzen
    assert 0.0 <= report.overall_score <= 100.0, "overall_score außerhalb [0, 100]"
    assert report.n_scenarios == 10, f"Erwartet 10 Szenarien, erhalten: {report.n_scenarios}"
    assert 0 <= report.n_passed <= report.n_scenarios

    # Felder nicht leer
    assert report.system_name, "system_name ist leer"
    assert report.worst_scenario, "worst_scenario ist leer"
    assert report.best_scenario, "best_scenario ist leer"
    assert report.scenario_results, "scenario_results ist leer"

    # Szenario-Ergebnisse
    for sid, res in report.scenario_results.items():
        assert 0.0 <= res.mushra_mean <= 100.0, f"mushra_mean für '{sid}' außerhalb [0, 100]: {res.mushra_mean}"
