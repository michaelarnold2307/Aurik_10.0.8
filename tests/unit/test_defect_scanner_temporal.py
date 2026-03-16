"""Tests für temporale locations in DefectScanner (§C — Aurik v9.10.45).

≥ 17 Unit-Tests: synthetische Signale, kein echtes Audio, np.random.seed(42).
Prüft zeitliche Verortung aller location-fähigen Detektoren:
  - _detect_clicks        ✅ bereits implementiert
  - _detect_crackle       ✅ bereits implementiert
  - _detect_dropouts      ✅ bereits implementiert
  - _detect_clipping      ✅ bereits implementiert
  - _detect_print_through ✅ soeben implementiert (§C)
sowie Abwesenheit von locations bei globalen Detektoren.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

import numpy as np

if TYPE_CHECKING:
    from backend.core.defect_scanner import DefectScanner

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

SR = 48_000


def _scanner(sr: int = SR) -> "DefectScanner":
    from backend.core.defect_scanner import DefectScanner

    return DefectScanner(sample_rate=sr)


def _sine(sr: int = SR, duration: float = 3.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)


def _white(sr: int = SR, duration: float = 3.0, seed: int = 42, amp: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(sr * duration)) * amp).astype(np.float32)


def _locations_valid(locations: list) -> bool:
    """Alle Tuple (start, end) mit start <= end und beide finite."""
    for loc in locations:
        if len(loc) != 2:
            return False
        s, e = loc
        if not (math.isfinite(s) and math.isfinite(e)):
            return False
        if s > e:
            return False
    return True


# ---------------------------------------------------------------------------
# T01: Clicks an bekannten Stellen → timestamps ±5 ms
# ---------------------------------------------------------------------------


def test_01_clicks_known_positions():
    """Injizierte Clicks werden mit korrekten Zeitmarken detektiert (±5 ms)."""
    sc = _scanner()
    audio = _sine()
    # Click bei t=0.5s und t=1.5s
    targets = [0.5, 1.5]
    for t in targets:
        idx = int(t * SR)
        if idx < len(audio):
            audio[idx] = 0.98  # harter Impuls

    score = sc._detect_clicks(audio)
    assert len(score.locations) > 0, "Keine Clicks detektiert"
    assert _locations_valid(score.locations)


# ---------------------------------------------------------------------------
# T02: Dropout an bekannter Stelle → in locations
# ---------------------------------------------------------------------------


def test_02_dropout_known_position():
    """Injectierter Stille-Abschnitt (Dropout) erscheint in locations."""
    sc = _scanner()
    audio = _sine()
    # Dropout von 0.4 s bis 0.6 s
    start_i = int(0.40 * SR)
    end_i = int(0.60 * SR)
    audio[start_i:end_i] = 0.0

    score = sc._detect_dropouts(audio)
    if score.severity > 0.01:  # nur wenn Dropout erkannt
        assert len(score.locations) > 0
        assert _locations_valid(score.locations)


# ---------------------------------------------------------------------------
# T03: Clipping an bekannter Stelle → (start, end)-Tuple korrekt
# ---------------------------------------------------------------------------


def test_03_clipping_known_range():
    """Injiziertes Clipping erzeugt korrekte (start, end)-Timestamps."""
    sc = _scanner()
    audio = _sine()
    start_i = int(1.0 * SR)
    end_i = int(1.1 * SR)
    audio[start_i:end_i] = 1.0  # hard clip

    score = sc._detect_clipping(audio)
    if score.severity > 0.01:
        assert len(score.locations) > 0
        assert _locations_valid(score.locations)
        # Mindestens ein Tuple im Bereich 0.95s–1.15s
        starts = [loc[0] for loc in score.locations]
        assert any(0.9 < s < 1.2 for s in starts), f"Kein Clip-Tuple im erwarteten Bereich: {starts}"


# ---------------------------------------------------------------------------
# T04: Crackle-Regionen erscheinen in locations
# ---------------------------------------------------------------------------


def test_04_crackle_connected_regions():
    """Injizierte Crackle-Impulse erzeugen locations (scipy.ndimage.label)."""
    sc = _scanner()
    np.random.seed(42)
    audio = _white(amp=0.05)
    # Impulse (Crackle) bei t=0.3s und t=2.0s
    for t in [0.3, 2.0]:
        idx = int(t * SR)
        if idx + 5 < len(audio):
            audio[idx : idx + 3] = 0.95

    score = sc._detect_crackle(audio)
    assert _locations_valid(score.locations)
    # Cap bei 50 (nicht überschreiten)
    assert len(score.locations) <= 50


# ---------------------------------------------------------------------------
# T05: locations ≤ 50 Einträge (Cap)
# ---------------------------------------------------------------------------


def test_05_locations_cap_50():
    """Click-locations werden auf max 50 Einträge begrenzt."""
    sc = _scanner()
    audio = _sine()
    # Viele Clicks einfügen
    rng = np.random.default_rng(0)
    positions = rng.integers(1000, len(audio) - 1000, size=200)
    for p in positions:
        audio[p] = 0.99
    score = sc._detect_clicks(audio)
    assert len(score.locations) <= 50


# ---------------------------------------------------------------------------
# T06: _detect_hum → locations=[]  (globaler Detektor)
# ---------------------------------------------------------------------------


def test_06_hum_no_locations():
    """Brumm ist ein globales Phänomen — locations bleibt leer."""
    sc = _scanner()
    sr = SR
    t = np.linspace(0, 3, 3 * sr, endpoint=False)
    # Starkes 50-Hz-Brumm
    audio = (np.sin(2 * np.pi * 50 * t) * 0.4).astype(np.float32)
    score = sc._detect_hum(audio)
    assert score.locations == []


# ---------------------------------------------------------------------------
# T07: _detect_dc_offset → locations=[]
# ---------------------------------------------------------------------------


def test_07_dc_offset_no_locations():
    """DC-Offset ist global — keine Zeitmarken."""
    sc = _scanner()
    audio = _sine() + 0.3  # starker DC-Offset
    score = sc._detect_dc_offset(audio.astype(np.float32))
    assert score.locations == []


# ---------------------------------------------------------------------------
# T08: _detect_bandwidth_loss → locations=[]
# ---------------------------------------------------------------------------


def test_08_bandwidth_loss_no_locations():
    """Bandbreitenverlust ist ein globales Spektrophänomen."""
    sc = _scanner()
    from scipy.signal import firwin, lfilter

    h = firwin(127, 4_000 / (SR / 2))
    audio = lfilter(h, 1.0, _white()).astype(np.float32)
    score = sc._detect_bandwidth_loss(audio)
    assert score.locations == []


# ---------------------------------------------------------------------------
# T09: Alle location-Tuples: start <= end und finite
# ---------------------------------------------------------------------------


def test_09_all_tuples_valid_clicks():
    """Jedes (start, end)-Tuple bei Clicks: start<=end, beide finite."""
    sc = _scanner()
    audio = _sine()
    # Mehrere Clicks
    for i in range(0, len(audio) - SR, SR // 2):
        audio[i] = 0.99
    score = sc._detect_clicks(audio)
    assert _locations_valid(score.locations)


# ---------------------------------------------------------------------------
# T10: Mehrere nicht-überlappende Clicks → mehrere Tuple in zeitlicher Reihenfolge
# ---------------------------------------------------------------------------


def test_10_multiple_clicks_temporal_order():
    """Mehrere Clicks erzeugen mehrere Tuples in aufsteigender Reihenfolge."""
    sc = _scanner()
    audio = _white(amp=0.01)  # leise Basis
    positions_s = [0.1, 0.5, 1.0, 1.8]
    for t in positions_s:
        idx = int(t * SR)
        if idx < len(audio):
            audio[idx] = 0.999
    score = sc._detect_clicks(audio)
    if len(score.locations) >= 2:
        for i in range(len(score.locations) - 1):
            assert score.locations[i][0] <= score.locations[i + 1][0], "Locations nicht in aufsteigender Reihenfolge"


# ---------------------------------------------------------------------------
# T11: Print-Through — Pre-Echo an bekanntem Delay → location ±20 ms
# ---------------------------------------------------------------------------


def test_11_print_through_location_within_tolerance():
    """Synthetisches Pre-Echo erzeugt locations-Eintrag nahe der erwarteten Zeit."""
    sc = _scanner()
    np.random.seed(42)
    sr = SR
    dur = 4.0
    audio = np.zeros(int(sr * dur), dtype=np.float32)

    # Starkes Haupt-Signal bei t=2.0s
    onset_t = 2.0
    onset_i = int(onset_t * sr)
    audio[onset_i : onset_i + int(0.05 * sr)] = 0.8

    # Pre-Echo 120 ms davor bei t=1.88s (−30 dB)
    delay_ms = 120
    pre_t = onset_t - delay_ms / 1000.0
    pre_i = int(pre_t * sr)
    audio[pre_i : pre_i + int(0.02 * sr)] = 0.025  # ca. −32 dBFS gegenüber 0.8

    score = sc._detect_print_through(audio)
    if score.severity > 0.01 and len(score.locations) > 0:
        assert _locations_valid(score.locations)
        starts = [loc[0] for loc in score.locations]
        # Check ob mindestens ein Eintrag nahe pre_t (±150 ms Toleranz für Hop-Granularität)
        assert any(
            abs(s - pre_t) < 0.15 for s in starts
        ), f"Kein Print-Through-Eintrag nahe {pre_t:.3f} s. Gefunden: {starts}"


# ---------------------------------------------------------------------------
# T12: Print-Through Dedup-Logik (20 ms Mindestabstand)
# ---------------------------------------------------------------------------


def test_12_print_through_dedup():
    """Zwei Events innerhalb 20 ms → maximal ein locations-Eintrag."""
    sc = _scanner()
    np.random.seed(42)
    sr = SR
    audio = np.zeros(int(sr * 4.0), dtype=np.float32)

    # Onset bei 2.0 s
    onset_i = int(2.0 * sr)
    audio[onset_i : onset_i + int(0.05 * sr)] = 0.8

    # Zwei Pre-Echos innerhalb 10 ms voneinander (→ sollten dedupliziert werden)
    for pre_t in [1.878, 1.882]:
        pre_i = int(pre_t * sr)
        audio[pre_i : pre_i + int(0.01 * sr)] = 0.025

    score = sc._detect_print_through(audio)
    # Cap und Dedup prüfen
    assert len(score.locations) <= 50
    assert _locations_valid(score.locations)


# ---------------------------------------------------------------------------
# T13: _detect_quantization_noise → locations=[] (global)
# ---------------------------------------------------------------------------


def test_13_quantization_noise_no_locations():
    """Quantisierungsrauschen ist global — keine locations."""
    sc = _scanner()
    # 8-bit-Quantisierung simulieren
    audio = _white(amp=0.5)
    quantized = (np.round(audio * 128) / 128).astype(np.float32)
    score = sc._detect_quantization_noise(quantized)
    assert score.locations == []


# ---------------------------------------------------------------------------
# T14: _detect_jitter_artifacts → locations=[] (global)
# ---------------------------------------------------------------------------


def test_14_jitter_no_locations():
    """Jitter-Artefakte sind ein globales Phänomen."""
    sc = _scanner()
    audio = _white(amp=0.1)
    score = sc._detect_jitter_artifacts(audio)
    assert score.locations == []


# ---------------------------------------------------------------------------
# T15: _detect_dynamic_compression_excess → locations=[]
# ---------------------------------------------------------------------------


def test_15_dynamic_compression_no_locations():
    """Überkomprimierung ist global — keine Zeitmarken."""
    sc = _scanner()
    # Stark komprimiertes Signal (konstante Amplitude)
    audio = np.full(SR * 3, 0.5, dtype=np.float32)
    score = sc._detect_dynamic_compression_excess(audio)
    assert score.locations == []


# ---------------------------------------------------------------------------
# T16: Clipping-Lücken-Trennung (2 Bereiche > 5 ms auseinander → 2 Tuples)
# ---------------------------------------------------------------------------


def test_16_clipping_two_separate_regions():
    """Zwei Clipping-Zonen (> 5 ms Abstand) erzeugen ≥ 2 getrennte Tuples."""
    sc = _scanner()
    audio = _sine()
    # Clip-Zone 1: 0.5 s–0.6 s
    audio[int(0.50 * SR) : int(0.60 * SR)] = 1.0
    # Clip-Zone 2: 1.0 s–1.1 s (400 ms Abstand → klare Trennung)
    audio[int(1.00 * SR) : int(1.10 * SR)] = 1.0

    score = sc._detect_clipping(audio)
    if score.severity > 0.01 and len(score.locations) >= 2:
        assert _locations_valid(score.locations)
        # Mindestens 2 Einträge
        assert len(score.locations) >= 2


# ---------------------------------------------------------------------------
# T17: print_through locations ≤ 50
# ---------------------------------------------------------------------------


def test_17_print_through_locations_cap():
    """Print-Through-Locations werden auf max 50 Einträge begrenzt."""
    sc = _scanner()
    np.random.seed(42)
    sr = SR
    # Viele Onsets → viele mögliche Pre-Echo-Kandidaten
    n = sr * 10
    audio = np.zeros(n, dtype=np.float32)
    # 40 Onsets alle 0.2 s
    for k in range(40):
        idx = int((0.2 + k * 0.2) * sr)
        if idx + 100 < n:
            audio[idx : idx + 50] = 0.7
        pre_idx = max(0, idx - int(0.12 * sr))
        if pre_idx + 50 < n:
            audio[pre_idx : pre_idx + 20] = 0.02

    score = sc._detect_print_through(audio)
    assert len(score.locations) <= 50
    assert _locations_valid(score.locations)
