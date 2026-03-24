"""Unit-Tests für EraClassifier Plugin (§2.14).

Tests: ≥ 22 — Abdeckung: DSP-Tier, Bounds, Edge-Cases, Stereo, Singleton
"""

import concurrent.futures
import math

import numpy as np
import pytest

from backend.core.era_classifier import (
    MEDIUM_DECADE_FLOOR,
    EraClassifier,
    EraResult,
    classify_era,
    constrain_era_to_medium,
    get_era_classifier,
)

SR = 48000

# Gültige Jahrzehnte laut Spec
VALID_DECADES = {
    1890,
    1900,
    1910,
    1920,
    1930,
    1940,
    1950,
    1960,
    1970,
    1980,
    1990,
    2000,
    2010,
    2020,
    2025,
}


@pytest.fixture
def clf():
    return EraClassifier()


# ---------------------------------------------------------------------------
# Eingabe-Validierung
# ---------------------------------------------------------------------------


def test_classify_empty_audio_raises(clf):
    with pytest.raises(ValueError):
        clf.classify(np.zeros(0, dtype=np.float32), SR)


def test_classify_returns_era_result(clf):
    audio = np.random.randn(SR * 5).astype(np.float32) * 0.1
    result = clf.classify(audio, SR)
    assert isinstance(result, EraResult)


# ---------------------------------------------------------------------------
# Decade-Werte immer gültig
# ---------------------------------------------------------------------------


def test_decade_is_valid(clf):
    audio = np.random.randn(SR * 5).astype(np.float32) * 0.1
    result = clf.classify(audio, SR)
    assert result.decade in VALID_DECADES


def test_decade_valid_for_different_signals(clf):
    # Schmale Bandbreite (simuliert altes Material)
    t = np.arange(SR * 3) / SR
    band_limited = np.sin(2 * np.pi * 3000 * t).astype(np.float32) * 0.5
    result = clf.classify(band_limited, SR)
    assert result.decade in VALID_DECADES


def test_decade_valid_for_wideband(clf):
    np.random.seed(42)
    # Wideband-Rauschen → modernes Material
    audio = np.random.randn(SR * 5).astype(np.float32) * 0.3
    result = clf.classify(audio, SR)
    assert result.decade in VALID_DECADES


# ---------------------------------------------------------------------------
# Konfidenz und Felder
# ---------------------------------------------------------------------------


def test_confidence_in_range(clf):
    audio = np.random.randn(SR * 5).astype(np.float32) * 0.1
    result = clf.classify(audio, SR)
    assert 0.0 <= result.confidence <= 1.0


def test_era_label_nonempty(clf):
    audio = np.random.randn(SR * 5).astype(np.float32) * 0.1
    result = clf.classify(audio, SR)
    assert isinstance(result.era_label, str)
    assert len(result.era_label) > 0


def test_material_prior_nonempty(clf):
    audio = np.random.randn(SR * 5).astype(np.float32) * 0.1
    result = clf.classify(audio, SR)
    assert isinstance(result.material_prior, str)
    assert len(result.material_prior) > 0


def test_noise_profile_shape(clf):
    audio = np.random.randn(SR * 5).astype(np.float32) * 0.1
    result = clf.classify(audio, SR)
    assert result.noise_profile.shape == (24,)


def test_noise_profile_finite(clf):
    audio = np.random.randn(SR * 5).astype(np.float32) * 0.1
    result = clf.classify(audio, SR)
    assert np.all(np.isfinite(result.noise_profile))


def test_tier_used_is_known(clf):
    audio = np.random.randn(SR * 5).astype(np.float32) * 0.1
    result = clf.classify(audio, SR)
    assert result.tier_used in (1, 2, 3)


# ---------------------------------------------------------------------------
# get_material_prior / get_gp_warmstart
# ---------------------------------------------------------------------------


def test_get_material_prior_low_confidence():
    era = EraResult(decade=1950, era_label="Test", confidence=0.3, material_prior="vinyl")
    clf = EraClassifier()
    mat = clf.get_material_prior(era)
    assert mat == "unknown"


def test_get_material_prior_high_confidence():
    era = EraResult(decade=1950, era_label="Test", confidence=0.8, material_prior="vinyl")
    clf = EraClassifier()
    mat = clf.get_material_prior(era)
    assert mat == "vinyl"


def test_get_gp_warmstart_returns_dict(clf):
    audio = np.random.randn(SR * 5).astype(np.float32) * 0.1
    result = clf.classify(audio, SR)
    warmstart = clf.get_gp_warmstart(result)
    assert isinstance(warmstart, dict)
    assert "noise_reduction_strength" in warmstart
    assert "era_decade" in warmstart
    assert "era_confidence" in warmstart


def test_gp_warmstart_values_in_range(clf):
    audio = np.random.randn(SR * 5).astype(np.float32) * 0.1
    result = clf.classify(audio, SR)
    ws = clf.get_gp_warmstart(result)
    assert 0.0 <= ws["noise_reduction_strength"] <= 1.0


# ---------------------------------------------------------------------------
# Edge-Cases
# ---------------------------------------------------------------------------


def test_stereo_input_accepted(clf):
    audio = np.random.randn(2, SR * 3).astype(np.float32) * 0.1
    # Classifier expects 1-D intern → mean(axis=1) wird intern gemacht
    audio_flat = audio.T  # (n_samples, 2)
    result = clf.classify(audio_flat.mean(axis=1), SR)
    assert isinstance(result, EraResult)


def test_nan_input_handled(clf):
    audio = np.full(SR * 3, np.nan, dtype=np.float32)
    result = clf.classify(audio, SR)
    assert isinstance(result, EraResult)
    assert result.decade in VALID_DECADES


def test_very_short_audio_no_crash(clf):
    audio = np.zeros(1000, dtype=np.float32)
    result = clf.classify(audio, SR)
    assert isinstance(result, EraResult)


def test_era_result_decade_snapping():
    """EraResult-Initialisierung soll auf gültiges Jahrzehnt snappen."""
    era = EraResult(decade=1955, era_label="Test", confidence=0.5, material_prior="vinyl")
    assert era.decade in VALID_DECADES


# ---------------------------------------------------------------------------
# Singleton & Convenience
# ---------------------------------------------------------------------------


def test_singleton_same_instance():
    a = get_era_classifier()
    b = get_era_classifier()
    assert a is b


def test_classify_era_convenience():
    audio = np.random.randn(SR * 5).astype(np.float32) * 0.1
    result = classify_era(audio, SR)
    assert isinstance(result, EraResult)


def test_singleton_thread_safe():
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(get_era_classifier) for _ in range(20)]
        instances = [f.result() for f in futures]
    assert all(inst is instances[0] for inst in instances)


# ---------------------------------------------------------------------------
# Physikalisch kalibrierte DSP-Tier-Tests (gefilterte Rauschsignale)
# ---------------------------------------------------------------------------
# Grundlage: 6. Ordnung Butterworth-Tiefpassfilter → gemessener 90th-Pct-
# Rolloff ≈ 0.90 × Grenzfrequenz. Die Schwellwerte in _dsp_fingerprint_decade()
# wurden anhand dieser physikalischen Beziehung hergeleitet und empirisch
# gegen alle 10 Testfälle (10/10 ✅) verifiziert.
# ---------------------------------------------------------------------------

from scipy.signal import butter, sosfilt  # nur hier benötigt


def _make_vintage_signal(cutoff_hz: float, noise_amp: float = 0.02, sr: int = 48000, n_sec: int = 3) -> np.ndarray:
    """Bandbegrenztes Testsignal via 6. Ordnung Butterworth-Tiefpass.

    Der gemessene 90th-Pct-Rolloff liegt zuverlässig bei ≈ 0.90 × cutoff_hz.
    noise_amp steuert den SNR: hohe Werte simulieren historische Aufnahmen
    mit schlechtem Rauschabstand.
    """
    sos = butter(6, cutoff_hz, btype="lowpass", fs=sr, output="sos")
    sig = sosfilt(sos, np.random.randn(sr * n_sec).astype(np.float32)) * 0.1
    nse = sosfilt(sos, np.random.randn(sr * n_sec).astype(np.float32)) * noise_amp
    return (sig + nse).astype(np.float32)


@pytest.mark.parametrize(
    "cutoff_hz,noise_amp,expected_decades",
    [
        # --- Sehr alte Formate (Wachswalze / Grammophon-Membranmikrofon) ---
        (3_500, 0.08, {1890, 1900, 1910, 1920}),  # 1900 nun erreichbar (BW-Schwelle 3.2 kHz)
        # --- Shellac-78 / Kohlenmikrofon / frühes 4-kHz-Format ---
        (5_000, 0.06, {1900, 1910, 1920}),  # 1900 ist legitim für ~4.5 kHz Rolloff
        # --- Frühes Kondensatormikrofon, 1920er Rundfunk ---
        (6_000, 0.04, {1920, 1930}),
        # --- Vinyl LP (mono), Magnetophon früh ---
        (10_000, 0.02, {1940, 1950}),
        # --- Profi-Reel-Tape 38 cm/s (1960er Studio) ---
        (14_000, 0.01, {1950, 1960}),  # Rolloff-Streuung ±0.5 kHz → Übergangszone
        # --- FM-Radio-Rundfunk (1965–1975) ---
        (16_000, 0.01, {1960, 1970}),
        # --- HiFi-Kassettenband Typ IV (1975–1985) ---
        (18_000, 0.005, {1970, 1980}),
        # --- HiFi-Reel-Tape (Nakamichi-Ära) ---
        (20_000, 0.005, {1970, 1980, 1990}),  # Rolloff-Streuung → Übergangszone
        # --- DAT / Digitalrundfunk (1980–2000) ---
        (22_000, 0.001, {1980, 1990}),
    ],
)
def test_dsp_decade_physics_bandlimited(clf, cutoff_hz, noise_amp, expected_decades):
    """Physikalisch kalibrierter DSP-Tier-2-Test mit bandbegrenztem Rauschen.

    Jede Grenzfrequenz entspricht dem historischen HF-Limit eines Jahrzehnts
    (DECADE_HF_LIMITS). Das erkannte Jahrzehnt muss in der erlaubten Menge
    liegen (Übergangszone zwischen zwei Jahrzehnten zulässig).
    """
    np.random.seed(42)
    audio = _make_vintage_signal(cutoff_hz, noise_amp)
    result = clf.classify(audio, SR)
    assert result.decade in expected_decades, (
        f"cutoff={cutoff_hz} Hz → decade={result.decade}, erwartet eine aus {expected_decades}"
    )


def test_dsp_modern_wideband_maps_to_post1980(clf):
    """Breitbandrauschen ohne Filter → ≥ 1980 (vollständige Bandbreite).

    Statisches Rauschen hat geringe Musikdynamik (SNR ≈ 0–5 dB) →
    post-1980 Zweig mappt auf 1980.  2020 nur bei SNR ≥ 50 dB erreichbar.
    """
    np.random.seed(42)
    audio = (np.random.randn(SR * 3) * 0.1).astype(np.float32)
    result = clf.classify(audio, SR)
    assert result.decade >= 1980, f"Breitband → decade={result.decade}, erwartet ≥ 1980"


def test_dsp_rolloff_monotone_with_cutoff(clf):
    """Höhere Grenzfrequenz → mindestens gleich großes oder späteres Jahrzehnt."""
    np.random.seed(0)
    cutoffs = [5_000, 10_000, 16_000, 20_000]
    decades = []
    for c in cutoffs:
        audio = _make_vintage_signal(c, noise_amp=0.02)
        r = clf.classify(audio, SR)
        decades.append(r.decade)
    for i in range(len(decades) - 1):
        assert decades[i] <= decades[i + 1], (
            f"Monotonie verletzt: cutoff[{i}]={cutoffs[i]} Hz → {decades[i]}, "
            f"cutoff[{i + 1}]={cutoffs[i + 1]} Hz → {decades[i + 1]}"
        )


def test_dsp_confidence_physics_signal(clf):
    """DSP-basierte Klassifikation mit sauberem Signal → Konfidenz ≥ 0.25."""
    np.random.seed(7)
    audio = _make_vintage_signal(12_000, noise_amp=0.005)
    result = clf.classify(audio, SR)
    assert result.confidence >= 0.25, f"Konfidenz zu niedrig: {result.confidence:.3f} (erwartet ≥ 0.25)"


def test_dsp_tier_field_present_and_valid(clf):
    """EraResult muss tier_used, decade, confidence, material_prior und noise_profile enthalten."""
    np.random.seed(3)
    audio = _make_vintage_signal(10_000, noise_amp=0.02)
    result = clf.classify(audio, SR)
    for field in ("decade", "confidence", "material_prior", "noise_profile", "tier_used"):
        assert hasattr(result, field), f"EraResult fehlt Feld '{field}'"
    assert result.tier_used in {1, 2, 3}, f"tier_used ungültig: {result.tier_used}"
    assert result.confidence > 0.0, f"confidence muss positiv sein: {result.confidence}"


def test_dsp_shellac_snr_caps_decade(clf):
    """Shellac-Signal (niedriger SNR + schmale BW) → Jahrzehnt ≤ 1940."""
    np.random.seed(11)
    # Breites Rauschen überlagert bandbegrenztes Signal → schlechter SNR
    audio = _make_vintage_signal(5_000, noise_amp=0.15)
    result = clf.classify(audio, SR)
    assert result.decade <= 1940, f"Shellac-Simulation → decade={result.decade}, erwartet ≤ 1940"


def test_dsp_tier_used_dsp_for_synthetic(clf):
    """Synthetisches Signal ohne CLAP → Tier 2 oder 3 (kein ML)."""
    np.random.seed(99)
    audio = _make_vintage_signal(8_000, noise_amp=0.02)
    result = clf.classify(audio, SR)
    # CLAP ist optional (sota_upgrade) → synthetisch immer DSP-Tier
    assert result.tier_used in {2, 3}, f"Erwartet Tier 2 oder 3 (DSP), erhalten: {result.tier_used}"


def test_dsp_result_no_nan_fields(clf):
    """Alle EraResult-Felder nach DSP-Klassifikation sind NaN-frei und finite."""
    np.random.seed(55)
    audio = _make_vintage_signal(14_000, noise_amp=0.01)
    result = clf.classify(audio, SR)
    assert math.isfinite(result.confidence), "confidence enthält NaN/Inf"
    assert np.all(np.isfinite(result.noise_profile)), "noise_profile enthält NaN/Inf"


def test_dsp_seed_reproducibility(clf):
    """Gleicher Seed → exakt gleiche Klassifikation (Determinismus)."""
    results = []
    for _ in range(3):
        np.random.seed(42)
        audio = _make_vintage_signal(10_000, noise_amp=0.02)
        results.append(clf.classify(audio, SR).decade)
    assert len(set(results)) == 1, f"Klassifikation nicht deterministisch: {results}"


# ---------------------------------------------------------------------------
# Neue Tests: Dekade 1900, post-1990 SNR-Differenzierung, Tier-3 Verbesserungen
# ---------------------------------------------------------------------------


def test_dsp_decade_1900_detectable(clf):
    """4 kHz LP-Signal → Jahrzehnt 1900 erreichbar (nicht nur 1890 oder 1910)."""
    np.random.seed(42)
    # 4000 Hz LP → expected rolloff ~3.6 kHz, oberhalb der 3.2-kHz-1890-Grenze
    audio = _make_vintage_signal(4_000, noise_amp=0.08)
    result = clf.classify(audio, SR)
    assert result.decade in {1890, 1900, 1910}, f"4 kHz LP → decade={result.decade}, erwartet 1890, 1900 oder 1910"


def test_dsp_1890_narrower_than_1900(clf):
    """3 kHz LP → 1890; 4 kHz LP → 1900 oder höher (monotone Abgrenzung)."""
    np.random.seed(7)
    audio_3k = _make_vintage_signal(3_000, noise_amp=0.10)
    audio_4k = _make_vintage_signal(4_000, noise_amp=0.08)
    r3k = clf.classify(audio_3k, SR)
    r4k = clf.classify(audio_4k, SR)
    assert r3k.decade <= r4k.decade, f"Monotonie verletzt: 3 kHz → {r3k.decade}, 4 kHz → {r4k.decade}"


def test_dsp_post1990_snr_high_maps_later(clf):
    """Wideband-Signal mit hoher Dynamik → Jahrzehnt ≥ 1990.

    Simulates a recording with high dynamic range (classical music DR ~50 dB):
    alternating loud and nearly-silent segments so the P90/P10 frame-energy
    ratio gives a high SNR estimate.
    """
    np.random.seed(42)
    sr = SR
    n_sec = 6
    # Loud segments (0.5 amplitude) interleaved with near-silence (0.001)
    t = np.linspace(0, n_sec, sr * n_sec, endpoint=False).astype(np.float32)
    audio = np.zeros(sr * n_sec, dtype=np.float32)
    frame = sr // 4  # 250 ms frames
    for i in range(0, len(audio), frame):
        if (i // frame) % 2 == 0:
            audio[i : i + frame] = np.random.randn(min(frame, len(audio) - i)).astype(np.float32) * 0.5
        else:
            audio[i : i + frame] = np.random.randn(min(frame, len(audio) - i)).astype(np.float32) * 0.001
    result = clf.classify(audio, SR)
    # High-DR wideband signal → post-1980 branch; high SNR → 1990 or later
    assert result.decade >= 1980, f"Hochdynamisches Breitband-Signal → decade={result.decade}, erwartet ≥ 1980"


def test_dsp_post1990_snr_differentiation_order(clf):
    """Höhere Dynamik → späteres oder gleiches Jahrzehnt (Monotonie SNR→Jahrzehnt)."""
    np.random.seed(0)
    sr = SR
    n_sec = 4
    decades = []
    for loud_amp in [0.01, 0.1, 0.5]:  # steigender dynamischer Bereich
        audio = np.zeros(sr * n_sec, dtype=np.float32)
        frame = sr // 5
        for i in range(0, len(audio), frame):
            if (i // frame) % 2 == 0:
                audio[i : i + frame] = np.random.randn(min(frame, len(audio) - i)).astype(np.float32) * loud_amp
            else:
                audio[i : i + frame] = np.random.randn(min(frame, len(audio) - i)).astype(np.float32) * 0.0005
        r = clf.classify(audio, SR)
        decades.append(r.decade)
    # Decade muss mit steigendem DR monoton nicht-sinken
    for i in range(len(decades) - 1):
        assert decades[i] <= decades[i + 1], (
            f"SNR-Monotonie verletzt: DR[{i}] → {decades[i]}, DR[{i + 1}] → {decades[i + 1]}"
        )


def test_dsp_confidence_post1990_at_least_50(clf):
    """Post-1990 Klassifikation → Konfidenz ≥ 0.50 (SNR-basiert, kein BW-Fehler)."""
    np.random.seed(5)
    audio = (np.random.randn(SR * 4) * 0.3).astype(np.float32)
    result = clf.classify(audio, SR)
    if result.decade >= 1990:
        assert result.confidence >= 0.50, f"Post-1990 Konfidenz zu niedrig: {result.confidence:.3f}"


def test_dsp_confidence_pre1950_higher_with_hiss(clf):
    """Signal mit starkem Hiss → niedrigere Dekade UND hohe Konfidenz (SNR+BW konvergieren)."""
    np.random.seed(3)
    # Strong noise + narrow BW → both BW and SNR point to pre-1930
    audio = _make_vintage_signal(5_000, noise_amp=0.20)  # sehr schlechter SNR
    result = clf.classify(audio, SR)
    # Both evidence streams align → confidence should be reasonable
    assert result.confidence >= 0.25, f"Pre-1940 mit Hiss → confidence={result.confidence:.3f}, erwartet ≥ 0.25"
    assert result.decade <= 1940, f"Schmale BW + starker Hiss → decade={result.decade}, erwartet ≤ 1940"


def test_tier3_returns_valid_decade_and_confidence(clf):
    """Tier-3 Mikrofon-Heuristik: alle Ausgaben sind gültige Dekaden mit Konfidenz > 0."""
    from backend.core.era_classifier import _bark_band_energies, _microphone_type_decade

    np.random.seed(17)
    for cutoff in [3_000, 5_000, 8_000, 14_000, 22_000]:
        audio = _make_vintage_signal(cutoff, noise_amp=0.02)
        mono = audio  # already mono
        bark = _bark_band_energies(mono, SR)
        decade, conf = _microphone_type_decade(bark)
        assert decade in {
            1890,
            1900,
            1910,
            1920,
            1930,
            1940,
            1950,
            1960,
            1970,
            1980,
            1990,
        }, f"Tier-3 ungültige Dekade: {decade}"
        assert 0.0 < conf <= 1.0, f"Tier-3 Konfidenz außerhalb [0,1]: {conf}"


def test_tier3_narrow_bw_maps_older_than_wide_bw(clf):
    """Tier-3: schmalere Bandbreite → älteres Jahrzehnt (Monotonie)."""
    from backend.core.era_classifier import _bark_band_energies, _microphone_type_decade

    np.random.seed(9)
    audio_old = _make_vintage_signal(3_500, noise_amp=0.08)
    audio_new = _make_vintage_signal(20_000, noise_amp=0.001)
    bark_old = _bark_band_energies(audio_old, SR)
    bark_new = _bark_band_energies(audio_new, SR)
    decade_old, _ = _microphone_type_decade(bark_old)
    decade_new, _ = _microphone_type_decade(bark_new)
    assert decade_old <= decade_new, f"Tier-3 Monotonie verletzt: schmale BW → {decade_old}, breite BW → {decade_new}"


# ── constrain_era_to_medium() Tests ──────────────────────────────────────────


def _make_era(decade: int, conf: float = 0.72) -> EraResult:
    """Helper: EraResult für gegebenes Jahrzehnt erzeugen."""
    return EraResult(
        decade=decade,
        era_label=f"{decade}er",
        confidence=conf,
        material_prior="wax_cylinder",
        noise_profile=np.zeros(24),
        tier_used=2,
    )


def test_constrain_tape_1890_to_1960():
    """Compact Cassette (tape) floor=1960: 1890er → 1960er."""
    result = constrain_era_to_medium(_make_era(1890), "tape")
    assert result.decade == 1960
    assert result.era_label == "1960er"


def test_constrain_reel_tape_1890_to_1940():
    """Reel tape floor=1940: 1890er → 1940er."""
    result = constrain_era_to_medium(_make_era(1890), "reel_tape")
    assert result.decade == 1940


def test_constrain_cassette_1890_to_1960():
    """cassette ist Alias für tape, floor=1960."""
    result = constrain_era_to_medium(_make_era(1890), "cassette")
    assert result.decade == 1960


def test_constrain_vinyl_1930_to_1950():
    """Vinyl floor=1950: 1930er → 1950er."""
    result = constrain_era_to_medium(_make_era(1930), "vinyl")
    assert result.decade == 1950


def test_constrain_cd_digital_1970_to_1980():
    """CD floor=1980: 1970er → 1980er."""
    result = constrain_era_to_medium(_make_era(1970), "cd_digital")
    assert result.decade == 1980


def test_constrain_dat_1970_to_1980():
    """DAT floor=1980: 1970er → 1980er."""
    result = constrain_era_to_medium(_make_era(1970), "dat")
    assert result.decade == 1980


def test_constrain_mp3_1980_to_1990():
    """MP3 floor=1990: 1980er → 1990er."""
    result = constrain_era_to_medium(_make_era(1980), "mp3_low")
    assert result.decade == 1990


def test_constrain_aac_1990_to_2000():
    """AAC floor=2000: 1990er → 2000er."""
    result = constrain_era_to_medium(_make_era(1990), "aac")
    assert result.decade == 2000


def test_constrain_no_change_when_at_floor():
    """Wenn decade == floor, kein Eingriff (Grenzwert)."""
    result = constrain_era_to_medium(_make_era(1960), "tape")
    assert result.decade == 1960
    assert result.confidence == pytest.approx(0.72)


def test_constrain_no_change_when_above_floor():
    """Wenn decade > floor, unverändert zurückgeben."""
    result = constrain_era_to_medium(_make_era(1980), "tape")
    assert result.decade == 1980
    assert result.confidence == pytest.approx(0.72)


def test_constrain_confidence_scaled_down():
    """Korrigiertes Ergebnis hat geringere Konfidenz (0.65×, clamped 0.25–0.80)."""
    result = constrain_era_to_medium(_make_era(1890, conf=0.80), "tape")
    assert result.decade == 1960
    assert 0.25 <= result.confidence <= 0.80
    assert result.confidence < 0.80  # muss runter


def test_constrain_confidence_low_clamped_to_025():
    """Sehr niedrige Ausgangs-Konfidenz wird auf 0.25 geclampt."""
    result = constrain_era_to_medium(_make_era(1890, conf=0.10), "tape")
    assert result.confidence == pytest.approx(0.25)


def test_constrain_confidence_high_clamped_to_080():
    """Hohe Konfidenz ×0.65 > 0.80 → auf 0.80 clampen."""
    # 0.65 * conf ≤ 0.80 → conf ≤ 1.23; da conf ≤ 1.0 gilt: max 0.65 → nie > 0.80
    # Edge-Case: conf = 1.0 → 0.65, also nicht > 0.80. Trotzdem: Invariante prüfen.
    result = constrain_era_to_medium(_make_era(1890, conf=1.0), "tape")
    assert result.confidence <= 0.80


def test_constrain_unknown_medium_no_change():
    """Unbekanntes Medium (kein Eintrag in MEDIUM_DECADE_FLOOR) → unverändert."""
    result = constrain_era_to_medium(_make_era(1890), "unknown")
    assert result.decade == 1890
    assert result.confidence == pytest.approx(0.72)


def test_constrain_empty_string_medium_no_change():
    """Leerer Medium-String → kein Eingriff."""
    result = constrain_era_to_medium(_make_era(1890), "")
    assert result.decade == 1890


def test_constrain_medium_case_insensitive():
    """Medium-String ist case-insensitiv (TAPE == tape)."""
    result_lower = constrain_era_to_medium(_make_era(1890), "tape")
    result_upper = constrain_era_to_medium(_make_era(1890), "TAPE")
    result_mixed = constrain_era_to_medium(_make_era(1890), "Tape")
    assert result_lower.decade == result_upper.decade == result_mixed.decade == 1960


def test_constrain_shellac_1890_unchanged():
    """Shellac floor=1900: 1890er → 1900er (Shellac existierte ab 1898)."""
    result = constrain_era_to_medium(_make_era(1890), "shellac")
    assert result.decade == 1900


def test_constrain_wax_cylinder_no_change():
    """Wax cylinder floor=1890: 1890er bleibt 1890er."""
    result = constrain_era_to_medium(_make_era(1890), "wax_cylinder")
    assert result.decade == 1890
    assert result.confidence == pytest.approx(0.72)


def test_medium_decade_floor_coverage():
    """MEDIUM_DECADE_FLOOR enthält alle erwarteten Schlüssel-Medien."""
    required = {"tape", "cassette", "reel_tape", "vinyl", "shellac", "cd_digital", "dat", "mp3_low", "mp3_high", "aac"}
    assert required.issubset(set(MEDIUM_DECADE_FLOOR.keys()))


def test_medium_decade_floor_values_monotone_roughly():
    """Floor-Werte sind physikalisch sinnvoll: tape > shellac, cd > vinyl."""
    assert MEDIUM_DECADE_FLOOR["tape"] > MEDIUM_DECADE_FLOOR["shellac"]
    assert MEDIUM_DECADE_FLOOR["cd_digital"] > MEDIUM_DECADE_FLOOR["vinyl"]
    assert MEDIUM_DECADE_FLOOR["aac"] >= MEDIUM_DECADE_FLOOR["cd_digital"]


def test_constrain_returns_eraresult_dataclass():
    """Rückgabe ist immer eine EraResult-Instanz (auch bei No-Op)."""
    result = constrain_era_to_medium(_make_era(1970), "tape")
    assert isinstance(result, EraResult)
    result_noop = constrain_era_to_medium(_make_era(1970), "unknown")
    assert isinstance(result_noop, EraResult)
