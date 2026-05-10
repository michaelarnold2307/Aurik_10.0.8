"""§2.64 v9.12.2 — Unit-Tests für _fast_goal_snapshot Proxy-Kalibrierung.

Schützt vor Regression der drei systemischen Proxy-Bugs, die zu persistenten
zu niedrigen Delta-Werten für ALLE Import-Songs führten:

1. Single-Segment-Bias: spektrale Proxys nur auf Zentrum-Segment → jetzt Multi-Segment.
2. authentizitaet-ACF-Crash nach phase_24: ACF auf ersten 8192 Samples → jetzt Zentral-Drittel.
3. transparenz-Proxy: Single-Segment-Perzentil-SNR → jetzt Vollsignal + SFM-Blend.
"""

import numpy as np

# Importpfad: _fast_goal_snapshot ist @staticmethod in UnifiedRestorerV3
# Wir importieren nur die statische Methode, nicht den vollen UV3.
from backend.core.unified_restorer_v3 import UnifiedRestorerV3

_SR = 48000
_snap = UnifiedRestorerV3._fast_goal_snapshot


def _make_tonal_audio(n: int = _SR * 4, sr: int = _SR) -> np.ndarray:
    """Tonales Testsignal: Summe harmonischer Sinustöne (simuliert Musik)."""
    t = np.arange(n) / sr
    sig = (
        0.5 * np.sin(2 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2 * np.pi * 440.0 * t)
        + 0.15 * np.sin(2 * np.pi * 880.0 * t)
        + 0.05 * np.sin(2 * np.pi * 1760.0 * t)
    ).astype(np.float32)
    return sig / (np.max(np.abs(sig)) + 1e-8)


def _make_noisy_audio(n: int = _SR * 4, sr: int = _SR) -> np.ndarray:
    """Rausch-Signal: weißes Rauschen (simuliert defektes Audio)."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(n) * 0.3).astype(np.float32)


def _make_dropout_repaired_audio(n: int = _SR * 4, sr: int = _SR) -> np.ndarray:
    """Simuliert audio NACH phase_24 Dropout-Repair:
    - Intro (erste 0.5 s): inpainted (linearer Ramp — andere ACF-Struktur)
    - Rest: normales tonales Signal
    """
    sig = _make_tonal_audio(n, sr)
    # Überschreibe Anfang mit Inpainting-ähnlichem Material (linearer Ramp)
    dropout_len = int(0.5 * sr)
    sig[:dropout_len] = np.linspace(0.0, sig[dropout_len], dropout_len).astype(np.float32)
    return sig


def _make_compressed_pop(n: int = _SR * 4, sr: int = _SR) -> np.ndarray:
    """Simuliert komprimiertes Pop/Schlager: dichtes harmonisches Signal mit hohem RMS,
    kein Dynamikabfall (wie typischer Schlager nach Mastering).
    """
    t = np.arange(n) / sr
    # Dichte harmonische Schichtung simuliert Pop-Mix
    sig = sum(
        amp * np.sin(2 * np.pi * freq * t)
        for freq, amp in [(110, 0.4), (220, 0.35), (440, 0.3), (880, 0.25), (1760, 0.2), (3520, 0.15)]
    )
    sig = np.array(sig, dtype=np.float32)
    # Harte Begrenzung simuliert Mastering-Kompression (hohe 5th-Perzentile ≈ 0.3)
    sig = np.clip(sig, -0.8, 0.8)
    return (sig / (np.max(np.abs(sig)) + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# Test 1: Natuerlichkeit-Proxy — tonales Signal > Rauschen (Multi-Segment-fix)
# ---------------------------------------------------------------------------


class TestNatuerlichkeitProxy:
    def test_tonal_higher_than_noise(self):
        """Tonales Signal muss deutlich höhere natuerlichkeit haben als Rauschen."""
        tonal = _make_tonal_audio()
        noisy = _make_noisy_audio()
        s_tonal = _snap(tonal, _SR)
        s_noisy = _snap(noisy, _SR)
        assert "natuerlichkeit" in s_tonal
        assert s_tonal["natuerlichkeit"] > s_noisy["natuerlichkeit"] + 0.15, (
            f"Tonal={s_tonal['natuerlichkeit']:.3f} sollte > Noise={s_noisy['natuerlichkeit']:.3f} + 0.15"
        )

    def test_multi_segment_stability(self):
        """Pause im Zentrum-Segment darf natuerlichkeit nicht kollabieren.

        §2.64 v9.12.2: Single-Segment-Bug — Pause im Zentrum gab natuerlichkeit ≈ 0.02.
        Multi-Segment-Fix: 25%/50%/75% → stabile Messung auch bei Zentrum-Pause.
        """
        sig = _make_tonal_audio()
        # Zentriertes Stille-Segment (genau dort, wo Single-Segment messen würde)
        center = len(sig) // 2
        pause_len = 4096
        sig_with_pause = sig.copy()
        sig_with_pause[center - pause_len // 2 : center + pause_len // 2] = 0.0
        s = _snap(sig_with_pause, _SR)
        assert s["natuerlichkeit"] > 0.25, (
            f"natuerlichkeit={s['natuerlichkeit']:.3f} trotz Zentrum-Pause "
            f"(Multi-Segment-Fix muss Tonalität aus 25%/75%-Segmenten erkennen)"
        )


# ---------------------------------------------------------------------------
# Test 2: Authentizitaet-Proxy — Crash nach phase_24-Inpainting verhindert
# ---------------------------------------------------------------------------


class TestAuthentizitaetProxy:
    def test_dropout_repaired_intro_doesnt_crash_acf(self):
        """§2.64 v9.12.2: ACF auf Zentral-Drittel vermeidet phase_24-Inpainting-Crash.

        Alt: ACF auf mono[:8192] — nach Dropout-Repair am Intro kollabiert Peak 0.71→0.06.
        Neu: ACF auf mono[N//3 : N//3+8192] — stabiler Bereich, kein Absturz.
        """
        sig_repaired = _make_dropout_repaired_audio()
        sig_clean = _make_tonal_audio()
        s_rep = _snap(sig_repaired, _SR)
        s_clean = _snap(sig_clean, _SR)
        # Repaired signal sollte ähnliche authentizitaet haben wie sauberes Signal
        # (nicht um >0.30 darunter liegen — das war der Bug)
        assert "authentizitaet" in s_rep
        gap = s_clean["authentizitaet"] - s_rep["authentizitaet"]
        assert gap < 0.30, (
            f"authentizitaet-Gap nach Dropout-Repair-Intro: clean={s_clean['authentizitaet']:.3f} "
            f"repaired={s_rep['authentizitaet']:.3f} gap={gap:.3f} — Regression des §2.64 v9.12.2 Fixes!"
        )

    def test_authentizitaet_voiced_signal(self):
        """Harmonisches Signal muss authentizitaet > 0.5 haben."""
        sig = _make_tonal_audio()
        s = _snap(sig, _SR)
        assert s["authentizitaet"] > 0.50, f"authentizitaet={s['authentizitaet']:.3f} für tonales Signal — sollte > 0.5"


# ---------------------------------------------------------------------------
# Test 3: Transparenz-Proxy — komprimiertes Pop/Schlager darf nicht auf ≤ 0.20 kollabieren
# ---------------------------------------------------------------------------


class TestTransparenzProxy:
    def test_compressed_pop_not_near_zero(self):
        """§2.64 v9.12.2: transparenz-Proxy darf für komprimiertes Pop nicht ≤ 0.20 sein.

        Alt: Single-Segment 5th/99th-Perzentil-SNR → bei loud Passage 5th-Pct ≈ 0.2,
        peak ≈ 0.25 → log10(1.25)/5 ≈ 0.02 → transparenz=0.02. Bug!
        Neu: Vollsignal-Perzentile + SFM-Blend → stabiler Wert > 0.20 für Musik.
        """
        pop = _make_compressed_pop()
        s = _snap(pop, _SR)
        assert "transparenz" in s
        assert s["transparenz"] >= 0.20, (
            f"transparenz={s['transparenz']:.3f} — Vollsignal-Proxy-Fix muss ≥ 0.20 liefern "
            f"für komprimiertes Pop/Schlager"
        )

    def test_tonal_higher_than_noise(self):
        """Tonales Signal (klar) muss höhere transparenz haben als Weißrauschen."""
        tonal = _make_tonal_audio()
        noisy = _make_noisy_audio()
        s_t = _snap(tonal, _SR)
        s_n = _snap(noisy, _SR)
        assert s_t["transparenz"] >= s_n["transparenz"], (
            f"Tonal={s_t['transparenz']:.3f} sollte >= Noise={s_n['transparenz']:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 4: Brillanz-Proxy — HF-reiches Material > HF-armes Material
# ---------------------------------------------------------------------------


class TestBrillanzProxy:
    def test_hf_rich_higher_than_hf_poor(self):
        """Helles Signal (HF > 4 kHz) muss höhere brillanz haben als dunkles."""
        t = np.arange(_SR * 4) / _SR
        hf_rich = (0.5 * np.sin(2 * np.pi * 8000.0 * t) + 0.3 * np.sin(2 * np.pi * 12000.0 * t)).astype(np.float32)
        hf_poor = (0.5 * np.sin(2 * np.pi * 100.0 * t) + 0.3 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
        s_hf = _snap(hf_rich, _SR)
        s_lf = _snap(hf_poor, _SR)
        assert s_hf["brillanz"] > s_lf["brillanz"] + 0.10, (
            f"HF-reich={s_hf['brillanz']:.3f} sollte > LF-arm={s_lf['brillanz']:.3f} + 0.10"
        )


# ---------------------------------------------------------------------------
# Test 5: Vollständigkeit — alle 14 Goal-Schlüssel vorhanden
# ---------------------------------------------------------------------------


class TestSnapshotCompleteness:
    def test_all_goals_present(self):
        """_fast_goal_snapshot muss alle 14 Musical-Goal-Schlüssel liefern."""
        sig = _make_tonal_audio()
        s = _snap(sig, _SR)
        required_keys = {
            "natuerlichkeit",
            "authentizitaet",
            "timbre_authentizitaet",
            "tonal_center",
            "artikulation",
            "emotionalitaet",
            "mikrodynamik",
            "groove",
            "transparenz",
            "waerme",
            "basskraft",
            "sep_fidelity",
            "brillanz",
            "raumtiefe",
        }
        missing = required_keys - set(s.keys())
        assert not missing, f"Fehlende Goal-Schlüssel: {missing}"

    def test_all_values_in_range(self):
        """Alle Proxy-Werte müssen in [0, 1] liegen."""
        sig = _make_tonal_audio()
        s = _snap(sig, _SR)
        out_of_range = {k: v for k, v in s.items() if not (0.0 <= v <= 1.0)}
        assert not out_of_range, f"Proxy-Werte außerhalb [0,1]: {out_of_range}"

    def test_short_audio_returns_empty(self):
        """Zu kurzes Audio (< 512 Samples) muss leeres Dict zurückgeben."""
        s = _snap(np.zeros(256, dtype=np.float32), _SR)
        assert s == {}, f"Kurzes Audio muss {{}} liefern, aber: {s}"

    def test_stereo_no_crash(self):
        """Stereo-Input (2, N) darf nicht crashen."""
        stereo = np.stack([_make_tonal_audio(), _make_tonal_audio() * 0.9])  # (2, N)
        s = _snap(stereo, _SR)
        assert isinstance(s, dict)
        assert len(s) >= 10
