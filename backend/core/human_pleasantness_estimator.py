"""
§v10 Human Pleasantness Estimator (HPE) — „Klingt das angenehm für menschliche Ohren?"

Der HPE ist die fehlende Brücke zwischen Auriks technischen Metriken und
dem, worauf es wirklich ankommt: menschlichem Wohlklang.

Während PMGG auf „Ähnlichkeit zum Original" optimiert (preservation),
optimiert der HPE auf PSYCHOAKUSTISCHE ANGENEHMHEIT — unabhängig davon,
ob das Ergebnis vom Original abweicht. Eine sauberere, klarere Aufnahme
DARF anders klingen als das verrauschte Original.

Wissenschaftliche Basis (ISO 532, Zwicker/Fastl, ANSI S3.4):
- Sharpness (Zwicker): acum/Zwicker — zu scharf = unangenehm
- Roughness (Fastl): asper/Zwicker — Modulation 15-300Hz = rau
- Tonalness: Verhältnis tonale/rauschartige Komponenten
- Loudness (Moore/Glasberg): sone/ISO 532 — zu laut = unangenehm
- Fluctuation Strength: vacil/Zwicker — Modulation <20Hz = schwankend

Composite Score P ∈ [0.0, 1.0] wo 1.0 = maximal angenehm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PleasantnessResult:
    """Ergebnis der psychoakustischen Angenehmheits-Prüfung."""

    score: float  # 0.0 = sehr unangenehm, 1.0 = sehr angenehm
    sharpness_zwicker: float  # acum (≤ 2 = angenehm, > 3 = scharf)
    roughness_asper: float  # asper (≤ 1 = glatt, > 2 = rau)
    loudness_sone: float  # sone (10-25 = angenehm für Musik)
    tonalness: float  # 0-1 (höher = tonaler = angenehmer)
    fluctuation_vacil: float  # vacil (≤ 1 = stabil, > 2 = schwankend)

    # Interpretation
    label: str = ""  # "Sehr angenehm", "Angenehm", "Neutral", "Anstrengend"
    issues: list[str] = field(default_factory=list)
    recommendation: str = ""


def compute_pleasantness(
    audio: np.ndarray,
    sr: int,
    *,
    original_audio: np.ndarray | None = None,
) -> PleasantnessResult:
    """Berechnet den psychoakustischen Angenehmheits-Score.

    Args:
        audio:           Mono oder Stereo Audio
        sr:              Sample-Rate (muss ≥ 44100 sein)
        original_audio:  Optionales Original zum Vergleich

    Returns:
        PleasantnessResult mit Score und Detail-Werten
    """
    arr = np.asarray(audio, dtype=np.float64)
    # Mono für Analyse
    if arr.ndim == 2:
        mono = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)
    else:
        mono = arr
    mono = np.atleast_1d(mono).ravel()

    # ── 1. Sharpness (Zwicker, acum) ──
    sharpness = _compute_zwicker_sharpness(mono, sr)

    # ── 2. Roughness (Fastl, asper) ──
    roughness = _compute_roughness(mono, sr)

    # ── 3. Loudness (vereinfachtes Moore/Glasberg via RMS + Gewichtung) ──
    loudness_sone = _estimate_loudness_sone(mono, sr)

    # ── 4. Tonalness ──
    tonalness = _compute_tonalness(mono, sr)

    # ── 5. Fluctuation Strength ──
    fluctuation = _compute_fluctuation_strength(mono, sr)

    # ── Composite Score ──
    # Gewichte: Sharpness und Roughness sind die dominanten Faktoren
    # für „angenehm/unangenehm" (Zwicker & Fastl 2007, Kap. 11)
    s_sharp = max(0.0, 1.0 - (sharpness - 1.5) / 2.5)  # sharpness < 1.5 → score=1, > 4 → score=0
    s_rough = max(0.0, 1.0 - roughness / 2.0)           # roughness < 0.5 → score=1, > 2 → score=0
    s_loud = max(0.0, 1.0 - abs(loudness_sone - 18.0) / 15.0)  # optimal bei ~18 sone
    s_tonal = tonalness                                   # tonalness direkt als Score
    s_fluct = max(0.0, 1.0 - fluctuation / 2.0)

    score = float(np.clip(
        s_sharp * 0.30 + s_rough * 0.25 + s_loud * 0.15 + s_tonal * 0.20 + s_fluct * 0.10,
        0.0, 1.0
    ))

    # ── Label & Issues ──
    issues = []
    if sharpness > 3.0:
        issues.append(f"Zu scharf ({sharpness:.1f} acum) — Höhen abdämpfen")
    elif sharpness < 1.0:
        issues.append(f"Zu dumpf ({sharpness:.1f} acum) — Höhen anheben")
    if roughness > 1.5:
        issues.append(f"Zu rau ({roughness:.1f} asper) — Modulation glätten")
    if loudness_sone > 30:
        issues.append(f"Zu laut ({loudness_sone:.0f} sone) — Pegel senken")
    elif loudness_sone < 8:
        issues.append(f"Zu leise ({loudness_sone:.0f} sone) — Pegel anheben")
    if tonalness < 0.3:
        issues.append(f"Zu rauschhaft ({tonalness:.2f}) — mehr tonale Anteile erwünscht")

    if score >= 0.75:
        label = "Sehr angenehm"
        rec = "Keine Änderungen nötig — für menschliche Ohren optimiert."
    elif score >= 0.55:
        label = "Angenehm"
        rec = "Leichte Optimierungen möglich — " + (issues[0] if issues else "insgesamt gut.")
    elif score >= 0.35:
        label = "Neutral"
        rec = "Spürbare Verbesserung möglich — " + (issues[0] if issues else "Details prüfen.")
    else:
        label = "Anstrengend"
        rec = "Deutliche Überarbeitung nötig — " + (issues[0] if issues else "alle Dimensionen prüfen.")

    return PleasantnessResult(
        score=score,
        sharpness_zwicker=sharpness,
        roughness_asper=roughness,
        loudness_sone=loudness_sone,
        tonalness=tonalness,
        fluctuation_vacil=fluctuation,
        label=label,
        issues=issues,
        recommendation=rec,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Psychoakustische Messfunktionen
# ═══════════════════════════════════════════════════════════════════════════

def _compute_zwicker_sharpness(mono: np.ndarray, sr: int) -> float:
    """Zwicker Sharpness (acum): gewichtetes Verhältnis hoher zu tiefer Frequenzen.

    Vereinfachte Zwicker-Methode: Bark-Skala mit Gewichtungsfunktion g(z).
    Sharpness S = 0.11 * ∫ N'(z) * g(z) * z dz / ∫ N'(z) dz
    """
    n_fft = 4096
    if len(mono) < n_fft:
        return 1.5  # Default für kurze Signale

    # Spektrum
    spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # Bark-Skala (24 Bänder, Zwicker)
    bark_edges = np.array([0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270,
                           1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300,
                           6400, 7700, 9500, 12000, 15500])

    specific_loudness = np.zeros(len(bark_edges) - 1)
    for i in range(len(bark_edges) - 1):
        mask = (freqs >= bark_edges[i]) & (freqs < bark_edges[i + 1])
        if mask.sum() > 0:
            specific_loudness[i] = np.mean(spec[mask])

    # Zwicker-Gewichtung g(z)
    z_values = np.arange(1, len(bark_edges))
    g_z = np.ones_like(z_values, dtype=float)
    g_z[z_values > 15] = 0.15 * z_values[z_values > 15] - 1.25  # g(z) steigt ab Bark 16

    numerator = np.sum(specific_loudness * g_z * z_values)
    denominator = np.sum(specific_loudness) + 1e-12

    sharpness = 0.11 * numerator / denominator
    return float(np.clip(sharpness, 0.5, 5.0))


def _compute_roughness(mono: np.ndarray, sr: int) -> float:
    """Roughness (asper): Modulation 15-300 Hz im Signal.

    Vereinfacht: RMS-Varianz in 25ms-Fenstern als Proxy für Rauigkeit.
    """
    win = int(0.025 * sr)  # 25ms
    if len(mono) < 2 * win:
        return 0.5

    rms_vals = []
    for i in range(0, len(mono) - win, win // 2):
        chunk = mono[i:i + win]
        rms_vals.append(float(np.sqrt(np.mean(chunk ** 2))))

    rms_vals = np.array(rms_vals)
    if len(rms_vals) < 4:
        return 0.5

    # Roughness ≈ Varianz der RMS / mittlere RMS
    rms_mean = np.mean(rms_vals) + 1e-12
    rms_var = np.var(rms_vals)
    roughness = float(np.clip(rms_var / (rms_mean ** 2) * 10.0, 0.1, 5.0))
    return roughness


def _estimate_loudness_sone(mono: np.ndarray, sr: int) -> float:
    """Vereinfachte Loudness in Sone (ISO 532 via RMS + Frequenzgewichtung).

    Genauere Implementierung in dsp/psychoacoustics.py (Zwicker/ISO 532-1).
    """
    rms = float(np.sqrt(np.mean(mono ** 2)) + 1e-12)
    rms_db = 20.0 * np.log10(rms)

    # Pegel → Sone (vereinfacht: 1 sone = 40 phon bei 1 kHz)
    # 0 dBFS ≈ 100 dB SPL → pegeldB = dBFS + 100
    spl_estimate = rms_db + 100.0

    # ISO 226 Näherung: phon → sone
    if spl_estimate < 40:
        sone = 0.0
    else:
        sone = 2.0 ** ((spl_estimate - 40.0) / 10.0)

    return float(np.clip(sone, 0.0, 100.0))


def _compute_tonalness(mono: np.ndarray, sr: int) -> float:
    """Tonalness: Verhältnis tonaler zu rauschartigen Komponenten.

    Hohe Tonalness = klare Tonhöhen, angenehm.
    Niedrige Tonalness = rauschdominiert, weniger angenehm.
    """
    n_fft = 4096
    if len(mono) < n_fft:
        return 0.5

    spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))
    spec_db = 20.0 * np.log10(np.maximum(spec, 1e-10))

    # Finde Peaks (tonale Komponenten)
    peaks = []
    for i in range(2, len(spec_db) - 2):
        if spec_db[i] > spec_db[i - 1] and spec_db[i] > spec_db[i + 1] and spec_db[i] > spec_db[i - 2] + 3 and spec_db[i] > spec_db[i + 2] + 3:
            peaks.append(spec_db[i])

    if not peaks:
        return 0.1  # Keine klaren Töne

    # Tonalness = Energie in Peaks / Gesamtenergie
    peak_energy = np.sum(10 ** (np.array(peaks) / 10))
    total_energy = np.sum(10 ** (spec_db / 10)) + 1e-12

    tonalness = float(np.clip(peak_energy / total_energy * 3.0, 0.0, 1.0))
    return tonalness


def _compute_fluctuation_strength(mono: np.ndarray, sr: int) -> float:
    """Fluctuation Strength (vacil): Modulation < 20 Hz.

    Hohe Fluktuation = unangenehmes „Wabern".
    """
    win = int(1.0 * sr)  # 1s Fenster
    if len(mono) < 2 * win:
        return 0.5

    # RMS-Verlauf in 1s-Schritten
    rms_timeline = []
    for i in range(0, len(mono) - win, win):
        chunk = mono[i:i + win]
        rms_timeline.append(float(np.sqrt(np.mean(chunk ** 2))))

    if len(rms_timeline) < 3:
        return 0.5

    rms_timeline = np.array(rms_timeline)
    rms_db = 20.0 * np.log10(rms_timeline + 1e-12)

    # Fluktuation = Varianz des RMS über 1-Sekunden-Blöcke
    fluctuation = float(np.clip(np.std(rms_db) / 3.0, 0.1, 5.0))
    return fluctuation


# ═══════════════════════════════════════════════════════════════════════════
# Vergleich: Original vs. Restauriert
# ═══════════════════════════════════════════════════════════════════════════

def compare_pleasantness(
    original: np.ndarray,
    restored: np.ndarray,
    sr: int,
) -> dict:
    """Vergleicht die Angenehmheit von Original und restauriertem Audio.

    Gibt zurück, OB und WARUM das restaurierte Audio angenehmer (oder weniger
    angenehm) als das Original ist.

    Returns:
        dict mit 'improved', 'delta_score', 'original', 'restored', 'verdict'
    """
    orig = compute_pleasantness(original, sr)
    rest = compute_pleasantness(restored, sr)

    delta = rest.score - orig.score
    improved = delta > 0.03  # Mindestschwelle für „hörbare Verbesserung"

    if improved and rest.score >= 0.70:
        verdict = "Deutlich angenehmer — die Restaurierung verbessert den Höreindruck spürbar."
    elif improved:
        verdict = "Etwas angenehmer — leichte Verbesserung des Höreindrucks."
    elif delta > -0.03:
        verdict = "Kein signifikanter Unterschied in der Angenehmheit."
    else:
        verdict = f"Weniger angenehm — die Restaurierung könnte überbearbeitet sein. {rest.recommendation}"

    return {
        "improved": improved,
        "delta_score": float(delta),
        "original": {"score": orig.score, "label": orig.label, "issues": orig.issues},
        "restored": {"score": rest.score, "label": rest.label, "issues": rest.issues},
        "verdict": verdict,
    }
