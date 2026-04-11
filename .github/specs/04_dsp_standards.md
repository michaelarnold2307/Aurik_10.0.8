
# Aurik 9 — Spec 04: DSP-Standards & SOTA-Algorithmen
>
> Psychoakustische Fundierung, SOTA-Entscheidungsmatrix, Pflicht-Algorithmen.
> Algorithmen ab 2018 als Minimum. Legacy-Algorithmen als Primärverarbeitung VERBOTEN.

## §4.1 Pflicht-Konzepte (mindestens eines pro DSP-Funktion)

| Konzept | Anwendung | Referenz |
| --- | --- | --- |
| **OMLSA / IMCRA** | Rauschunterdrückung stationär | Cohen (2002, 2003) |
| **Consistent Wiener Filter** | Spektrale Restaurierung (modernisierter Wiener) | Le Roux & Vincent (2013) |
| **PGHI** | Phasenkonsistenz nach Spektral-Modifikation | Perraudin et al. (2013) |
| **NMF mit β-Divergenz** | Spektrale Dekomposition, Inpainting | Févotte & Idier (2011) |
| **pYIN / SPICE** | Pitch-Tracking f₀ | Mauch & Dixon (2014) |
| **RBME** | Decrackle, Spectral Smoothing | Bando et al. (2019) |
| **Sinusoidal + Stochastic Modeling** | Dropout-Inpainting | Serra & Smith (1990) |
| **DTW / Optimal Transport** | Zeitausrichtung, Pitch-Korrektur | Cuturi (2013) |
| **Multi-Resolution STFT (MRSA)** | Adaptive Fenstergrößen | Bello et al. (2005) |
| **Psychoacoustic Masking** | Restaurierungs-Regler | ISO 11172-3 |
| **Harmonic Lattice (Fletcher)** | Inharmonizität-konsistente Partials | Fletcher (1964) |
| **Beat-Tracking (madmom)** | Phrasen-Kontext für Inpainting | Böck et al. (2016) |
| **DDSP (Eigenimplementierung)** | Instrument-Resonanzmodellierung | Engel et al. (ICLR 2020) |
| **ASA (Bregman)** | Common Onset, Common Fate | Bregman (1990) |
| **Virtual Pitch** | Missing Fundamental → BassKraftMetric | Moore et al. (2006) |
| **ISO 226:2023 Equal-Loudness** | BrillanzMetric/WaermeMetric-Gewichtung | ISO 226:2023 |
| **LUFS / ITU-R BS.1770-5** | Lautstärkenormalisierung (2023) | ITU-R BS.1770-5 |

> **DDSP-Implementierung**: Aurik nutzt eine leichtgewichtige NumPy/SciPy-Eigenimplementierung (`dsp/ddsp_synth.py`) — **KEIN** Google-`ddsp`-PyPI-Paket (benötigt TensorFlow). Die Eigenimplementierung deckt additive Synthese + Rauschfilter vollständig ab und ist out-of-the-box ohne TF lauffähig.

### §4.1a Sample-Rate-Vertrag (Dual-SR, [RELEASE_MUST])

- Analyse-/Klassifikations-Module arbeiten mit nativer Import-SR (`analysis_sr`).
- DSP/ML-Verarbeitungsstufen arbeiten strikt mit `processing_sr = 48000`.
- Resampling-Fehler in Richtung 48 kHz sind harte Fehler (fail-fast), kein Silent-Fallback auf Nicht-48k-Processing.
- Komponenten mit modellnativer SR duerfen intern resamplen (z. B. 48k -> 44.1k -> 48k), aber nur innerhalb der Komponente und mit Rueckgabe in `processing_sr=48000`.

---

## §4.1b [RELEASE_MUST] Psychoakustische Lautheitsmessung nach ISO 532-1 (Zwicker/Fastl)

**Warum LUFS nicht ausreicht**: LUFS (ITU-R BS.1770-5) verwendet K-Weighting (~A-Gewichtung +
Hochregal +4 dB). K-Weighting detektiert Tieftonrumpeln (200–300 Hz) nicht als Lautheitszunahme,
die das Gehör mit bis zu +6 Phon wahrnimmt (ISO 226:2023 Equal-Loudness-Contours). Ergebnis:
Rumble-Filter-Phasen werden bei LUFS-Only-Check fälschlich als lautheitsneutral eingestuft.

**MidPipeline-Guard nach subtraktiven Phasen mit großem Breitband-Impact**:

```python
# backend/core/dsp/psychoacoustics.py
def compute_specific_loudness_zwicker(audio: np.ndarray, sr: int) -> float:
    """
    ISO 532-1 stationäre Methode (Zwicker/Fastl).
    Returns total loudness N in sone.
    - Bark-Filterbank: 24 kritische Bänder (0–16 kHz)
    - Fletcher-Munson-Kurven: ISO 226:2023 Tabelle
    - Referenz: 1 sone = 40 phon bei 1 kHz
    """
```

**ΔN-Entscheidungstabelle** (Δ = output_sone - input_sone):

| ΔN (sone) | Wahrnehmung | Pipeline-Reaktion |
| --- | --- | --- |
| ≤ 0.5 | Lautheitsneutral | OK |
| 0.5 – 1.0 | Grenzbereich | INFO in `metadata["loudness_delta_sone"]` |
| 1.0 – 2.0 | Wahrnehmbar lauter | WARNING in `metadata` + im PhaseConductor-State |
| > 2.0 | Deutliche Verfälschung | FAIL → Dry/Wet-Anteil erhöhen (per-Phase trocken beimischen), kein Harter Rollback |

**Mapping sone → phon** für Diagnose: `phon = 40 + 33.2 × log10(max(N, 0.001) / 1.0)` (gültig für N ≥ 1 sone)

**Implementierungspfad**:

- Modul: `backend/core/dsp/psychoacoustics.py` (neu)
- Aufruf: `backend/core/unified_restorer_v3.py` — §2.45a MidPipeline-Guard nach breitbandigen subtraktiven Phasen
- Methode: Stationäre ISO 532-1 (not zeitvariant) — ausreichend für Pipeline-Check, Laufzeit ≤ 50 ms / 5-s-Fenster
- Approximation: 24 Butterworth-Bandpass-Filter (Bark-Skala), nicht physikalische Cochlea-Simulation

**Rückwärtskompatibilität**: Ergänzt §2.45a (Gated-RMS-Guard) — LUFS-Check bleibt erhalten als Broadcast-Metrik.
LUFS = Distribution-Standard; Sone = psychoakustische Lästigkeitsmetrik. Beide sind Pflicht.

---

## §4.2 Verbotene Legacy-Algorithmen als Primärverarbeitung

```python
# ABSOLUT VERBOTEN als Haupt-Algorithmus:
Ephraim & Malah (1984) Wiener-Filter   # → Ersatz: OMLSA/IMCRA
Klassischer Wiener-Filter               # → Ersatz: Consistent Wiener (Le Roux 2013)
Simple Spectral Subtraction             # → Ersatz: MMSE-LSA + OMLSA
Medianfilter-Declicker (primitiv)       # → Ersatz: RBME + iterative Konsistenz
YIN Pitch-Tracker                       # → Ersatz: pYIN / CREPE
LPC Ordnung < 16                        # → Ersatz: High-Order LPC + Burg-Algorithmus
np.fft.rfft ohne PGHI nach Modifikation  # → PGHI zwingend
griffin_lim() als Phasengenerator (Endschritt)  # → PGHI / Vocos / HiFi-GAN (Griffin-Lim randomisiert Phasen → IPD-Verlust → Raumtiefe kollabiert, §8.3.1)
RMS-Normalisierung statt LUFS           # → ITU-R BS.1770-5
Peak-Normalisierung bei Restaurierung   # → LUFS + True-Peak

# VERBOTENE METRIKEN für Musikqualitätsbewertung:
PESQ    # Telefonband 300–3400 Hz
DNSMOS  # 16 kHz Sprachkorpus
NISQA   # Sprach-CNN, keine Musik-Daten
STOI    # Sprachverständlichkeit 150–5000 Hz
ViSQOL --speech  # Voice-Priors → Musik systematisch falsch bewertet
```

---

## §4.3 Frequenzband-Referenzen

```python
BARK_EDGES_HZ = [
    20, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
    1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400,
    5300, 6400, 7700, 9500, 12000, 15500
]
def hz_to_erb(f_hz: float) -> float:
    return 21.4 * math.log10(1.0 + f_hz / 229.0)
def hz_to_mel(f_hz: float) -> float:
    return 2595.0 * math.log10(1.0 + f_hz / 700.0)
```

---

## §4.4 SOTA-ML-Entscheidungsmatrix (Stand 2025/2026)

| Anwendungsfall | Primär (SOTA) | DSP-Fallback (Post-2018) | VERBOTEN |
| --- | --- | --- | --- |
| Breitrauschen (Gesang/Vokal) | ML: **DeepFilterNet v3.II** (Formant-/F0-Struktur nutzt Gesangs-Harmonik; energy_bias=−6 dB Pflicht) | OMLSA/IMCRA | ~~Wiener 1984~~ |
| Breitrauschen (rein instrumental, PANNs Vocals < 0.4) | DSP: **OMLSA/IMCRA** (kein Vocal-Prior, musik-neutral) | DeepFilterNet v3.II + energy_bias=−9 dB | ~~Wiener 1984~~ |
| Nicht-stationäres Rauschen | ML: **DeepFilterNet v3.II** | MMSE-LSA + IMCRA | ~~Spectral Subtraction~~ |
| Diffuses Raumrauschen / Dereverb | ML: **SGMSE+** (ONNX) | WPE (nara_wpe) → NumPy-WPE → OMLSA | ~~einfacher Bandpass~~ |
| Stem-Separation Vocals | ML: **MelBandRoformer** (`bs_roformer_plugin`, 860 MB ONNX) | NMF-β | — |
| Stem-Separation Instrumental | ML: **MDX23C** (`mdx23c_plugin`, Kim_Vocal_2/Kim_Inst) | NMF-β | — |
| Bandbreiten-Erweiterung | ML: **AudioSR** | Sinusoidal + Stoch. Modeling | ~~Harmonics-EQ~~ |
| Dropout < 50 ms | DSP: **NMF-β + Sinusoidal** | Consistent Wiener | ~~Yule-Walker AR~~ |
| Dropout 50–999 ms | ML: **CQTdiff+** → DiffWave | Spectral Interpolation | ~~einfaches AR~~ |
| Codec-Artefakte | ML: **Apollo** (Band-Sequence Mamba) | Spectral Repair + PGHI | ~~EQ-Anhebung~~ |
| Pitch-Tracking (mono/Gesang) | ML: **FCPE** → CREPE → RMVPE (nur wenn stabil verifiziert) | PESTO → pYIN | ~~YIN~~ |
| Polyphoner Pitch | ML: **BasicPitch** | Spektrale Peak-Verfolgung | ~~CREPE mono~~ |
| Instrument-Resonanz | DSP: **DDSP** (Eigenimpl.) | Sinusoidal + Stoch. | ~~fixe Formant-EQ~~ |
| Formanten F1–F4 | ML: **DeepFormants CNN** (ONNX) | LPC (Burg, Ordnung **30–40 bei 48 kHz-SR**, alternativ: Downsampling auf 16 kHz → LPC Ord. 16 → Upsampling) | ~~LPC < 12~~ |
| Neuronale Synthese | ML: **Vocos 48 kHz nativ** ONNX → 44,1 kHz → BigVGAN v2 → HiFi-GAN | PGHI-ISTFT | ~~Griffin-Lim~~ |
| Generatives Inpainting | ML: **Flow Matching** | CQTdiff+ → DiffWave → NMF-β | ~~DDPM 1000 Schritte~~ |
| Audio-Tagging | ML: **BEATs** (iter3) → PANNs CNN14 | DSP Spectral Fingerprint | — |
| MOS (ohne Referenz) | ML: **VERSA** → SingMOS (Gesang, PANNs Vocals ≥ 0.3–0.7, Blend-Zone) | PQS-Gammatone-DSP | ~~PESQ/DNSMOS/CDPAM~~ |
| MOS-Verifikation Gesang | ML: **UTMOS** (`utmos_plugin`, ≥18 MB PyTorch) → SingMOS | VERSA → PQS-Gammatone | ~~PESQ/NISQA~~ |
| Music/Vocal Enhancement | ML: **MP-SENet 2023** ONNX | SGMSE+ ONNX → OMLSA DSP | ~~DCCRN/FullSubNet+~~ |
| MOS (mit Referenz) | ML: **ViSQOL v3** (**`--audio` PFLICHT**) | PQS-DSP | ~~--speech Mode~~ |
| Phasen-Rekonstruktion | DSP: **PGHI** | Griffin-Lim ≥ 32 Iter. | ~~Direkte ISTFT~~ |
| Decrackle | DSP: **RBME + iterative Konsistenz** | Sparse Bayes | ~~Medianfilter~~ |
| Spektral-Matching | DSP: **Optimal Transport** | Multibänder-EQ | ~~fixe EQ-Kurve~~ |
| Groove / Timing | DSP: **Onset-DTW (madmom RNN)** | Beat-Tracking (librosa) | ~~fixes Raster~~ |

**MP-SENet ONNX Laufzeitvertrag (Release-Must):**

- Eingabeformat bleibt `noisy_amp/noisy_pha: [batch, 201, time]`.
- Das aktuell gebündelte ONNX-Artefakt zeigt trotz `dynamic_axes` einen faktisch festen Zeit-Shape (`time=32`) in internen Reshape-Knoten.
- Implementierungspflicht: segmentierte Inferenz in festen 32-Frame-Chunks mit Stitching zurück auf Originallänge.
- Bei Reshape-/Layout-Fehlern: kontrollierter Fallback (Layout-Retry, dann OMLSA-DSP), niemals Hard-Crash.

---

## §4.5 Pflicht-Algorithmus-Spezifikationen

### Rauschunterdrückung (Phase 03, 29)

```text
Pflicht: OMLSA (Cohen & Berdugo 2002) + IMCRA-Variante (Cohen 2003)
Gain-Glättung: MMSE-LSA
G_floor: 0.85 an HPG-protected_bins, 0.10 sonst

Stem-bewusstes NR-Routing (PANNs-konditioniert, nach TDP-Trennung):

  A) Perkussiver Stem (TDP audio_percussive):
     → KEIN NR, KEIN DeepFilterNet — nur phase_01 + phase_27 (Klick/Pop)
     Begründung: Transienten-Angriffe werden durch NR-Latenz vernichtet

  B) Harmonischer Stem MIT Gesang (PANNs Singing ≥ 0.4):
     PRIMÄR: DeepFilterNet v3.II (Formant-/F0-Struktur von Gesang nutzt
              die Harmonik-Modellierung des Netzes; NICHT weil Gesang = Sprache)
     Musik-Konfiguration:
         energy_bias = −6.0 dB (Pflicht — reduziert aggressive NR in Harmonik-Regionen)
         G_floor via HarmonicPreservationGuard (0.85 an Partial-Bins)
     FALLBACK: OMLSA/IMCRA + MMSE-LSA

  C) Harmonischer Stem REIN INSTRUMENTAL (PANNs Singing < 0.4):
     PRIMÄR: OMLSA/IMCRA (kein Vocal-Prior — musik-neutral, adaptiv)
     Begründung: DeepFilterNet würde harmonische Obertonstrukturen
                  (Streicher, Bläser, Chords) nach Vocal-Prior
                  systematisch als Rauschen fehlbewerten
     SEKUNDÄR: DeepFilterNet v3.II mit vergrößertem energy_bias = −9.0 dB
               (nur wenn OMLSA SNR-Schätzung unsicher: IMCRA-SNR < 5 dB)

Kritische Architekturnotiz (Stand April 2026):
    Es existiert kein öffentlich verfügbares, auf Musikdaten (Gesang + Instrumente)
    trainiertes neuronales Denoise-Modell in DeepFilterNet-Qualitätsstufe.
    OMLSA/IMCRA ist daher für rein instrumentales Material de-facto co-primär,
    nicht nur Fallback. DeepFilterNet wird NUR bei erkanntem Gesang (PANNs Singing
    ≥ 0.4) als Primär eingesetzt — NICHT für gesprochene Sprache (Aurik restauriert
    Musik, keine Podcasts/Hörbücher). Diese Einschätzung ist bis zum Erscheinen
    eines musik-spezifischen ML-Denoiser-Modells verbindlich.
```

### Inpainting (Phase 24, 55)

```text
Kurze Lücken < 50 ms: NMF mit β-Divergenz (β=0, Itakura-Saito) + PGHI
    β-Wert-Referenz (normativ, Lücke-F-Fix v9.10.100):
        β=0 → Itakura-Saito-Divergenz (IS): minimiert relative Energiefehler,
               gewichtet kleine Energieunterschiede stark → OPTIMAL für impulsive
               Lücken (Klicks < 50 ms) und Transienten (Attack-Bins).
        β=1 → Kullback-Leibler-Divergenz (KL): bessere Approximation bei harmonischen,
               tonal-stationären Lücken — NICHT für impulsive Transienten verwenden.
        β=2 → Euklidisch: nur für breitbandige stationäre Rekonstruktion.
    Transient-Lücken (< 50 ms, Onset-Kontext): β=0 (IS) — PFLICHT (Artikel-/MicroDyn-Schutz)
    Harmonische/tonale Lücken (< 50 ms, Stationär-Kontext): β=1 (KL) erlaubt.
    VERBOTEN: β=1 für kurze Transient-Lücken (zerstört Attack-Energie-Verteilung).
Lange Lücken 50–999 ms: CQTdiff+ (Moliner & Välimäki, ICASSP 2023)
    - CQT-Domänen-Diffusion konditioniert auf Phrasen-Kontext ±30 s
    - PGHI für phasenkonsistente Rücktransformation
    Fallback: CQTdiff+ → DiffWave → NMF-β (β=0)
VERBOTEN: VoiceFixer v2 (Sprach-only, VCTK-Korpus)
VERBOTEN: VampNet (kein gebündeltes Plugin, kein stabiler ONNX-Export)
```

### Codec-Artefakte (Phase 23, 50)

```text
Pflicht: Apollo (Zhang et al. 2024)
    - Band-Splitting-RNN: Audio → 24 Sub-Bänder
    - Mamba-Backbone-Sequenzmodellierung
    - Musical Goals Check post-Rekonstruktion (Brillanz ≥ 0.85, Wärme ≥ 0.80)
Fallback: Resemble-Enhance ONNX → DSP Spectral Repair + PGHI
```

### Amplituden-Clipping (Phase 23 — CLIPPING-Typ) §4.5a [RELEASE_MUST]

```text
Diskriminierung: classify_clipping() MUSS VOR Reparatur ausgeführt werden.
    SOFT_SATURATION → Phase 23 ÜBERSPRINGEN (§5 Vintage-Guard)
    CLIPPING         → ADMM-Declipping

Primär: ADMM-basierte Sparse-Recovery (Záviška et al., EUSIPCO 2021)
    Modell: geclippte Abtastwerte y_c = clip(x, -A, +A)
    Minimierungsproblem:
        min_x  ||x||_1   s.t.   x_free = y_free   (ungesättigte Stellen exakt)
                                |x_clip| >= A       (gesättigte Stellen ≥ Clipwert)
    ADMM-Parameter:
        rho     = 0.1        (Penalty, adaptiv: rho *= 1.5 wenn Residuum > 10× Dual)
        max_iter = 200       (typisch 80–120 Konvergenz)
        tol     = 1e-4       (Primal- und Duales Residuum unter Schwelle)
    Wavelet-Prior (Daubechies db4, Level 5) als Sparsifying-Transform
    Frequenzband: 20 Hz – 20 kHz (keine Sub-Bass-Beschränkung)
    Transient-Guard: Attack-Bins (onset ±5 ms) erhalten rho × 3.0 → stärkerer Erhalt

Fallback: Consistent Wiener Filter mit Clip-Constraints
    → Spectral Repair + PGHI falls beide fehlschlagen

INVARIANTE: Nach ADMM-Declipping MUSS Transient-Shape-Korrelation
    mit pre-clipping Signal ≥ 0.88 (ArticulationMetric-Grenzwert).
    Unterschreitung → Safety-Blend mit Dry-Signal (max 30 % Wet).

VERBOTEN: Apollo für Amplituden-Clipping (Apollo = Codec-Artefakte-Training).
    Apollo NUR für digitale Codec-Komprimierungsartefakte (mp3/aac/ogg-Distortion).
```

### Print-Through-Reduktion (Phase 29, reel_tape)

```text
Pflicht: Bidirektionale Adaptive Temporal Subtraction (LMS-basiert, Widrow & Stearns 1985)
Physikalisches Modell: Print-Through entsteht auf BEIDEN Seiten der Masterwicklung
    (Kopie VOR dem Original = Pre-Echo bei Vorwärtswicklung,
     Kopie NACH dem Original = Post-Echo bei Rückwärtswicklung),
     Amplitude nimmt nichtlinear mit Wicklungsabstand ab (α_pre ≠ α_post).
Schritte:
    1. Kreuzkorrelation-Peak ±600 ms → delay_pre, delay_post (beide Seiten)
    2. LMS-Adaptivfilter separat für Pre- und Post-Echo:
       alpha_pre  ∈ [0.03, 0.25]  (schwächeres Prä-Echo)
       alpha_post ∈ [0.05, 0.35]  (stärkeres Post-Echo)
    3. audio_clean[t] = audio[t]
                        − alpha_pre  · audio[t + delay_pre]
                        − alpha_post · audio[t − delay_post]
    4. Spectral Coherence vor/nach ≥ 0.90 + PGHI
Fallback: NMF-β Dekomposition (einseitig, nur Post-Echo)
VERBOTEN: Comb-Filter, einseitiges α-Modell als Pflicht-Implementierung
```

### Multi-Resolution STFT (Phase 03, 06, 07, 23, 50)

```python
# MRSA-Fenster @ SR=48000 Hz:
ZONES = {
    "sub_bass":   {"win": 65536, "hop": 16384, "hz": (20, 250)},
    "mid_low":    {"win": 16384, "hop": 4096,  "hz": (250, 800)},
    "mid":        {"win": 8192,  "hop": 2048,  "hz": (800, 2000)},
    "presence":   {"win": 1024,  "hop": 256,   "hz": (2000, 8000)},
    "air":        {"win": 128,   "hop": 32,    "hz": (8000, 24000)},
}
# PGHI per Zone; Kreuzfade Hanning 10 ms an Zonenübergängen
```

### Neuronale Synthese / Vocoder-Kaskade (wenn PQS-MOS < 4.3)

```text
4-stufige Fallback-Kaskade (Studio-2026):
    1. Vocos 48 kHz nativ — vocos_mel_spec_48khz.onnx (CPUExecutionProvider)
       Mel-Bins 80; True-Peak −1.0 dBTP nach Synthese
       # VERBOTEN: vocos_mel_spec_24khz.onnx (24-kHz-Variante — SR-Mismatch zu processing_sr=48000)
    2. BigVGAN-v2 — bigvgan_v2 (0,4 GB, ONNX/PyTorch, GPU-beschleunigt via ml_device_manager)
    3. HiFi-GAN (3,6 MB ONNX) — Tertiär-Fallback
    4. PGHI-ISTFT — DSP-Endfall-Fallback
VERBOTEN: Griffin-Lim als Endschritt in Studio-2026

```

### Pit-Korrektur (Phase 12, 31)

```text
Primär: FCPE → CREPE → RMVPE (nur wenn stabil verifiziert) + DTW
Bei Gesang (PANNs Vocals ≥ 0.4):
    PSOLA (Moulines & Charpentier 1990) — formanterhaltend bei Transposition > ±2 Halbton
    Phase-Vocoder: nur für perkussive / nicht-vokale Segmente (HPSS-detektiert)
DSP-Fallback: PESTO (Riou et al. ISMIR 2023) → pYIN (Mauch & Dixon 2014)
```

### Klavier-Inharmonizität §4.5b (Phase 52) [RELEASE_MUST]

```text
Physikalisches Modell (Fletcher 1964):
    f_n = n · f_0 · sqrt(1 + B · n²)
    B = Inharmonizitätskoeffizient (abhängig von Saitenlänge, Durchmesser, Material)

B-Schätzung per Oktavlage (aus MIDI-Pitch oder f₀-Track):
    MIDI 21–35  (Kontra-Oktave,  21–62 Hz):  B ∈ [0.00005, 0.0003]
    MIDI 36–47  (Groß-Oktave,    65–247 Hz): B ∈ [0.0003,  0.001]
    MIDI 48–71  (Klein/1-Oktave, 130–987 Hz):B ∈ [0.001,   0.003]
    MIDI 72–108 (2–5-Oktave,     1047–4186 Hz):B ∈ [0.002, 0.008]

Schätzung (wenn kein MIDI-Input):
    1. f₀ per FCPE/CREPE → nächster MIDI-Pitch
    2. Partials f_2, f_3, f_4 aus Spektrum detektieren
    3. B = mean([(f_n / (n · f_0))² - 1] / n²)  für n = 2..4
    4. B im zulässigen Bereich für die Oktavlage klemmen (s. o.)

Anwendung in phase_52:
    - Partial-Lock: Obertöne auf Fletcher-Raster statt harmonisches int-Vielfaches
    - Resynthese (DDSP additive): f_n = n · f_0 · sqrt(1 + B · n²)
    - VERBOTEN: Partial-Locking auf exakte ganzzahlige Vielfache bei Klaviermaterial
      (ergibt zu hartes/synthetisches Ergebnis)

INVARIANTE: TimbralAuthenticityMetric ≥ 0.87 nach phase_52.
    B-Schätz-Fehler > 15 % → DSP-Fallback (sinusoidales Partial-Tracking ohne Lock).
```

### Dereverb — Early-Reflection-Guard §4.5c (Phase 20, 49) [RELEASE_MUST]

```text
Physikalische Grundlage:
    Schallfeld = Direktschall + frühe Reflexionen (0–80 ms) + diffuser Nachhall (> 80 ms)
    Frühe Reflexionen (0–50 ms): definieren Raumcharakter, Wärme, Quellenlokalisierung
    (Haas-Effekt / Precedence Effect — Blauert 1997)
    Diffuser Nachhall (> 80 ms): maskierend, verschleiert Transienten, Artikulation

Pflicht-Algorithmus (SGMSE+ und WPE):
    1. RIR-Schätzung (Blind): Kreuzkorrelations-Dekonvolution oder
       ML-RT60-Schätzer (DPRNN-Head in SGMSE+ oder spectral-based RT60)
    2. Early-Reflection-Guard: C80 und D50 VOR und NACH Dereverb messen
           C80 = 10 · log10(∫₀⁸⁰ₘₛ h²(t)dt / ∫₈₀ₘₛ^∞ h²(t)dt)   [Klarheitsmaß, Musik]
           D50 = ∫₀⁵⁰ₘₛ h²(t)dt / ∫₀^∞ h²(t)dt                    [Deutlichkeitsmaß — sekundär; C80 hat Vorrang für Musik]
    3. Wet-Mix-Begrenzung: Dereverb-Intensität MUSS begrenzt werden, wenn:
           ΔC80 = C80_post − C80_pre > 6 dB  → Wet weiter reduzieren
           ΔD50 > 0.12                         → Wet weiter reduzieren
       Ziel: ΔC80 ≤ 6 dB, ΔD50 ≤ 0.12 (kein Raumcharakter-Verlust)
    4. Early-Reflection-Blend: Residual der ersten 50 ms aus Originalschall
       (vor Dereverb) mit alpha = 0.35 zurückzumischen falls ΔC80 > 4 dB:
           audio_out[t] = audio_dereverb[t] + 0.35 · audio_early[t]  (t ∈ [0, 50ms] jedes Onset)
    5. Safety-Revert: Wenn C80_post < C80_pre − 2 dB → vollständiger Rollback

Messung (einfache Approximation ohne RIR-Inversion):
    RT60-Proxy: Energie-Abfall von −5 dBFS auf −35 dBFS in Stille-Segmenten (Silero VAD)
    C80-Proxy: Energie-Ratio 80 ms / Rest in Onset-Fenstern (madmom Onset-Detektion)

INVARIANTE: NatuerlichkeitMetric darf nach phase_49 nicht sinken
    (Early-Reflection-Blend stellt Raumluft wieder her).
VERBOTEN: Aggressives Dereverb ohne Early-Reflection-Guard (klingt klinisch/steril).
```

### Phasen-Rekonstruktion (nach JEDER Spektral-Modifikation)

```text
PFLICHT: PGHI (Perraudin et al. 2013)
Fallback: Griffin-Lim ≥ 32 Iterationen
ABSOLUT VERBOTEN: Direkte ISTFT auf modifiziertem Betragsspektrum
```

**Ausnahmevertrag (eng begrenzt, RELEASE_MUST) — `phase_12_wow_flutter_fix` / Tape-Head-Level-v2:**

Direkte STFT→iSTFT-Rekonstruktion ist ausschließlich für den
`TAPE_HEAD_LEVEL_DIP`-Unterpfad in `phase_12` zulässig, wenn alle Bedingungen erfüllt sind:

1. Ziel ist frequenzabhängige Kopfkontakt-/Spacing-Loss-Kompensation (kein generatives Inpainting).
2. Linked-Stereo-Invariante: identische spektrale Gain-Maske für L und R.
3. Boundary-konsistent: SciPy-kompatibler Boundary-Modus (`even|odd|constant|zeros|None`),
   identisch zwischen STFT und iSTFT-Pfad.
4. Länge exakt konserviert (`len(out) == len(in)` via trim/pad).
5. SNR-Guard aktiv (Noise-Floor-nahe Bins nicht boosten) + `max_gain_db <= 15`.

**Verboten bleibt** direkte ISTFT für alle anderen spektralen Rekonstruktionsphasen.

**PGHI-Compliance-Status (Stand April 2026):**
Phasen mit PGHI-Rekonstruktion: `phase_03`, `phase_06`, `phase_20`, `phase_23`, `phase_24`, `phase_28`, `phase_29`, `phase_31`.
Alle anderen STFT-nutzenden Phasen verwenden kein modifiziertes Betragsspektrum (nur Analyse-STFT ohne Rücktransformation).

**[RELEASE_MUST] PGHI STFT/iSTFT-boundary-Invariante (v9.10.100):**
Jeder PGHI-Rekonstruktionsaufruf nach einer STFT-Modifikation MUSS `boundary`-konsistent sein:

```python
# PFLICHT-Regel: Externes STFT mit scipy-Standard (boundary='zeros') MUSS
# mit iSTFT boundary=True rekonstruiert werden.
# Internes STFT (PGHI-intern mit boundary=None) nutzt boundary=False in iSTFT.
#
# FALSCH (Amplitude am Rand ~1–3 dB zu niedrig, erste/letzte ~10 ms abgedämpft):
audio_out = pghi_reconstruct_from_stft(Zxx)  # ohne n_samples
# RICHTIG:
audio_out = pghi_reconstruct_from_stft(Zxx, n_samples=len(audio_in))
#   n_samples=: Exakte Längetreue durch Trim/Pad auf Originallänge.
#   dsp/pghi.py::_istft() erhält boundary=True wenn STFT extern (scipy-Standard)
#   und boundary=False nur für intern berechnete STFT-Frames.
#
# VERBOTEN: mode='edge' beim Padding von iSTFT-Kurzausgaben — klont gedämpften Endwert.
# PFLICHT: mode='constant', constant_values=0.0 (Stille statt Artefakt-Klon).
```

### Dithering (24→16 bit Export)

```text
PRIMÄR: POW-r Typ 3 (Wannamaker et al. 1992) — ~+6 dB effektiver SNR
FALLBACK: TPDF-Dithering (±1 LSB)
VERBOTEN: Truncation ohne Dithering
```

### MP3-Export

```text
PRIMÄR: LAME VBR V0 via FFmpeg (q:a=0) — adaptiv bis 320 kbps, ≈245 kbps Ø
VERBOTEN: CBR für Restaurierungsausgaben — CBR erzeugt Pre-Echo auf restaurierten
          Transienten (TDP §7.x, MDEM §8.3)
```

### Head-Bump Kompensation (Phase 04, Tape-Material) v9.10.128

Tape-Transport-Köpfe erzeugen eine LF-Resonanz, deren Mittenfrequenz umgekehrt proportional zur Aufnahmegeschwindigkeit ist (physik: Wellenlänge = Spaltlänge · 2 bei Resonanzfrequenz).

```text
KOMPENSATION: Parametrische Kerbfilter (IIR-Peaking, gain_db negativ) pro Bandgeschwindigkeit:
  1.875 ips (Kassette):  f_bump = 70 Hz,  cut = −2.5 dB, Q = 1.2
  3.75  ips:             f_bump = 90 Hz,  cut = −2.5 dB, Q = 1.2
  7.5   ips (Standard-Reel):  f_bump = 130 Hz, cut = −2.0 dB, Q = 1.3
  15    ips (Semi-Pro):  f_bump = 180 Hz, cut = −1.5 dB, Q = 1.4
  30    ips (Master):    f_bump = 250 Hz, cut = −1.0 dB, Q = 1.5

QUELLEN: Zar (1989) "Magnetic Recording Handbook"; Jorgensen (1996) "The Complete Handbook
         of Magnetic Recording"; White (2000) "The Recording Studio Handbook".
AKTIVIERUNG: Via kwargs["tape_speed_ips"] in phase_04.process() — optional, nur wenn Bandgeschwindigkeit bekannt.
FALLBACK: Keine Kompensation (kein crash); unbekannte Geschwindigkeit (>150% Abweichung) = bypass.
```

### Intermodulation Distortion — Bispektrum-Analyse (Phase 63) v9.10.128

Bispektrum-Kohärenz bestätigt IMD-Produkte kausal (Wishart 2013; Kim & Powers 1979):

```text
B(f1, f2) = E[X(f1) · X(f2) · conj(X(f1+f2))]

Normierte Bispektrum-Kohärenz:
  b_coherence = |B(f1,f2)| / (P(f1) · P(f2) · P(f1+f2))^(1/3)

Schwellwert: b_coherence > 0.15 → IMD-Produkt bestätigt (vs. Rauschen/Harmonik)

§2.51 STEREO-COMPLIANCE: Notch-Maske wird aus dem Mid-Signal berechnet und
symmetrisch auf Mid (100%) und Side (30%) angewendet. Kein unabhängiges L/R-Processing.
```

### Tonträgerketten-Charakteristika — BW-Grenzen (Phase 55) v9.10.128

```text
MATERIAL-SPEZIFISCHE INPAINTING-BANDBREITENBEGRENZUNG (§0 Primum non nocere):

  wax_cylinder  : ≤ 5000 Hz   (akustische Aufnahme, Horn-Resonator)
  wire_recording: ≤ 6000 Hz   (Stahldrähte 1940–1955, mechanische BW-Begrenzung)
  lacquer_disc  : ≤ 8000 Hz   (Schnitt-Spezifikation frühe Direktschnittplatten)

IMPLEMENTATION: Butterworth 4th-Order Tiefpass (sosfiltfilt, zero-phase) nach
  jedem Gap-Fill in _process_channel() — verhindert AR/Diffusion-Halluzination
  von HF-Inhalt, der im Quellmaterial nie vorhanden war.
```

---

## §12 Referenzen (Auswahl — Pflicht-Algorithmen)

- Cohen & Berdugo (2002): IMCRA — _Noise Estimation by Minima Controlled Recursive Averaging_
- Cohen (2003): OMLSA — _Noise Spectrum Estimation in Adverse Environments_
- Le Roux & Vincent (2013): _Consistent Wiener Filtering for Audio Source Separation_
- Perraudin et al. (2013): PGHI — _A Non-Iterative Method for STFT Phase (Re)construction_
- Févotte & Idier (2011): _Algorithms for NMF with the β-Divergence_
- Mauch & Dixon (2014): pYIN — _A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions_
- Fletcher (1964): _Normal Vibration Frequencies of a Stiff Piano String_
- Engel et al. (2020): DDSP — ICLR 2020 (Google Magenta)
- Bregman (1990): _Auditory Scene Analysis_ — MIT Press
- Moore, Glasberg & Baer (2006): Virtual Pitch — JASA
- Zhang et al. (2024): Apollo — _Band-sequence Modeling for High-Quality Music Restoration_
- Moliner & Välimäki (ICASSP 2023): CQTdiff+ — _Solving Audio Inverse Problems with a Diffusion Model_
- Lu et al. (2023): BS-RoFormer — _Music Source Separation with Band-Split RoPE Transformer_
- Siuzdak (2023): Vocos — _Resynthesizing Speech and Music with Neural Vocoders_
- Wannamaker et al. (1992): POW-r — _A Theory of Nonsubtractive Dither_
- Lipman et al. (2023): _Flow Matching for Generative Modeling_
- Radford et al. (2022): Whisper — _Robust Speech Recognition via Large-Scale Weak Supervision_
- ISO 226:2023: _Acoustics — Normal Equal-Loudness-Level Contours_
- ITU-R BS.1770-5 (2023): _Algorithms to measure audio programme loudness_
- Záviška et al. (2021): _A Proper Version of Synthesis-based Sparse Audio Declipper_ — EUSIPCO 2021
- Blauert (1997): _Spatial Hearing — The Psychophysics of Human Sound Localization_ — MIT Press
- Kuttruff (2009): _Room Acoustics_ — 5th Ed. (C80/D50 Definitionen §4.5c)

---

## §4.6 [RELEASE_MUST] Loudness-Guard DSP-Invarianten (v9.11.5)

DSP-Pflichtregeln für alle Loudness-Drift-Guards in der Pipeline (§2.45a).

### §4.6a Gated-RMS-Messung

```python
# VERBOTEN — globaler RMS misst Stille mit:
rms_db = 20.0 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-10)

# PFLICHT — Gated-RMS (nur musikalische Frames):
def _rms_dbfs_gated(audio, frame_size=2048, gate_dbfs=-50.0, min_gate_ratio=0.05):
    # Stereo → Mono downmix vor Framing
    if audio.ndim == 2:
        mono = (audio[0] + audio[1]) * 0.5
    else:
        mono = audio
    # Frame-basierte Messung
    n_frames = len(mono) // frame_size
    frames = mono[:n_frames * frame_size].reshape(n_frames, frame_size)
    frame_rms_db = 20.0 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-10)
    # Gate: nur Frames > gate_dbfs
    mask = frame_rms_db > gate_dbfs
    if mask.sum() < max(1, int(n_frames * min_gate_ratio)):
        return float(np.mean(frame_rms_db))  # Fallback: ungated
    return float(np.mean(frame_rms_db[mask]))
```

### §4.6b Envelope-Aware Gain

```python
# VERBOTEN — uniformer Gain amplifiziert Stille:
audio *= gain_factor

# PFLICHT — musik-selektiver Gain:
def _musical_gain_envelope(audio, gain, gate_dbfs=-50.0, sr=48000):
    # Gate-Envelope: 1.0 für Musik-Frames, 0.0 für Stille
    # Crossfade: 10 ms Hann-Smoothing an Gate-Übergängen
    # Gain-Formel pro Sample: out = audio * (1.0 + (gain - 1.0) * gate_envelope)
    # → Stille (gate=0): out = audio * 1.0 (unverändert)
    # → Musik (gate=1): out = audio * gain (verstärkt)
```

### §4.6c Soft-Limiter

```python
# VERBOTEN — unbedingter Soft-Limiter komprimiert Musikdynamik:
_abs = np.abs(audio)
_over = _abs > 0.92
audio = np.where(_over, np.sign(audio) * (0.92 + 0.08 * np.tanh((_abs - 0.92) / 0.08)), audio)

# PFLICHT — bedingter Soft-Limiter nur bei echtem Clipping-Risiko:
peak = float(np.max(np.abs(audio)))
if peak > 0.98:  # Nur bei echtem Clipping-Risiko
    _abs = np.abs(audio)
    _over = _abs > 0.92
    if np.any(_over):
        audio = np.where(_over, np.sign(audio) * (0.92 + 0.08 * np.tanh((_abs - 0.92) / 0.08)), audio)
audio = np.clip(audio, -1.0, 1.0)  # Finales Sicherheitsnetz
```

### Rationale

Diese drei Invarianten verhindern das Silence-Amplification-Problem: Subtraktive Phasen entfernen Rauschen aus Stille-Segmenten → globaler RMS sinkt → Guard meldet falschen Pegelkollaps → uniformer Makeup-Gain amplifiziert entrauschte Stille → Soft-Limiter komprimiert Musikpeaks. Ergebnis: Musik leiser, Stille lauter — exakt das Gegenteil des Ziels.

> Kreuzreferenz: Spec 02 §2.45a-I bis §2.45a-IV
