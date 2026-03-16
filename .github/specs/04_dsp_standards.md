# Aurik 9 — Spec 04: DSP-Standards & SOTA-Algorithmen

> Psychoakustische Fundierung, SOTA-Entscheidungsmatrix, Pflicht-Algorithmen.
> Algorithmen ab 2018 als Minimum. Legacy-Algorithmen als Primärverarbeitung VERBOTEN.

---

## §4.1 Pflicht-Konzepte (mindestens eines pro DSP-Funktion)

| Konzept | Anwendung | Referenz |
|---|---|---|
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

> **DDSP-Implementierung**: Aurik nutzt eine leichtgewichtige NumPy/SciPy-Eigenimplementierung
> (`dsp/ddsp_synth.py`) — **KEIN** Google-`ddsp`-PyPI-Paket (benötigt TensorFlow). Die Eigenimplementierung
> deckt additive Synthese + Rauschfilter vollständig ab und ist out-of-the-box ohne TF lauffähig.

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
|---|---|---|---|
| Breitrauschen | ML: **DeepFilterNet v3.II** | OMLSA/IMCRA | ~~Wiener 1984~~ |
| Nicht-stationäres Rauschen | ML: **DeepFilterNet v3.II** | MMSE-LSA + IMCRA | ~~Spectral Subtraction~~ |
| Diffuses Raumrauschen / Dereverb | ML: **SGMSE+** (ONNX) | WPE (nara_wpe) → NumPy-WPE → OMLSA | ~~einfacher Bandpass~~ |
| Stem-Separation | ML: **MDX23C** (Kim_Vocal_2/Kim_Inst) | NMF-β | — |
| Bandbreiten-Erweiterung | ML: **AudioSR** | Sinusoidal + Stoch. Modeling | ~~Harmonics-EQ~~ |
| Dropout < 50 ms | DSP: **NMF-β + Sinusoidal** | Consistent Wiener | ~~Yule-Walker AR~~ |
| Dropout 50–999 ms | ML: **CQTdiff+** → VampNet → DiffWave | NMF-β + Sinusoidal | ~~einfaches AR~~ |
| Codec-Artefakte | ML: **Apollo** (Band-Sequence Mamba) | Spectral Repair + PGHI | ~~EQ-Anhebung~~ |
| Pitch-Tracking (mono/Gesang) | ML: **RMVPE** → CREPE → FCPE | pYIN | ~~YIN~~ |
| Polyphoner Pitch | ML: **BasicPitch** | Spektrale Peak-Verfolgung | ~~CREPE mono~~ |
| Instrument-Resonanz | DSP: **DDSP** (Eigenimpl.) | Sinusoidal + Stoch. | ~~fixe Formant-EQ~~ |
| Formanten F1–F4 | ML: **DeepFormants CNN** (ONNX) | LPC (Burg, Ordnung **30–40 bei 48 kHz-SR**, alternativ: Downsampling auf 16 kHz → LPC Ord. 16 → Upsampling) | ~~LPC < 12~~ |
| Neuronale Synthese | ML: **Vocos 44.1 kHz** ONNX → BigVGAN v2 → HiFi-GAN | PGHI-ISTFT | ~~Griffin-Lim~~ |
| Generatives Inpainting | ML: **Flow Matching** | CQTdiff+ → DiffWave → NMF-β | ~~DDPM 1000 Schritte~~ |
| Audio-Tagging | ML: **BEATs** (iter3) → PANNs CNN14 | DSP Spectral Fingerprint | — |
| MOS (ohne Referenz) | ML: **VERSA** → SingMOS (Gesang) | PQS-Gammatone-DSP | ~~PESQ/DNSMOS/CDPAM~~ |
| Speech/Music Enhancement | ML: **MP-SENet 2023** ONNX | SGMSE+ ONNX | OMLSA DSP | ~~DCCRN/FullSubNet+~~ |
| MOS (mit Referenz) | ML: **ViSQOL v3** (**`--audio` PFLICHT**) | PQS-DSP | ~~--speech Mode~~ |
| Phasen-Rekonstruktion | DSP: **PGHI** | Griffin-Lim ≥ 32 Iter. | ~~Direkte ISTFT~~ |
| Decrackle | DSP: **RBME + iterative Konsistenz** | Sparse Bayes | ~~Medianfilter~~ |
| Spektral-Matching | DSP: **Optimal Transport** | Multibänder-EQ | ~~fixe EQ-Kurve~~ |
| Groove / Timing | DSP: **Onset-DTW (madmom RNN)** | Beat-Tracking (librosa) | ~~fixes Raster~~ |

---

## §4.5 Pflicht-Algorithmus-Spezifikationen

### Rauschunterdrückung (Phase 03, 29)
```
Pflicht: OMLSA (Cohen & Berdugo 2002) + IMCRA-Variante (Cohen 2003)
Gain-Glättung: MMSE-LSA
G_floor: 0.85 an HPG-protected_bins, 0.10 sonst
DeepFilterNet v3.II Musik-Konfiguration:
    energy_bias = −6.0 dB (reduziert aggressive NR in Harmonik-Regionen)
    G_floor via HarmonicPreservationGuard
```

### Inpainting (Phase 24, 55)
```
Kurze Lücken < 50 ms: NMF mit β-Divergenz (β=1, Itakura-Saito) + PGHI
Lange Lücken 50–999 ms: CQTdiff+ (Moliner & Välimäki, ICASSP 2023)
    - CQT-Domänen-Diffusion konditioniert auf Phrasen-Kontext ±30 s
    - PGHI für phasenkonsistente Rücktransformation
    Fallback: FlowAudio → CQTdiff+ → VampNet → DiffWave → NMF-β
VERBOTEN: VoiceFixer v2 (Sprach-only, VCTK-Korpus)
```

### Codec-Artefakte (Phase 23, 50)
```
Pflicht: Apollo (Zhang et al. 2024)
    - Band-Splitting-RNN: Audio → 24 Sub-Bänder
    - Mamba-Backbone-Sequenzmodellierung
    - Musical Goals Check post-Rekonstruktion (Brillanz ≥ 0.85, Wärme ≥ 0.80)
Fallback: Resemble-Enhance ONNX → DSP Spectral Repair + PGHI
```

### Print-Through-Reduktion (Phase 29, reel_tape)
```
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

### Neuronale Synthese / Vocos (wenn PQS-MOS < 4.3)
```
Primär: Vocos 0.2.0 — vocos_mel_spec_24khz.onnx (CPUExecutionProvider)
    Mel-Bins 80; True-Peak −1.0 dBTP nach Synthese
    Fallback: HiFi-GAN (3,6 MB ONNX) → PGHI-ISTFT
VERBOTEN: Griffin-Lim als Endschritt in Studio-2026
```

### Pit-Korrektur (Phase 12, 31)
```
Primär: pYIN (Mauch & Dixon 2014) + DTW
Bei Gesang (PANNs Vocals ≥ 0.4):
    PSOLA (Moulines & Charpentier 1990) — formanterhaltend bei Transposition > ±2 Halbton
    Phase-Vocoder: nur für perkussive / nicht-vokale Segmente (HPSS-detektiert)
```

### Phasen-Rekonstruktion (nach JEDER Spektral-Modifikation)
```
PFLICHT: PGHI (Perraudin et al. 2013)
Fallback: Griffin-Lim ≥ 32 Iterationen
ABSOLUT VERBOTEN: Direkte ISTFT auf modifiziertem Betragsspektrum
```

### Dithering (24→16 bit Export)
```
PRIMÄR: POW-r Typ 3 (Wannamaker et al. 1992) — ~+6 dB effektiver SNR
FALLBACK: TPDF-Dithering (±1 LSB)
VERBOTEN: Truncation ohne Dithering
```

---

## §12 Referenzen (Auswahl — Pflicht-Algorithmen)

- Cohen & Berdugo (2002): IMCRA — *Noise Estimation by Minima Controlled Recursive Averaging*
- Cohen (2003): OMLSA — *Noise Spectrum Estimation in Adverse Environments*
- Le Roux & Vincent (2013): *Consistent Wiener Filtering for Audio Source Separation*
- Perraudin et al. (2013): PGHI — *A Non-Iterative Method for STFT Phase (Re)construction*
- Févotte & Idier (2011): *Algorithms for NMF with the β-Divergence*
- Mauch & Dixon (2014): pYIN — *A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions*
- Fletcher (1964): *Normal Vibration Frequencies of a Stiff Piano String*
- Engel et al. (2020): DDSP — ICLR 2020 (Google Magenta)
- Bregman (1990): *Auditory Scene Analysis* — MIT Press
- Moore, Glasberg & Baer (2006): Virtual Pitch — JASA
- Zhang et al. (2024): Apollo — *Band-sequence Modeling for High-Quality Music Restoration*
- Moliner & Välimäki (ICASSP 2023): CQTdiff+ — *Solving Audio Inverse Problems with a Diffusion Model*
- Lu et al. (2023): BS-RoFormer — *Music Source Separation with Band-Split RoPE Transformer*
- Siuzdak (2023): Vocos — *Resynthesizing Speech and Music with Neural Vocoders*
- Wannamaker et al. (1992): POW-r — *A Theory of Nonsubtractive Dither*
- Lipman et al. (2023): *Flow Matching for Generative Modeling*
- Radford et al. (2022): Whisper — *Robust Speech Recognition via Large-Scale Weak Supervision*
- ISO 226:2023: *Acoustics — Normal Equal-Loudness-Level Contours*
- ITU-R BS.1770-5 (2023): *Algorithms to measure audio programme loudness*
