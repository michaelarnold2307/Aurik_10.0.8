---
applyTo: "{backend/core/dsp/*.py,plugins/*.py}"
---

# DSP / Plugin-Regeln (normativ, Aurik 9.12.x)

## ML-Device — IMMER über ml_device_manager

```python
# VERBOTEN:
model.to("cuda")                        # direkt, ignoriert AMD/ROCm
providers = ["CUDAExecutionProvider"]   # ignoriert DirectML/ROCm

# RICHTIG (Heavy-Plugin):
from backend.core.ml_device_manager import get_torch_device, get_ort_providers
model.to(get_torch_device("PluginName"))   # fp16 + Tier automatisch
session = ort.InferenceSession(path, providers=get_ort_providers("PluginName"))

# Light-Plugin / DSP:
model.to("cpu")
torch.set_num_threads(os.cpu_count())
providers = ["CPUExecutionProvider"]
```

## §2.62 Psychoakustischer Masking-Guard (NR-Algorithmen)

```python
# PFLICHT vor jedem NR-Aufruf (DeepFilterNet, OMLSA, SGMSE+):
from backend.core.dsp.psychoacoustics import compute_masking_threshold_iso11172

masking_threshold = compute_masking_threshold_iso11172(audio, sr)
# Bark-Skala (ISO 11172-3): 0–24 Bark, ~24 kritische Bänder bis 22 kHz
# ERB-Skala (ISO 532-1, Moore-Glasberg): genauer für Lautheit — für Loudness-Normalisierung
# → Masking-Guard nutzt Bark (ISO 11172-3); Loudness-Normalisierung nutzt ERB (ISO 532-1)

# MMSE-LSA Gain (Ephraim-Malah 1985, Log-Spectral Amplitude Estimator):
# BESSER als einfacher Wiener-Filter-Floor — erhält Spektralform, reduziert Musical Noise:
from scipy.special import exp1 as expint1  # E1(v) = exponential integral; exp1(v) = ∫_v^∞ exp(-t)/t dt
for band in range(n_bands):
    xi = max(noisy_power[band] / noise_estimate[band] - 1.0, 0.0)  # a-priori SNR
    gamma = noisy_power[band] / noise_estimate[band]               # a-posteriori SNR
    v = xi / (1.0 + xi) * gamma
    G_mmse_lsa = xi / (1.0 + xi) * np.exp(0.5 * expint1(v))       # MMSE-LSA gain
    G_floor[band] = max(G_mmse_lsa, 0.10, masking_threshold[band] / noise_estimate[band])
    # VERBOTEN: G_floor < 0.10 in Bändern mit Musik-Energie > -60 dBFS

# Ergebnis: kein "totes Stille"-Artefakt zwischen Phrasen
# VERBOTEN: einfacher Wiener-Filter ohne Masking-Floor — verursacht Musical-Noise
```

## Noise-Schätzung — IMCRA/OMLSA (stationär + nicht-stationär)

```python
# KANONISCH — Rausch-Schätzung für Musik (nicht-stationäre Rausch-Quellen):
# OMLSA (Cohen & Berdugo, 2004) = Optimal Modified Log-Spectral Amplitude
# → besser als Spectral Subtraction für Musik (verarbeitet harmonische Störgeräusche)
# Noise-Schätzung via IMCRA (Cohen, 2003):
#   - gleitende Minimum-Schätzung + Sprachpräsenz-Posterior
#   - adaptive Zeitkonstanten: tau_min=0.04s, alpha_d=0.85, alpha_s=0.9
# DeepFilterNet + OMLSA gemeinsam: DFN für Breitband-NR, OMLSA für Restgeräusch
from backend.core.dsp.noise_estimator import compute_imcra_noise_estimate
noise_psd = compute_imcra_noise_estimate(audio, sr, alpha_d=0.85, alpha_s=0.9)
# Initialphase (2s) → konservative Schätzung (Faktor 1.3 × Minimum)
```

## DeepFilterNet — Energy-Bias-Pflicht (§0j)

```python
# VERBOTEN: DeepFilterNet ohne energy_bias auf Vokal/Instrumental
# Harmonik-Regionen werden ohne Bias als Rauschen klassifiziert

# RICHTIG — energy_bias modifiziert DFNs internen Noise-Floor-Schätzwert:
# DFN schätzt Rauschumgebung → energy_bias verschiebt die Entscheidungsgrenze
# sodass harmonische Energie NICHT als Rauschen abgetragen wird
# (energy_bias in dB → intern: noise_floor_estimate *= 10^(energy_bias/20))
if panns_singing >= 0.35:
    # Vokal-Material: Register-adaptiv für präziseste Einstellung
    # (Register-Detektor NUR auf Vokal-Material aufrufen — für Instrumental sinnlos!)
    from backend.core.dsp.vocal_register_detector import detect_vocal_register_temporal
    register = detect_vocal_register_temporal(audio, sr)
    energy_bias = _REGISTER_BIAS.get(register.dominant, -6.0)
    # Fallback -6.0 dB wenn register.dominant kein Eintrag in _REGISTER_BIAS
elif is_instrumental:
    energy_bias = -9.0  # dB — Instrumental: aggressivere Schutzgrenze; kein Register
else:
    energy_bias = -6.0  # dB — Default (unklares Material)
```

## HNR-Guard nach NR auf Gesangsmaterial

```python
# PFLICHT wenn panns_singing >= 0.25 + nach DFN/SGMSE+/OMLSA:
from backend.core.dsp.hnr_guard import apply_hnr_blend

audio_out = apply_hnr_blend(audio_pre_nr, audio_post_nr, sr)
# ΔHNR > 3 dB → automatischer Dry-Blend
# verhindert "klinischen" Klang nach aggressivem NR
```

## LPC-Ordnung — Material-abhängig

```python
# VERBOTEN: LPC-Ordnung < 16 bei 16 kHz oder < 30 bei 48 kHz
# RICHTIG:
if analysis_sr == 48000:
    lpc_order = 30  # bis 40 für breite Formant-Tracks
elif analysis_sr == 16000:
    lpc_order = 16  # bevorzugt für Shellac BW ≤ 8 kHz (Downsampling → 16k)
# lpc_formant_tracker.py verwendet _LPC_ORDER=16 + _LPC_ANALYSIS_SR=16000
```

## Singleton-Pattern — ALLE Kernmodule

```python
import threading

_instance = None
_lock = threading.Lock()

def get_my_plugin():
    global _instance  # oder list-Container: _holder = [None]
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MyPlugin()
    return _instance
```

## ONNX-Chunking (Heavy-Plugins)

```python
# KANONISCH — Chunk-Verarbeitung mit Overlap-Add:
chunk_size = 65536  # ~1.4s bei 48kHz
overlap = 4096
for i in range(0, n_samples, chunk_size - overlap):
    chunk = audio[..., i: i + chunk_size]
    out_chunk = session.run(None, {"input": chunk})[0]
    # shape[-1] statt len() — len(out_chunk) wäre 2 für 2D-Stereo-Output!
    out_len = out_chunk.shape[-1]
    # Overlap-Add mit Hann-Fenster (window: shape (out_len,) oder (1, out_len) für Broadcasting)
    output[..., i: i + out_len] += out_chunk * window[:out_len]

# OOM-Fallback → DSP-Kette, nie Crash:
try:
    result = heavy_model.process(audio)
except (RuntimeError, MemoryError):
    metadata["ml_fallbacks_used"]["model_name"] = True
    result = _dsp_fallback(audio, sr)
```

## MIIPHER-Fallback (SNR < 10 dB + Gesang)

```python
from plugins.miipher_plugin import get_miipher_plugin

if noise_snr_db < 10.0 and panns_singing >= 0.35:
    miipher = get_miipher_plugin()
    if miipher.should_activate(noise_snr_db, panns_singing):
        audio_pre_miipher = audio.copy()  # für HNR-Blend-Referenz
        audio = miipher.enhance(audio, sr)
        # Intern: Stub → DeepFilterNet(-6dB) → Wiener-Fallback
        # PFLICHT: apply_hnr_blend() nach MIIPHER — ΔHNR > 3 dB → Dry-Blend (§0p HNR-Schutz)
        from backend.core.dsp.hnr_guard import apply_hnr_blend
        audio = apply_hnr_blend(audio_pre_miipher, audio, sr)
```

## §0p Formant-Schutz (Pflicht bei `panns_singing ≥ 0.25`)

```python
# KANONISCH — nach jeder DSP-Phase auf Gesangsmaterial:
from backend.core.dsp.formant_guard import check_formant_integrity

pre_formants = check_formant_integrity(audio_pre, sr)  # F1–F4 via LPC
post_formants = check_formant_integrity(audio_post, sr)

for f_idx in range(4):  # F1, F2, F3, F4
    shift_db = abs(post_formants[f_idx] - pre_formants[f_idx])
    if shift_db > 2.0:  # > ±2 dB = sofortiger Rollback
        logger.warning("formant_shift F%d = %.1f dB > 2.0 → rollback", f_idx+1, shift_db)
        return audio_pre  # keine Ausnahmen
```

## §0p Vibrato-Schutzzone (Pflicht bei `panns_singing ≥ 0.25`)

```python
# KANONISCH — vor jeder NR/Transient/Dynamics-Phase auf Gesangsmaterial:
from backend.core.dsp.natural_performance_detector import detect_performance_artifacts

artifacts = detect_performance_artifacts(audio, sr)
vibrato_mask = artifacts.vibrato_mask  # bool-Array, True = geschützte Frame

# Alle Phasen in Vibrato-Zonen: strength auf max. 0.20 begrenzen
if np.any(vibrato_mask):
    strength = min(strength, 0.20)  # Vibrato ist Naturalness-Marker
    logger.debug("vibrato_guard: strength capped to 0.20 (%d frames)", np.sum(vibrato_mask))
```

## §0p Passaggio-Zone-Awareness (MIIPHER/DFN bei Gesang)

```python
# KANONISCH — Registerübergangszonen bei ML-NR:
from backend.core.dsp.vocal_register_detector import detect_vocal_register_temporal

register_map = detect_vocal_register_temporal(audio, sr)
# register_map.passaggio_zones: List[(start_s, end_s)]

# Energy-Bias in Übergangszone = Mittelwert Brust+Kopf:
_REGISTER_BIAS = {
    "chest": -6.0, "head": -6.0, "falsetto": -9.0,
    "passaggio": -3.0,  # Mittelwert Brust/Kopf
    "fry": -12.0, "whisper": -15.0,
}
energy_bias = _REGISTER_BIAS.get(register_map.dominant, -6.0)
```

## Timbral Coherence Guard

```python
from backend.core.dsp.timbral_coherence_guard import (
    extract_song_noise_profile,
    compute_timbral_coherence_score,
)

noise_profile = extract_song_noise_profile(audio_pre_pipeline, sr)
# ... nach NR-Phasen ...
coherence = compute_timbral_coherence_score(audio_post_nr, noise_profile, sr)
if coherence < 0.80:
    logger.warning("timbral_coherence=%.3f < 0.80 — Rollback auf pre-NR (§CSTC)", coherence)
    audio_post_nr = audio_pre_pipeline  # Rollback
# VERBOTEN: assert coherence >= 0.80 — assert ist in Produktionscode durch -O deaktivierbar
# Vinyl → rosa Rauschtextur; Tape → Brown+HF-Hiss; CD → Flat/Weiß
```

## Cross-Segment Timbral Coherence — Rauschtextur

```python
# Spektrale Form des Restrauschens MUSS zum Trägerprofil passen:
# Vinyl:   rosa   (1/f)
# Tape:    Brown + HF-Hiss
# CD:      Weiß / Flat
# Shellac: Spezifisches Oberflächenprofil
# Kohärenz-Score ≥ 0.80 Pflicht (§0a Rauschtextur-Invariante)
```

## Multi-Singer Detection

```python
from backend.core.dsp.vocal_register_detector import detect_multi_singer

if detect_multi_singer(audio, sr, panns_singing):
    metadata["multi_singer"] = True
    # Resemblyzer-Gate ÜBERSPRINGEN (Embedding für Einzelidentität, nicht Duett)
    # singer_identity_cosine-Gate deaktiviert
```

## Signal-relatives Gate (V04)

```python
# VERBOTEN:
gate_dbfs = -36.0  # feste Konstante

# RICHTIG:
from backend.core.dsp.gain_utils import compute_signal_relative_gate_dbfs
gate_dbfs = compute_signal_relative_gate_dbfs(
    pre_phase_audio, material_key=material_type
)
# reference_for_gate=pre_phase_audio — IMMER
```

## Bandfilter — Zero-Phase

```python
# VERBOTEN: sosfilt(sos, audio) addiert zu Original
# RICHTIG:
from scipy.signal import sosfiltfilt
filtered = sosfiltfilt(sos, audio)  # zero-phase überall wo Band auf Signal addiert
```

## STFT — Fenster und Overlap-Pflicht

```python
# KANONISCH für alle STFT-basierten Phasen (Rekonstruktionsqualität):
_STFT_HOP_FRACTION = 0.25     # 75 % Overlap — Pflichtstandard
_STFT_WINDOW = "hann"         # Hann-Fenster: perfekte Rekonstruktion bei 75 % Overlap
# VERBOTEN: hop_fraction > 0.5 (< 50 % Overlap) bei Synthesis-STFT
# VERBOTEN: Rechteck-Fenster (Rectangular) für Analyse und Synthese gleichzeitig

# Konsistenz-Invariante: Analyse-Fenster und Synthese-Fenster MÜSSEN gleich sein
# Ausnahme: PGHI/Vocos nutzen eigene interne Fensterung — nicht manuell überschreiben
```

## Phasenrekonstruktion — PGHI statt Griffin-Lim

```python
# VERBOTEN: griffinlim() als Endschritt (vgl. VERBOTEN.md V05)
# Ursache: iterative Phase-Estimation ist nicht-deterministisch + zu langsam

# RICHTIG — Option 1: PGHI (Phase Gradient Heap Integration, Prusa et al. 2017):
# Deterministisch, ein Pass, minimal-phasenäquivalent
# Parameter:
_PGHI_GAMMA = 0.25 * frame_size**2 / sr  # Frequenz-Zeit-Kopplung
_PGHI_TOLERANCE = 1e-6                    # Konvergenz-Schwelle
from backend.core.dsp.pghi import pghi_reconstruct
audio_out = pghi_reconstruct(magnitude, sr, n_fft=n_fft, hop_length=hop_length,
                              gamma=_PGHI_GAMMA, tol=_PGHI_TOLERANCE)

# RICHTIG — Option 2: Vocos (Siuzdak 2023) — vollständig neural, kein STFT-invert:
# Besser für hochqualitative Vokale; nutzt iSTFT-basierte Architektur ohne iterative Optimierung
# Weniger anfällig auf spektrale Löcher als PGHI
from plugins.vocos_plugin import get_vocos_plugin
voc = get_vocos_plugin()
audio_out = voc.decode(magnitude_features, sr=sr)  # CPU-only empfohlen (light)
# Entscheid PGHI vs Vocos: PGHI wenn Magnitude aus linearem DSP stammt;
# Vocos wenn Magnitude aus ML-Modell (DFN, SGMSE+) stammt
```

## Peak-Guard

```python
# VERBOTEN:
peak = np.max(np.abs(audio))

# RICHTIG:
peak = np.percentile(np.abs(audio), 99.9)
```
