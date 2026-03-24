# Aurik DSP Decision Guide — SOTA Model & Algorithm Selection

## Entscheidungsbaum: Wann welches Modell?

[Material-Type] × [DefectType] → [Modell-Empfehlung]

## SOTA-ML-Entscheidungsmatrix

| Aufgabe | PRIMÄR | FALLBACK | VERBOTEN |
|---|---|---|---|
| Noise Reduction (Vocals/Gesang) | DeepFilterNet v3.II | OMLSA+IMCRA | DTLN, RNNoise |
| Noise Reduction (rein instrumental) | OMLSA/IMCRA (kein Speech-Prior) | DeepFilterNet v3.II (energy_bias=−9 dB) | DTLN, RNNoise |
| Stem Separation Vocals | MelBandRoformer (`bs_roformer_plugin`) | MDX23C (Kim_Vocal_2), NMF-β | OpenUnmix |
| Stem Separation Instrumental | MDX23C (`mdx23c_plugin`, Kim_Inst) | HTDemucs-6s (Legacy), NMF-β | OpenUnmix |
| Audio Super-Resolution | AudioSR | Sinusoidal + Stoch. Modeling | SEGAN |
| Codec Artefakte | Apollo | Resemble-Enhance | MetricGAN+ |
| Pitch Estimation | FCPE | CREPE → PESTO → pYIN | SWIPE, YIN |
| Vocoding | Vocos 48 kHz nativ | Vocos 44,1 kHz → BigVGAN v2 → HiFi-GAN | WaveNet RT |
| Inpainting generativ | Flow Matching | CQTdiff+ → DiffWave | einfache Interpolation |
| Audio Tagging | BEATs iter3 | PANNs CNN14 | — |
| MOS-Schätzung Musik | VERSA | SingMOS (Gesang) → PQS-MOS (eigen) | DNSMOS, NISQA, CDPAM, PESQ |
| Dereverb | SGMSE+ (`sgmse_plugin`, TorchScript) | WPE (nara_wpe) → NumPy-WPE → OMLSA | einfacher Bandpass |
| Lyrics-Transcription | Whisper-Tiny ONNX | energy_segmentation_dsp | — |

## Verbotene Modelle & Begründungen

- PESQ: Sprach-Metrik, ungeeignet für Musik
- DNSMOS: DNS-Challenge, nicht für Musikrestaurierung validiert
- NISQA: Sprach-NarrowBand, keine Musikperzeption
- STOI: Nur Sprachverständlichkeit, keine musikalische Qualität
- DTLN: Für RT-Sprach-Denoising, nicht musik-optimiert
- RNNoise: WebRTC-Sprach-Stack, kein Musik-Support
- SEGAN: Überpädagogischer Ansatz, Artefakte bei Musik
- MetricGAN+: STOI-optimiert → Musik-irrelevant
- OpenUnmix: Veraltet, schlechtere Separation als MDX23C
- WaveNet RT: Zu langsam für RT, Artefakte bei Nicht-Sprache
- POLQA: Sprach-Metrik (wie PESQ), keine Musikvalidierung

## Integrations-Checklist (neue Modelle)

1. [ ] Lokal gebündelt (kein Download-Code in Produktion)
2. [ ] models/manifest.json v2 Eintrag
3. [ ] SHA256-Prüfsumme hinterlegt
4. [ ] Post-2018-DSP-Fallback definiert
5. [ ] SR=48000 Konformität geprüft
6. [ ] Musik-spezifischer Benchmark (nicht PESQ/DNSMOS)
7. [ ] Material × DefectType Mapping eingetragen
8. [ ] Plugin-Policy-Konformität (§11.3 specs/08)
9. [ ] Thread-safe Singleton-Integration

## Versionsmatrix

| Modell | Version | Eingebunden seit |
|---|---|---|
| DeepFilterNet | v3.II | Aurik 9.0 |
| MelBandRoformer | 860 MB ONNX | Aurik 9.10.x |
| HTDemucs | 6s ONNX | Aurik 9.10.x |
| MDX23C | Kim_Vocal_2 / Kim_Inst | Aurik 9.0 (Fallback) |
| Apollo | v1 TorchScript | Aurik 9.0 |
| FCPE | ONNX | Aurik 9.10.x |
| CREPE | full ONNX | Aurik 9.0 (Fallback) |
| Vocos | 48 kHz nativ ONNX | Aurik 9.10.x |
| BEATs | iter3 ONNX 90 MB | Aurik 9.10.x |
| PANNs | CNN14 ONNX | Aurik 9.0 (Fallback) |
| VERSA | PyTorch Checkpoint | Aurik 9.10.x |
| SGMSE+ | TorchScript 251 MB | Aurik 9.10.x |
| WPE | nara_wpe | Aurik 9.10.43 (Fallback) |
| Flow Matching | ONNX/PT | Aurik 9.10.x |
| Whisper-Tiny | ONNX 39 MB | Aurik 9.10.46b |
| HiFi-GAN | V2 ONNX | Aurik 9.0 (Fallback) |
| DiffWave | ONNX | Aurik 9.0 (Fallback) |
| Resemble-Enhance | ONNX 722 MB | Aurik 9.0 (Fallback) |
