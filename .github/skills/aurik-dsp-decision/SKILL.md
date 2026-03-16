# Aurik DSP Decision Guide — SOTA Model & Algorithm Selection

## Entscheidungsbaum: Wann welches Modell?

[Material-Type] × [DefectType] → [Modell-Empfehlung]

## SOTA-ML-Entscheidungsmatrix

| Aufgabe | PRIMÄR | FALLBACK | VERBOTEN |
|---|---|---|---|
| Noise Reduction | DeepFilterNet v3.II | OMLSA+IMCRA | DTLN, RNNoise |
| Stem Separation | MDX23C Kim_Vocal_2 / Kim_Inst | Demucs v4 | OpenUnmix |
| Audio Super-Resolution | BS-RoFormer / Mel-RoFormer | — | SEGAN |
| Codec Artefakte | Apollo | Resemble-Enhance | MetricGAN+ |
| Pitch Estimation | CREPE | pYIN | SWIPE, YIN |
| Vocoding | Vocos 24 kHz | HiFi-GAN V2 | WaveNet RT |
| Inpainting | FlowMatching | DiffWave | einfache Interpolation |
| Audio Tagging | PANNs CNN14 | — | — |
| MOS-Schätzung | CDPAM | PQS-MOS (eigen) | DNSMOS, NISQA |
| Dereverb | WPE (nara_wpe) | NumPy-WPE | SGMSE+ |
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
| MDX23C | Kim_Vocal_2 / Kim_Inst | Aurik 9.0 |
| BS-RoFormer | latest | Aurik 9.0 |
| Apollo | v1 | Aurik 9.0 |
| CREPE | full | Aurik 9.0 |
| Vocos | 24kHz | Aurik 9.0 |
| PANNs | CNN14 | Aurik 9.0 |
| CDPAM | v1 | Aurik 9.0 |
| WPE | nara_wpe | Aurik 9.10.43 |
| Whisper-Tiny | ONNX | Aurik 9.10.46b (v10.0) |
| HiFi-GAN | V2 | Aurik 9.0 (Fallback) |
| DiffWave | — | Aurik 9.0 (Fallback) |
| Resemble-Enhance | — | Aurik 9.0 (Fallback) |
| Demucs | v4 | Aurik 9.0 (Fallback) |
