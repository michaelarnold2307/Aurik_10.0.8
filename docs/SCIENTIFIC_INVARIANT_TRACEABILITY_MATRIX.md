# Scientific Invariant Traceability Matrix

Stand: 2026-05-20

Ziel: 1:1-Nachweis fuer zentrale Invarianten in Aurik gegen wissenschaftliche
Primaerquellen und konkrete Mess-/Validierungsprotokolle im Projekt.

## Legende

- `Theorie`: Primaerliteratur oder Normen, die die Regel wissenschaftlich tragen.
- `Operationalisierung`: Wie die Regel im Code/Workflow konkret angewendet wird.
- `Messprotokoll`: Reproduzierbarer Test-/Audit-Pfad.
- `Status`: `Direkt` (nahezu direkt aus Theorie/Norm) oder `Wissenschaftlich motiviert + kalibriert`.

## Source-IDs

| Source-ID | Referenz |
| --- | --- |
| `S01` | Ephraim & Malah (1984) |
| `S02` | Ephraim & Malah (1985) |
| `S03` | Cohen (2003) |
| `S04` | Cohen & Berdugo (2002) |
| `S05` | Nakatani et al. (2010) |
| `S06` | ITU-R BS.1770-5 |
| `S07` | EBU R128 |
| `S08` | Makhoul (1975) |
| `S09` | Boersma (1993) |
| `S10` | Titze (2000) |
| `S11` | Sundberg (1987) |
| `S12` | Griffin & Lim (1984) |
| `S13` | Prusa et al. (2017) |
| `S14` | Lapierre & Lefebvre (2017) |
| `S15` | Godsill et al. (1995) |
| `S16` | Miller (1992) |
| `S17` | Prame (2004) |
| `S18` | Jones et al. (2022) |

## Matrix

| Invariante | Theorie / Primaerquelle | Operationalisierung in Aurik | Messprotokoll | Status |
| --- | --- | --- | --- | --- |
| MMSE-LSA/OMLSA/IMCRA-basierte NR-Guards | `S01`, `S02`, `S03`, `S04` | Subtraktive NR wird durch psychoakustische Floors und konservative Caps begrenzt | Unit- und Gate-Tests fuer NR-Pfade, inklusive Oracle-Caps | Wissenschaftlich motiviert + kalibriert |
| Dereverb-Logik mit WPE-Familie | `S05` | Dereverb-Pfade als subtraktive Klasse, konservative Strength-Oracle-Konditionierung | Phase- und Integrations-Tests (u. a. late phases) | Wissenschaftlich motiviert + kalibriert |
| Loudness/TruePeak-Sicherheit | `S06`, `S07` | Output-orientierte Phasen (O10) mit strengen Caps und Export-Gates | Output-/Gate-Tests, Release-Gates | Direkt |
| Formant-/HNR-/Vibrato-Schutz fuer Vocals | `S08`, `S09`, `S10`, `S11` | Vocal-spezifische Invarianten und O7-Klassensteuerung | Vocal-/De-Esser-Regressionen und Gate-Checks | Wissenschaftlich motiviert + kalibriert |
| Zeit-Frequenz-Rekonstruktion/Phase-Konsistenz | `S12`, `S13` | Spektrale Reparaturpfade + konsistente Rekonstruktion in Reparaturketten | Spektral-Repair- und Konsistenztests | Wissenschaftlich motiviert + kalibriert |
| Artifact-Freedom fuer Musical-Noise/Pre-Echo | `S14`, `S15` | AFG-Detektoren und Recovery-Trigger fuer artefaktkritische Faelle (`artifact_freedom < 0.95`) | Gate-Regressionen + materialadaptive AFG-Audits | Wissenschaftlich motiviert + kalibriert |
| Formant-/Vibrato-Grenzen im Gesangspfad | `S16`, `S17`, `S18` | Formant-Toleranz, Vibrato-Tiefenschutz und Vokal-Recovery im §0p-Pfad | Vocal-Gate-Tests + De-Esser-/Vibrato-Regressionen | Wissenschaftlich motiviert + kalibriert |
| Multi-Goal Teamwork statt Single-Goal Dominanz | Multi-Objective Optimization Grundlagen (Pareto/weighted gap) | 15-Goal Weighted-Gap-Closure + Dominanz-Guard im Oracle | Oracle-Unit-Tests inkl. dominant_goal_guard | Wissenschaftlich motiviert + kalibriert |
| Transfer-chain-aware Strength-Konditionierung | Kaskadierte Degradationsmodelle in Audio-Restoration-Literatur; robuste Unsicherheitsgewichtung | Deterministischer `chain_factor` aus `material_key`, `transfer_chain`, `chain_confidence` skaliert Driver und Hard-Caps | `tests/unit/test_phase_strength_oracle.py` (chain-factor Tests) + Chain-aware Audit | Wissenschaftlich motiviert + kalibriert |
| Material-/Traegeradaptive Zielschwellen | Restaurationspraxis + materialphysikalische Grenzen (Carrier ceilings/floors) | PMGG/effective targets + material-adaptive Floors | Goal-baseline-Checks, PMGG/CIG/AFG Gates | Wissenschaftlich motiviert + kalibriert |

## Bibliographie (Kernquellen)

1. Ephraim, Y., Malah, D. (1984). Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator. IEEE TASSP.
2. Ephraim, Y., Malah, D. (1985). Speech enhancement using a minimum mean-square error log-spectral amplitude estimator. IEEE TASSP.
3. Cohen, I. (2003). Noise spectrum estimation in adverse environments: Improved minima controlled recursive averaging. IEEE TSAP.
4. Cohen, I., Berdugo, B. (2002). Speech enhancement for non-stationary noise environments. Signal Processing.
5. Nakatani, T. et al. (2010). Speech dereverberation based on variance-normalized delayed linear prediction. IEEE TASLP.
6. ITU-R BS.1770-5. Algorithms to measure audio programme loudness and true-peak audio level.
7. EBU R128. Loudness normalisation and permitted maximum level of audio signals.
8. Makhoul, J. (1975). Linear prediction: A tutorial review. Proceedings of the IEEE.
9. Boersma, P. (1993). Accurate short-term analysis of the fundamental frequency and harmonics-to-noise ratio.
10. Titze, I. R. (2000). Principles of Voice Production.
11. Sundberg, J. (1987). The Science of the Singing Voice.
12. Griffin, D., Lim, J. (1984). Signal estimation from modified short-time Fourier transform. IEEE TASSP.
13. Prusa, Z. et al. (2017). Phase Gradient Heap Integration for phase reconstruction.
14. Lapierre, J., Lefebvre, R. (2017). Pre-echo noise reduction in frequency-domain audio codecs. ICASSP. DOI: 10.1109/ICASSP.2017.7952243.
15. Godsill, S. J., Rayner, P. J. W., Cappe, O. (1995). Evaluation of short-time spectral attenuation techniques for the restoration of musical recordings. IEEE Transactions on Speech and Audio Processing. DOI: 10.1109/89.365378.
16. Miller, R. (1992). Formant frequency tuning in singing. Journal of Voice. DOI: 10.1016/S0892-1997(05)80150-X.
17. Prame, E. (2004). The relationship between measured vibrato characteristics and perception in Western operatic singing. Journal of Voice. DOI: 10.1016/j.jvoice.2003.09.003.
18. Jones, R. I. et al. (2022). Perception of vibrato rate by professional singing voice teachers. The Journal of the Acoustical Society of America. DOI: 10.1121/10.0015518.

## Reproduzierbarkeit

- Code-Referenz: `backend/core/dsp/phase_strength_oracle.py`, `backend/core/unified_restorer_v3.py`
- Test-Referenz: `tests/unit/test_phase_strength_oracle.py`
- Audit-Referenz: `audit/CHAIN_AWARE_ORACLE_WORLDCLASS_AUDIT_2026-05-20.md`
- Governance-Referenz: `.github/specs/02_pipeline_architecture.md`, `.github/specs/09_global_calibration_matrix.md`, `.github/instructions/pipeline.instructions.md`
