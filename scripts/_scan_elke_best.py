"""Minimal one-off scan: Elke Best MP3 — nur relevante non-stationary defects.
Usage: .venv_aurik/bin/python _scan_elke_best.py

Sicherer Schnellpfad: Audio via pedalboard laden und nur die Defekte prüfen,
die für wiederholende unnatürliche Signaländerungen entscheidend sind.
"""

import os
import sys
import time

from pedalboard.io import AudioFile

from backend.core.defect_scanner import DefectScanner, MaterialType

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

FILE = "Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3"

print(f"[1] Lade Audio direkt via pedalboard: {FILE}")
t0 = time.time()
with AudioFile(FILE) as f:
    sr = int(f.samplerate)
    raw = f.read(f.frames)  # (channels, samples)

audio_mono = raw.mean(axis=0).astype("float32")
dur_s = len(audio_mono) / sr
channels = raw.shape[0]
print(f"    -> SR={sr} Hz | Kanaele={channels} | Laenge={dur_s:.1f}s | geladen in {time.time() - t0:.2f}s")

print("[2] Starte gezielte Defektanalyse (dropouts / transport_bump / tape_head_level_dip) ...")
scanner = DefectScanner(sample_rate=sr, material_type=MaterialType.TAPE)
scanner.material_type = MaterialType.TAPE
scanner.thresholds = scanner.MATERIAL_SENSITIVITY[MaterialType.TAPE]

t1 = time.time()
drop = scanner._detect_dropouts(audio_mono)
t2 = time.time()
bump = scanner._detect_transport_bump(audio_mono)
t3 = time.time()
dip = scanner._detect_tape_head_level_dips(audio_mono)
t4 = time.time()

scores = [drop, bump, dip]
print("\n[3] Ergebnis relevante Defekte:")
print(f"\n  {'Defekttyp':<35} {'Severity':>8} {'Conf':>6} {'Locations':>10}")
print(f"  {'-' * 35} {'-' * 8} {'-' * 6} {'-' * 10}")

total_locations = 0
for sc in scores:
    n_loc = len(sc.locations or [])
    total_locations += n_loc
    marker = " <Ereignisse>" if n_loc > 0 else ""
    print(f"  {sc.defect_type.value:<35} {sc.severity:>8.3f} {sc.confidence:>6.2f} {n_loc:>10}{marker}")

print(f"\n  Gesamt Ereignis-Locations: {total_locations}")
print(f"\n  Laufzeiten: dropouts={t2 - t1:.2f}s | transport_bump={t3 - t2:.2f}s | tape_head_level_dip={t4 - t3:.2f}s")

top = max(scores, key=lambda s: len(s.locations or []))
if len(top.locations or []) > 0:
    print(f"\n[4] Erste 15 Locations von '{top.defect_type.value}' ({len(top.locations)} gesamt):")
    for i, (s, e) in enumerate(top.locations[:15]):
        print(f"    [{i + 1:3}] {s:7.2f}s - {e:7.2f}s  (Dauer {(e - s) * 1000:.0f} ms)")
    if len(top.locations) > 15:
        print(f"    ... und {len(top.locations) - 15} weitere")

print("\n[5] Scan abgeschlossen.")
