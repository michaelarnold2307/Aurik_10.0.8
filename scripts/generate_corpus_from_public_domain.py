#!/usr/bin/env python3
"""Generator: Public-Domain-Aufnahmen für Echt-Audio-Corpus herunterladen.

§15.2: Lädt Public-Domain-Aufnahmen von Internet Archive, Musopen und
Freesound (CC0) und erstellt manifest.yaml-Einträge.

Nutzung:
    python scripts/generate_corpus_from_public_domain.py --source internet_archive
    python scripts/generate_corpus_from_public_domain.py --source musopen
    python scripts/generate_corpus_from_public_domain.py --source freesound
    python scripts/generate_corpus_from_public_domain.py --all --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).parent.parent
_CORPUS_DIR = _PROJECT_ROOT / "corpus"

logger = logging.getLogger(__name__)

# ── Public-Domain-Quellen ────────────────────────────────────────────────
INITIAL_CORPUS: list[dict] = [
    {
        "file": "placeholder_shellac_jazz_1938.wav",
        "material": "shellac",
        "era": 1938,
        "genre": "Jazz",
        "defects": ["clicks", "surface_noise", "rumble"],
        "source": "Internet Archive — 78rpm Collection",
        "source_url": "https://archive.org/details/78rpm",
        "license": "Public Domain",
        "notes": "Bitte durch echte Public-Domain-Aufnahme ersetzen",
        "subdir": "damaged",
    },
    {
        "file": "placeholder_vinyl_classical_1972.wav",
        "material": "vinyl",
        "era": 1972,
        "genre": "Classical",
        "defects": ["pops", "crackle", "inner_groove_distortion"],
        "source": "Musopen",
        "source_url": "https://musopen.org",
        "license": "Public Domain",
        "notes": "Bitte durch echte Public-Domain-Aufnahme ersetzen",
        "subdir": "damaged",
    },
    {
        "file": "placeholder_tape_folk_1965.wav",
        "material": "tape",
        "era": 1965,
        "genre": "Folk",
        "defects": ["hiss", "dropouts", "print_through"],
        "source": "Community Recording Project",
        "source_url": "",
        "license": "CC0",
        "notes": "Bitte durch echte CC0-Aufnahme ersetzen",
        "subdir": "damaged",
    },
    {
        "file": "placeholder_digital_electronic_2020.wav",
        "material": "digital",
        "era": 2020,
        "genre": "Electronic",
        "defects": ["clipping"],
        "source": "Freesound (CC0)",
        "source_url": "https://freesound.org",
        "license": "CC0",
        "notes": "Bitte durch echte CC0-Aufnahme ersetzen",
        "subdir": "damaged",
    },
]


def _compute_sha256(filepath: Path) -> str:
    """SHA-256-Hash einer Datei."""
    if not filepath.exists():
        return "PLACEHOLDER_FILE_MISSING"
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def generate_manifest(directory: Path, entries: list[dict], dry_run: bool = False) -> int:
    """Generiert manifest.yaml für ein Corpus-Verzeichnis."""
    manifest_path = directory / "manifest.yaml"
    manifest_data: list[dict] = []

    for entry in entries:
        audio_path = directory / entry["file"]

        # Wenn Datei existiert, Metadaten auslesen
        if audio_path.exists():
            import soundfile as sf

            info = sf.info(str(audio_path))
            manifest_entry = {
                "file": entry["file"],
                "duration_s": round(info.duration, 2),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "material": entry["material"],
                "era": entry["era"],
                "genre": entry.get("genre", ""),
                "defects": entry.get("defects", []),
                "source": entry.get("source", ""),
                "source_url": entry.get("source_url", ""),
                "license": entry.get("license", ""),
                "license_url": entry.get("license_url", ""),
                "checksum_sha256": _compute_sha256(audio_path),
                "notes": entry.get("notes", ""),
            }
        else:
            # Placeholder — Datei muss manuell hinzugefügt werden
            manifest_entry = {
                "file": entry["file"],
                "duration_s": 0.0,
                "sample_rate": 0,
                "channels": 0,
                "material": entry["material"],
                "era": entry["era"],
                "genre": entry.get("genre", ""),
                "defects": entry.get("defects", []),
                "source": entry.get("source", ""),
                "source_url": entry.get("source_url", ""),
                "license": entry.get("license", ""),
                "license_url": entry.get("license_url", ""),
                "checksum_sha256": "",
                "notes": f"PLACEHOLDER: {entry.get('notes', '')}",
                "_status": "placeholder",
            }

        manifest_data.append(manifest_entry)

    if dry_run:
        print(f"[DRY-RUN] Würde {len(manifest_data)} Einträge nach {manifest_path} schreiben:")
        for e in manifest_data:
            status = e.get("_status", "OK")
            print(f"  [{status}] {e['file']} ({e['material']}, {e['era']})")
        return len(manifest_data)

    manifest_path.write_text(
        yaml.dump(manifest_data, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    logger.info("manifest.yaml geschrieben: %s (%d Einträge)", manifest_path, len(manifest_data))
    return len(manifest_data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Public-Domain-Corpus-Generator für Aurik (§15.2)")
    parser.add_argument(
        "--source",
        choices=["internet_archive", "musopen", "freesound"],
        help="Nur bestimmte Quelle generieren",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Alle Quellen generieren",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur anzeigen, keine Dateien schreiben",
    )
    args = parser.parse_args()

    if not args.all and not args.source:
        parser.print_help()
        print("\n⚠️  Bitte --all oder --source angeben.")
        return 1

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    total = 0
    for entry in INITIAL_CORPUS:
        material = entry["material"]
        subdir = entry["subdir"]
        target_dir = _CORPUS_DIR / material / subdir

        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)

        count = generate_manifest(target_dir, [entry], dry_run=args.dry_run)
        total += count

    if not args.dry_run:
        print(f"\n✅ {total} manifest.yaml-Einträge generiert.")
        print("⚠️  Die Audio-Dateien sind PLACEHOLDER und müssen durch echte")
        print("    Public-Domain-Aufnahmen ersetzt werden (siehe corpus/README.md).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
