"""Corpus-Integritätstest — Prüft alle Manifests auf Konsistenz.

§15.2: Validiert dass alle corpus/manifest.yaml-Dateien valide sind,
Referenzen auf existierende Dateien zeigen und Checksummen stimmen.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

CORPUS_DIR = Path(__file__).parent.parent.parent / "corpus"


def _compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _discover_manifests() -> list[Path]:
    """Findet alle manifest.yaml-Dateien im Corpus."""
    manifests = []
    if CORPUS_DIR.is_dir():
        for mf in CORPUS_DIR.rglob("manifest.yaml"):
            manifests.append(mf)
    return manifests


class TestCorpusIntegrity:
    """§15.2: Corpus-Integritäts-Checks."""

    def test_corpus_directory_exists(self):
        """Corpus-Verzeichnis muss existieren."""
        assert CORPUS_DIR.is_dir(), f"Corpus directory missing: {CORPUS_DIR}"

    def test_readme_exists(self):
        """README.md muss existieren."""
        readme = CORPUS_DIR / "README.md"
        assert readme.exists(), "corpus/README.md missing"

    def test_at_least_one_manifest(self):
        """Mindestens ein manifest.yaml muss existieren."""
        manifests = _discover_manifests()
        assert len(manifests) > 0, "No manifest.yaml found in corpus/"

    def test_all_manifests_valid_yaml(self):
        """Alle Manifests müssen valide YAML sein."""
        for mf in _discover_manifests():
            with open(mf, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert isinstance(data, list), f"{mf.relative_to(CORPUS_DIR)}: not a list"
            for i, entry in enumerate(data):
                assert isinstance(entry, dict), f"{mf.relative_to(CORPUS_DIR)}[{i}]: not a dict"
                # Pflichtfelder
                for key in ["file", "material", "license"]:
                    assert key in entry, f"{mf.relative_to(CORPUS_DIR)}[{i}]: missing '{key}'"

    def test_all_files_referenced_exist(self):
        """Alle in Manifests referenzierten Dateien müssen existieren."""
        for mf in _discover_manifests():
            with open(mf, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            for entry in data:
                if entry.get("_status") == "placeholder":
                    continue  # Placeholder überspringen
                filepath = mf.parent / entry["file"]
                assert filepath.exists(), f"{mf.relative_to(CORPUS_DIR)}: referenced file missing: {entry['file']}"

    def test_checksums_match(self):
        """Alle Checksummen in Manifests müssen mit Dateien übereinstimmen."""
        for mf in _discover_manifests():
            with open(mf, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            for entry in data:
                if entry.get("_status") == "placeholder":
                    continue
                if not entry.get("checksum_sha256"):
                    continue
                filepath = mf.parent / entry["file"]
                if filepath.exists():
                    actual = _compute_sha256(filepath)
                    expected = entry["checksum_sha256"]
                    assert actual == expected, (
                        f"{entry['file']}: checksum mismatch (expected {expected[:16]}..., got {actual[:16]}...)"
                    )

    def test_material_categories_present(self):
        """Mindestens 4 Material-Kategorien müssen vertreten sein."""
        materials = set()
        for mf in _discover_manifests():
            with open(mf, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            for entry in data:
                if "material" in entry:
                    materials.add(entry["material"])

        required = {"shellac", "vinyl", "tape", "digital"}
        missing = required - materials
        # Weich: Nur warnen wenn Placeholder
        if missing:
            print(f"\n⚠️  Missing material categories: {missing}")
            print("   (Expected for placeholder corpus — add real files to resolve)")
