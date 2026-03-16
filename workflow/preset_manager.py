"""
Preset Manager für Aurik Workflow
- Speichern/Laden von Presets (Settings) als JSON
- Integration mit BatchJobConfig
- SOTA-konform, modular, API-ready
"""

import json
from pathlib import Path
from typing import Any, Dict


class PresetManager:
    def __init__(self, preset_dir: Path = Path("presets")):
        self.preset_dir = preset_dir
        self.preset_dir.mkdir(parents=True, exist_ok=True)

    def save_preset(self, name: str, settings: Dict[str, Any]) -> Path:
        path = self.preset_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(settings, f, indent=2)
        return path

    def load_preset(self, name: str) -> Dict[str, Any]:
        path = self.preset_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Preset {name} nicht gefunden")
        with open(path, "r") as f:
            return json.load(f)

    def list_presets(self) -> list:
        return [p.stem for p in self.preset_dir.glob("*.json")]
