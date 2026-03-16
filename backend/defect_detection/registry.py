"""
Defect Detector Registry
=========================

Central registry for managing defect detectors.
"""

from backend.defect_detection.base import DefectDetector, DefectType


class DefectDetectorRegistry:
    """
    Registry for managing defect detectors.

    Allows registration, lookup, and batch execution of detectors.
    """

    def __init__(self):
        self._detectors: dict[str, DefectDetector] = {}
        self._detectors_by_type: dict[DefectType, list[DefectDetector]] = {}

    def register(self, detector: DefectDetector) -> None:
        """
        Register a defect detector.

        Args:
            detector: Detector instance to register
        """
        self._detectors[detector.name] = detector

        if detector.defect_type not in self._detectors_by_type:
            self._detectors_by_type[detector.defect_type] = []

        self._detectors_by_type[detector.defect_type].append(detector)

    def get(self, name: str) -> DefectDetector | None:
        """Get detector by name."""
        return self._detectors.get(name)

    def get_by_type(self, defect_type: DefectType) -> list[DefectDetector]:
        """Get all detectors for a specific defect type."""
        return self._detectors_by_type.get(defect_type, [])

    def get_all(self) -> list[DefectDetector]:
        """Get all registered detectors."""
        return list(self._detectors.values())

    def list_names(self) -> list[str]:
        """List all registered detector names."""
        return list(self._detectors.keys())

    def list_types(self) -> list[DefectType]:
        """List all defect types with registered detectors."""
        return list(self._detectors_by_type.keys())


# Global singleton registry
_global_registry = DefectDetectorRegistry()


def get_global_registry() -> DefectDetectorRegistry:
    """Get the global defect detector registry."""
    return _global_registry
