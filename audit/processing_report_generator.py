"""
Processing Report Generator für Explainability & Transparency

Implements GAP #4: Comprehensive Processing Reports.
Ein production-ready System braucht Explainability: "Warum wurde X gemacht?"

Architecture:
1. ProcessingReport - Strukturierte Report-Daten
2. ReportSection - Einzelne Sections (Issues, Modules, Metrics, etc.)
3. ReportExporter - Export zu Markdown, JSON, PDF
4. ProcessingReportGenerator - Main API

Report Sections:
- Input Analysis: Original audio characteristics
- Detected Issues: Clicks, clipping, noise, etc.
- Applied Modules: Welche Module wurden verwendet und warum
- Processing Parameters: Alle verwendeten Parameter
- Before/After Metrics: SNR, THD, LUFS, Musical Goals, etc.
- Confidence Scores: Confidence in decisions
- Recommendations: Suggestions für weitere improvements

Author: AURIK Development Team
Version: 1.0
Date: 2026-02-10
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ReportSectionType(Enum):
    """Types of report sections."""

    INPUT_ANALYSIS = "input_analysis"
    """Initial audio analysis: Duration, SR, channels, format."""

    DETECTED_ISSUES = "detected_issues"
    """Detected defects: Clicks, noise, clipping, etc."""

    APPLIED_MODULES = "applied_modules"
    """Processing modules that were applied."""

    PROCESSING_PARAMETERS = "processing_parameters"
    """Detailed processing parameters."""

    BEFORE_AFTER_METRICS = "before_after_metrics"
    """Objective metrics before/after processing."""

    MUSICAL_GOALS = "musical_goals"
    """Musical goals evaluation (Klarheit, Wärme, etc.)."""

    CONFIDENCE_SCORES = "confidence_scores"
    """Confidence in processing decisions."""

    RECOMMENDATIONS = "recommendations"
    """Recommendations for further improvement."""

    WARNINGS = "warnings"
    """Warnings about potential issues."""


@dataclass
class ReportSection:
    """
    A section of a processing report.
    """

    section_type: ReportSectionType
    """Type of section."""

    title: str
    """Section title."""

    content: dict[str, Any]
    """Section content (structured data)."""

    summary: str = ""
    """Brief summary (1-2 sentences)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "section_type": self.section_type.value,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
        }


@dataclass
class ProcessingReport:
    """
    Comprehensive processing report.

    Contains all information about the processing:
    - What was detected?
    - What was done?
    - Why was it done?
    - What were the results?
    - How confident are we?
    """

    # === Metadata ===
    report_id: str
    """Unique report ID."""

    timestamp: str
    """ISO 8601 timestamp."""

    aurik_version: str = "1.0.0"
    """AURIK version."""

    # === Input Info ===
    input_file: str = ""
    """Input filename."""

    output_file: str = ""
    """Output filename."""

    # === Sections ===
    sections: list[ReportSection] = field(default_factory=list)
    """Report sections."""

    # === Summary ===
    overall_summary: str = ""
    """Overall processing summary."""

    overall_confidence: float = 0.0
    """Overall confidence (0.0-1.0)."""

    processing_time_sec: float = 0.0
    """Total processing time in seconds."""

    def add_section(self, section: ReportSection):
        """Add a section to the report."""
        self.sections.append(section)

    def get_section(self, section_type: ReportSectionType) -> ReportSection | None:
        """Get a section by type."""
        for section in self.sections:
            if section.section_type == section_type:
                return section
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "aurik_version": self.aurik_version,
            "input_file": self.input_file,
            "output_file": self.output_file,
            "overall_summary": self.overall_summary,
            "overall_confidence": self.overall_confidence,
            "processing_time_sec": self.processing_time_sec,
            "sections": [s.to_dict() for s in self.sections],
        }


class ReportExporter:
    """
    Export ProcessingReport to various formats.

    Supported formats:
    - Markdown (.md) - Human-readable
    - JSON (.json) - Machine-readable
    - (PDF - future)
    """

    @staticmethod
    def export_markdown(report: ProcessingReport, output_path: Path) -> bool:
        """
        Export report als Markdown.

        Args:
            report: ProcessingReport
            output_path: Output .md file

        Returns:
            True if successful
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                # Header
                f.write(f"# AURIK Audio Restoration Report\n\n")
                f.write(f"**Report ID:** `{report.report_id}`  \n")
                f.write(f"**Timestamp:** {report.timestamp}  \n")
                f.write(f"**AURIK Version:** {report.aurik_version}  \n")
                f.write(f"**Processing Time:** {report.processing_time_sec:.2f} seconds  \n")
                f.write(f"\n---\n\n")

                # Files
                if report.input_file:
                    f.write(f"**Input File:** `{report.input_file}`  \n")
                if report.output_file:
                    f.write(f"**Output File:** `{report.output_file}`  \n")
                f.write(f"\n")

                # Overall Summary
                f.write(f"## Overall Summary\n\n")
                f.write(f"{report.overall_summary}\n\n")
                f.write(f"**Confidence:** {report.overall_confidence:.1%}  \n")
                f.write(f"\n---\n\n")

                # Sections
                for section in report.sections:
                    f.write(f"## {section.title}\n\n")

                    if section.summary:
                        f.write(f"_{section.summary}_\n\n")

                    # Format content based on section type
                    if section.section_type == ReportSectionType.INPUT_ANALYSIS:
                        ReportExporter._write_input_analysis_md(f, section.content)

                    elif section.section_type == ReportSectionType.DETECTED_ISSUES:
                        ReportExporter._write_detected_issues_md(f, section.content)

                    elif section.section_type == ReportSectionType.APPLIED_MODULES:
                        ReportExporter._write_applied_modules_md(f, section.content)

                    elif section.section_type == ReportSectionType.PROCESSING_PARAMETERS:
                        ReportExporter._write_processing_params_md(f, section.content)

                    elif section.section_type == ReportSectionType.BEFORE_AFTER_METRICS:
                        ReportExporter._write_before_after_metrics_md(f, section.content)

                    elif section.section_type == ReportSectionType.MUSICAL_GOALS:
                        ReportExporter._write_musical_goals_md(f, section.content)

                    elif section.section_type == ReportSectionType.CONFIDENCE_SCORES:
                        ReportExporter._write_confidence_scores_md(f, section.content)

                    elif section.section_type == ReportSectionType.RECOMMENDATIONS:
                        ReportExporter._write_recommendations_md(f, section.content)

                    elif section.section_type == ReportSectionType.WARNINGS:
                        ReportExporter._write_warnings_md(f, section.content)

                    else:
                        # Generic content
                        f.write(f"```json\n")
                        f.write(json.dumps(section.content, indent=2))
                        f.write(f"\n```\n\n")

                    f.write(f"\n---\n\n")

                # Footer
                f.write(f"\n*Report generated by AURIK v{report.aurik_version}*\n")

            logger.info(f"✅ Markdown report exported: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Markdown export failed: {e}")
            return False

    @staticmethod
    def _write_input_analysis_md(f, content: dict):
        """Write Input Analysis section."""
        f.write(f"| Property | Value |\n")
        f.write(f"|----------|-------|\n")
        for key, value in content.items():
            f.write(f"| {key} | {value} |\n")
        f.write(f"\n")

    @staticmethod
    def _write_detected_issues_md(f, content: dict):
        """Write Detected Issues section."""
        issues = content.get("issues", [])

        if not issues:
            f.write(f"✅ No significant issues detected.\n\n")
            return

        f.write(f"Detected {len(issues)} issue(s):\n\n")
        for issue in issues:
            # Handle plain string issues (e.g. ["noise", "clipping"])
            if isinstance(issue, str):
                f.write(f"- ⚠️ {issue}\n\n")
                continue
            severity = issue.get("severity", "MEDIUM")
            issue_type = issue.get("type", "Unknown")
            description = issue.get("description", "")
            confidence = issue.get("confidence", 0.0)

            emoji = "🔴" if severity == "HIGH" else "🟡" if severity == "MEDIUM" else "🟢"

            f.write(f"- {emoji} **{issue_type}** ({severity}, {confidence:.0%} confidence)\n")
            f.write(f"  _{description}_\n\n")

    @staticmethod
    def _write_applied_modules_md(f, content: dict):
        """Write Applied Modules section."""
        modules = content.get("modules", [])

        if not modules:
            f.write(f"No processing modules applied.\n\n")
            return

        f.write(f"Applied {len(modules)} processing module(s):\n\n")
        for i, module in enumerate(modules, 1):
            name = module.get("name", "Unknown")
            reason = module.get("reason", "")
            strength = module.get("strength", 0.0)

            f.write(f"{i}. **{name}** (strength: {strength:.0%})\n")
            f.write(f"   - Reason: {reason}\n\n")

    @staticmethod
    def _write_processing_params_md(f, content: dict):
        """Write Processing Parameters section."""
        f.write(f"```yaml\n")
        for key, value in content.items():
            f.write(f"{key}: {value}\n")
        f.write(f"```\n\n")

    @staticmethod
    def _write_before_after_metrics_md(f, content: dict):
        """Write Before/After Metrics section."""
        before = content.get("before", {})
        after = content.get("after", {})
        improvement = content.get("improvement", {})

        f.write(f"| Metric | Before | After | Improvement |\n")
        f.write(f"|--------|--------|-------|-------------|\n")

        for metric in sorted(before.keys()):
            b_val = before.get(metric, 0.0)
            a_val = after.get(metric, 0.0)
            imp = improvement.get(metric, 0.0)

            imp_str = f"+{imp:.1f}" if imp > 0 else f"{imp:.1f}"
            f.write(f"| {metric} | {b_val:.2f} | {a_val:.2f} | {imp_str} |\n")
        f.write(f"\n")

    @staticmethod
    def _write_musical_goals_md(f, content: dict):
        """Write Musical Goals section."""
        goals = content.get("goals", {})

        f.write(f"| Goal | Score | Status |\n")
        f.write(f"|------|-------|--------|\n")

        for goal_name, score in goals.items():
            status = "✅ Excellent" if score > 0.8 else "✓ Good" if score > 0.6 else "⚠ Fair"
            f.write(f"| {goal_name} | {score:.2f} | {status} |\n")
        f.write(f"\n")

    @staticmethod
    def _write_confidence_scores_md(f, content: dict):
        """Write Confidence Scores section."""
        scores = content.get("scores", {})

        for decision, confidence in scores.items():
            conf_str = f"{confidence:.0%}"
            emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
            f.write(f"- {emoji} **{decision}**: {conf_str}\n")
        f.write(f"\n")

    @staticmethod
    def _write_recommendations_md(f, content: dict):
        """Write Recommendations section."""
        recommendations = content.get("recommendations", [])

        if not recommendations:
            f.write(f"No additional recommendations.\n\n")
            return

        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")
        f.write(f"\n")

    @staticmethod
    def _write_warnings_md(f, content: dict):
        """Write Warnings section."""
        warnings = content.get("warnings", [])

        if not warnings:
            f.write(f"No warnings.\n\n")
            return

        for warning in warnings:
            f.write(f"⚠️ {warning}\n\n")

    @staticmethod
    def export_json(report: ProcessingReport, output_path: Path) -> bool:
        """
        Export report als JSON.

        Args:
            report: ProcessingReport
            output_path: Output .json file

        Returns:
            True if successful
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"✅ JSON report exported: {output_path}")
            return True

        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return False


class ProcessingReportGenerator:
    """
    Main API für Processing Report Generation.

    Usage:
        generator = ProcessingReportGenerator()

        report = generator.create_report(
            input_audio=audio,
            output_audio=result_audio,
            processing_history=history,
            input_file="input.wav",
            output_file="output.wav"
        )

        generator.export_report(report, "report.md", format="markdown")
    """

    def create_report(
        self,
        input_audio: np.ndarray,
        output_audio: np.ndarray,
        sample_rate: int,
        processing_history: dict[str, Any] | None = None,
        input_file: str = "",
        output_file: str = "",
        processing_time_sec: float = 0.0,
    ) -> ProcessingReport:
        """
        Create a comprehensive processing report.

        Args:
            input_audio: Original audio
            output_audio: Processed audio
            sample_rate: Sample rate
            processing_history: Dict mit applied modules, parameters, etc.
            input_file: Input filename
            output_file: Output filename
            processing_time_sec: Processing time

        Returns:
            ProcessingReport
        """
        # Generate report ID
        report_id = f"AURIK_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now().isoformat()

        report = ProcessingReport(
            report_id=report_id,
            timestamp=timestamp,
            input_file=input_file,
            output_file=output_file,
            processing_time_sec=processing_time_sec,
        )

        # === 1. Input Analysis ===
        input_section = self._create_input_analysis_section(input_audio, sample_rate)
        report.add_section(input_section)

        # === 2. Detected Issues ===
        if processing_history and "detected_issues" in processing_history:
            issues_section = self._create_detected_issues_section(processing_history["detected_issues"])
            report.add_section(issues_section)

        # === 3. Applied Modules ===
        _modules_raw = None
        if processing_history:
            _modules_raw = processing_history.get("applied_modules") or processing_history.get("modules_applied")
        if _modules_raw is not None:
            # Normalize: convert plain strings to dicts so _write_applied_modules_md works
            _modules_dicts = [
                m if isinstance(m, dict) else {"name": str(m), "reason": "", "strength": 1.0}
                for m in (_modules_raw if isinstance(_modules_raw, list) else [])
            ]
            modules_section = self._create_applied_modules_section(_modules_dicts)
            report.add_section(modules_section)

        # === 4. Processing Parameters ===
        _params_raw = None
        if processing_history:
            _params_raw = processing_history.get("parameters") or processing_history.get("processing_parameters")
        if _params_raw is not None:
            params_section = self._create_processing_params_section(
                _params_raw if isinstance(_params_raw, dict) else {"info": str(_params_raw)}
            )
            report.add_section(params_section)

        # === 5. Before/After Metrics ===
        metrics_section = self._create_before_after_metrics_section(input_audio, output_audio, sample_rate)
        report.add_section(metrics_section)

        # === 6. Musical Goals ===
        try:
            musical_goals_section = self._create_musical_goals_section(output_audio, sample_rate)
            report.add_section(musical_goals_section)
        except Exception as e:
            logger.warning(f"Musical goals evaluation failed: {e}")

        # === 7. Confidence Scores ===
        if processing_history and "confidence_scores" in processing_history:
            confidence_section = self._create_confidence_section(processing_history["confidence_scores"])
            report.add_section(confidence_section)

        # === 8. Recommendations ===
        recommendations_section = self._create_recommendations_section(input_audio, output_audio, sample_rate)
        report.add_section(recommendations_section)

        # === Overall Summary ===
        report.overall_summary = self._generate_overall_summary(report)
        report.overall_confidence = self._calculate_overall_confidence(report)

        return report

    def _create_input_analysis_section(self, audio: np.ndarray, sample_rate: int) -> ReportSection:
        """Create Input Analysis section."""
        duration_sec = len(audio) / sample_rate
        channels = 2 if audio.ndim == 2 else 1
        peak_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
        rms_db = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-10)

        content = {
            "Duration": f"{duration_sec:.2f} seconds",
            "Sample Rate": f"{sample_rate} Hz",
            "Channels": "Stereo" if channels == 2 else "Mono",
            "Peak Level": f"{peak_db:.1f} dBFS",
            "RMS Level": f"{rms_db:.1f} dBFS",
            "Dynamic Range": f"{peak_db - rms_db:.1f} dB",
        }

        return ReportSection(
            section_type=ReportSectionType.INPUT_ANALYSIS,
            title="Input Audio Analysis",
            content=content,
            summary=f"{duration_sec:.1f}s {channels}ch audio at {sample_rate}Hz",
        )

    def _create_detected_issues_section(self, issues: list[dict[str, Any]]) -> ReportSection:
        """Create Detected Issues section."""
        return ReportSection(
            section_type=ReportSectionType.DETECTED_ISSUES,
            title="Detected Issues",
            content={"issues": issues},
            summary=f"Detected {len(issues)} issue(s) requiring processing",
        )

    def _create_applied_modules_section(self, modules: list[dict[str, Any]]) -> ReportSection:
        """Create Applied Modules section."""
        return ReportSection(
            section_type=ReportSectionType.APPLIED_MODULES,
            title="Applied Processing Modules",
            content={"modules": modules},
            summary=f"Applied {len(modules)} processing module(s)",
        )

    def _create_processing_params_section(self, parameters: dict[str, Any]) -> ReportSection:
        """Create Processing Parameters section."""
        return ReportSection(
            section_type=ReportSectionType.PROCESSING_PARAMETERS,
            title="Processing Parameters",
            content=parameters,
            summary="Detailed processing configuration",
        )

    def _create_before_after_metrics_section(
        self, input_audio: np.ndarray, output_audio: np.ndarray, sample_rate: int
    ) -> ReportSection:
        """Create Before/After Metrics section."""
        # Simplified metrics (real implementation würde enhanced_metrics nutzen)

        # Peak
        before_peak = 20 * np.log10(np.max(np.abs(input_audio)) + 1e-10)
        after_peak = 20 * np.log10(np.max(np.abs(output_audio)) + 1e-10)

        # RMS
        before_rms = 20 * np.log10(np.sqrt(np.mean(input_audio**2)) + 1e-10)
        after_rms = 20 * np.log10(np.sqrt(np.mean(output_audio**2)) + 1e-10)

        before = {"Peak_dBFS": before_peak, "RMS_dBFS": before_rms}

        after = {"Peak_dBFS": after_peak, "RMS_dBFS": after_rms}

        improvement = {"Peak_dBFS": after_peak - before_peak, "RMS_dBFS": after_rms - before_rms}

        return ReportSection(
            section_type=ReportSectionType.BEFORE_AFTER_METRICS,
            title="Before/After Metrics",
            content={"before": before, "after": after, "improvement": improvement},
            summary="Objective metric comparison",
        )

    def _create_musical_goals_section(self, audio: np.ndarray, sample_rate: int) -> ReportSection:
        """Create Musical Goals section."""
        try:
            from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

            checker = MusicalGoalsChecker()
            goals = checker.measure_all(audio, sample_rate)

            avg_score = np.mean(list(goals.values()))

            return ReportSection(
                section_type=ReportSectionType.MUSICAL_GOALS,
                title="Musical Goals Evaluation",
                content={"goals": goals},
                summary=f"Average musical goals score: {avg_score:.2f}",
            )
        except Exception:
            return ReportSection(
                section_type=ReportSectionType.MUSICAL_GOALS,
                title="Musical Goals Evaluation",
                content={"goals": {}},
                summary="Musical goals evaluation not available",
            )

    def _create_confidence_section(self, confidence_scores: dict[str, float]) -> ReportSection:
        """Create Confidence Scores section."""
        avg_confidence = np.mean(list(confidence_scores.values()))

        return ReportSection(
            section_type=ReportSectionType.CONFIDENCE_SCORES,
            title="Confidence Scores",
            content={"scores": confidence_scores},
            summary=f"Average confidence: {avg_confidence:.0%}",
        )

    def _create_recommendations_section(
        self, input_audio: np.ndarray, output_audio: np.ndarray, sample_rate: int
    ) -> ReportSection:
        """Create Recommendations section."""
        recommendations = []

        # Check peak levels
        peak_db = 20 * np.log10(np.max(np.abs(output_audio)) + 1e-10)
        if peak_db > -1.0:
            recommendations.append(
                f"Consider True Peak limiting: Current peak is {peak_db:.1f} dBFS (target: < -1.0 dBTP)"
            )

        # Check RMS levels
        rms_db = 20 * np.log10(np.sqrt(np.mean(output_audio**2)) + 1e-10)
        if rms_db < -30.0:
            recommendations.append(f"Audio is very quiet (RMS: {rms_db:.1f} dBFS). Consider loudness normalization.")

        if not recommendations:
            recommendations.append("Audio meets professional standards. No additional processing recommended.")

        return ReportSection(
            section_type=ReportSectionType.RECOMMENDATIONS,
            title="Recommendations",
            content={"recommendations": recommendations},
            summary=f"{len(recommendations)} recommendation(s)",
        )

    def _generate_overall_summary(self, report: ProcessingReport) -> str:
        """Generate overall summary from report sections."""
        summaries = []

        for section in report.sections:
            if section.summary:
                summaries.append(section.summary)

        return " • ".join(summaries[:3])  # First 3 summaries

    def _calculate_overall_confidence(self, report: ProcessingReport) -> float:
        """Calculate overall confidence from confidence scores."""
        confidence_section = report.get_section(ReportSectionType.CONFIDENCE_SCORES)

        if confidence_section and "scores" in confidence_section.content:
            scores = confidence_section.content["scores"]
            if scores:
                return float(np.mean(list(scores.values())))

        return 0.8  # Default medium-high confidence

    def export_report(self, report: ProcessingReport, output_path: Path, format: str = "markdown") -> bool:
        """
        Export report to file.

        Args:
            report: ProcessingReport
            output_path: Output file path
            format: "markdown" or "json"

        Returns:
            True if successful
        """
        output_path = Path(output_path)

        if format.lower() == "markdown":
            return ReportExporter.export_markdown(report, output_path)
        elif format.lower() == "json":
            return ReportExporter.export_json(report, output_path)
        else:
            logger.error(f"Unknown export format: {format}")
            return False


# === Example Usage ===
if __name__ == "__main__":
    import soundfile as sf

    # Load audio
    input_audio, sr = sf.read("test_audio/test_input.wav")
    output_audio, _ = sf.read("test_output/processed.wav")

    # Mock processing history
    history = {
        "detected_issues": [
            {
                "type": "Background Noise",
                "severity": "MEDIUM",
                "description": "Moderate background noise detected (SNR: 18 dB)",
                "confidence": 0.85,
            },
            {
                "type": "Click Artifacts",
                "severity": "LOW",
                "description": "3 click artifacts detected",
                "confidence": 0.75,
            },
        ],
        "applied_modules": [
            {"name": "Spectral Denoiser", "reason": "Reduce background noise", "strength": 0.30},
            {"name": "Click Removal", "reason": "Remove detected click artifacts", "strength": 0.50},
        ],
        "parameters": {
            "processing_mode": "RESTORATION",
            "denoise_strength": 0.30,
            "click_removal_sensitivity": 0.50,
            "preserve_breaths": True,
        },
        "confidence_scores": {"Noise Detection": 0.85, "Click Detection": 0.75, "Overall Processing": 0.90},
    }

    # Generate report
    generator = ProcessingReportGenerator()

    report = generator.create_report(
        input_audio=input_audio,
        output_audio=output_audio,
        sample_rate=sr,
        processing_history=history,
        input_file="test_input.wav",
        output_file="processed.wav",
        processing_time_sec=12.5,
    )

    # Export Markdown
    generator.export_report(report, Path("test_output/processing_report.md"), format="markdown")

    # Export JSON
    generator.export_report(report, Path("test_output/processing_report.json"), format="json")

    print(f"\n✅ Processing Report Generated")
    print(f"   Report ID: {report.report_id}")
    print(f"   Sections: {len(report.sections)}")
    print(f"   Overall Confidence: {report.overall_confidence:.0%}")
