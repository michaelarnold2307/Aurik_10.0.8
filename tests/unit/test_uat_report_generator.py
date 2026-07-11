import pytest

"""Unit tests for UAT report generator marker parsing."""

from __future__ import annotations

from pathlib import Path

from audit.uat_report_generator import ResultStatus, UATReportGenerator


@pytest.mark.unit
def test_parse_pytest_output_prefers_machine_readable_marker(tmp_path: Path) -> None:
    generator = UATReportGenerator(output_dir=tmp_path, json_output=tmp_path / "uat.json")
    output = "\n".join(
        [
            'tests/test_uat_acceptance_criteria.py::test_restoration_criteria[R5] UAT_RESULT_JSON:{"criterion_id": "R5", "evidence": "segmentiertes Stereo ok", "kind": "restoration", "notes": "worst segment 2 corr 0.41->0.44", "result": "PASS", "timestamp": ""}',
            "PASSED",
        ]
    )

    generator.parse_pytest_output(output)

    assert len(generator.restoration_results) == 1
    result = generator.restoration_results[0]
    assert result.criterion_id == "R5"
    assert result.status == ResultStatus.PASSED
    assert result.evidence == "segmentiertes Stereo ok"
    assert result.notes == "worst segment 2 corr 0.41->0.44"


def test_scorecard_cell_combines_evidence_and_notes() -> None:
    cell = UATReportGenerator._scorecard_cell(
        "Real-Audio Musical Goals segmentiert gemessen",
        "worst segment 1 goal=artikulation delta=-0.021",
        limit=200,
    )

    assert "segmentiert" in cell
    assert "worst segment 1" in cell
