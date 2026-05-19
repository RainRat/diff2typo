import sys
import os
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import diff2typo

def test_should_enable_color_no_color():
    with patch.dict(os.environ, {"NO_COLOR": "1"}):
        assert diff2typo._should_enable_color(sys.stdout) is False

def test_should_enable_color_force_color():
    # Ensure NO_COLOR is not interfering
    with patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=True):
        assert diff2typo._should_enable_color(sys.stdout) is True

def test_render_visual_bar_full():
    bar = diff2typo._render_visual_bar(100.0, max_bar=10)
    assert bar == "█" * 10

def test_render_visual_bar_empty():
    bar = diff2typo._render_visual_bar(0.0, max_bar=10)
    # 0.0 percentage should result in 0 full blocks and fraction 0.0.
    # bar = "█" * 0 + blocks[0] + " " * (10 - 0 - 1)
    # blocks[0] is " "
    # So bar is " " + " " * 9 = " " * 10
    assert bar == " " * 10

def test_render_visual_bar_half():
    bar = diff2typo._render_visual_bar(50.0, max_bar=10)
    # (50.0 * 10) / 100 = 5.0
    # full_blocks = 5
    # fraction = 0.0
    # bar = "█" * 5 + " " + " " * (10 - 5 - 1) = "█████     "
    assert bar == "█" * 5 + " " * 5

def test_render_visual_bar_partial_block():
    bar = diff2typo._render_visual_bar(55.0, max_bar=10)
    # (55.0 * 10) / 100 = 5.5
    # full_blocks = 5
    # fraction = 0.5
    # frac_idx = int(0.5 * 8) = 4
    # blocks[4] is "▌"
    # bar = "█" * 5 + "▌" + " " * (10 - 5 - 1) = "█████▌    "
    assert "▌" in bar
    assert bar.startswith("█" * 5)

def test_format_analysis_summary_retention():
    report = diff2typo._format_analysis_summary(
        raw_count=10,
        filtered_items=["a", "b"],
        use_color=False
    )
    report_text = "\n".join(report)
    assert "Retention rate:" in report_text
    assert "20.0%" in report_text

def test_format_analysis_summary_unhashable():
    # Use unhashable items (like lists) to trigger the exception handler in line 186
    report = diff2typo._format_analysis_summary(
        raw_count=2,
        filtered_items=[["a"], ["b"]],
        use_color=False
    )
    report_text = "\n".join(report)
    assert "Unique items found:" in report_text
    # Should fall back to len(filtered_items) which is 2
    assert "Unique items:" in report_text
    assert " 2" in report_text

def test_format_analysis_summary_extra_metrics():
    report = diff2typo._format_analysis_summary(
        raw_count=1,
        filtered_items=["a"],
        extra_metrics={"Metric1": "Value1"},
        use_color=False
    )
    report_text = "\n".join(report)
    assert "Metric1:" in report_text
    assert "Value1" in report_text

def test_detect_format_from_extension_no_ext():
    # line 224
    assert diff2typo._detect_format_from_extension("output", ["csv", "arrow"], "arrow") == "arrow"

def test_detect_format_from_extension_unknown_ext():
    # line 240
    assert diff2typo._detect_format_from_extension("output.unknown", ["csv", "arrow"], "arrow") == "arrow"

def test_main_audit_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    diff_file = tmp_path / "diff.txt"
    diff_file.write_text("--- a/f\n+++ b/f\n@@ -1,1 +1,1 @@\n-correct\n+typo\n")

    dictionary_file = tmp_path / "words.csv"
    dictionary_file.write_text("correct\n")

    output_file = tmp_path / "output.txt"

    monkeypatch.setattr(sys, "argv", [
        "diff2typo.py",
        "--input", str(diff_file),
        "--output", str(output_file),
        "--dictionary", str(dictionary_file),
        "--mode", "audit",
        "--allowed", "allowed.csv"
    ])

    # Run main
    diff2typo.main()

    # Check stderr for "audit-item" (line 998)
    captured = capsys.readouterr()
    assert "Unique audit-items found:" in captured.err

def test_main_both_mode_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    diff_file = tmp_path / "diff.txt"
    diff_file.write_text("--- a/f\n+++ b/f\n@@ -1,3 +1,3 @@\n-typo1\n+correct1\n-typo2\n+correct2\n")

    dictionary_file = tmp_path / "words.csv"
    dictionary_file.write_text("typo1,correct1\n") # typo1 is a known typo, correct1 is correct

    output_file = tmp_path / "output.txt"

    monkeypatch.setattr(sys, "argv", [
        "diff2typo.py",
        "--input", str(diff_file),
        "--output", str(output_file),
        "--dictionary", str(dictionary_file),
        "--mode", "both",
        "--allowed", "allowed.csv"
    ])

    # Mock subprocess.run for filter_known_typos
    def mock_run(*args, **kwargs):
        return MagicMock(stdout="")
    monkeypatch.setattr(diff2typo.subprocess, "run", mock_run)

    diff2typo.main()

    captured = capsys.readouterr()
    # Lines 1002-1003
    assert "Typos found:" in captured.err
    assert "Corrections found:" in captured.err
