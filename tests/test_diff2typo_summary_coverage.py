import os
import sys
from unittest.mock import MagicMock, patch
import pytest

# Ensure the repository root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import diff2typo

def test_should_enable_color_env_vars():
    with patch.dict(os.environ, {"NO_COLOR": "1"}):
        assert diff2typo._should_enable_color(sys.stdout) is False

    with patch.dict(os.environ, {"FORCE_COLOR": "1"}):
        # Even if not a tty, FORCE_COLOR should override
        mock_stream = MagicMock()
        mock_stream.isatty.return_value = False
        assert diff2typo._should_enable_color(mock_stream) is True

def test_render_visual_bar_various():
    # 0%
    bar0 = diff2typo._render_visual_bar(0, max_bar=10)
    assert len(bar0) == 10
    assert "█" not in bar0

    # 100%
    bar100 = diff2typo._render_visual_bar(100, max_bar=10)
    assert bar100 == "█" * 10

    # 50%
    bar50 = diff2typo._render_visual_bar(50, max_bar=10)
    assert bar50.startswith("█" * 5)
    assert len(bar50) == 10

def test_format_analysis_summary_unhashable():
    # Trigger line 186-187 (unique count exception)
    unhashable_items = [["a"], ["b"]]
    report = diff2typo._format_analysis_summary(
        raw_count=2,
        filtered_items=unhashable_items,
        use_color=False
    )
    # Just verify it didn't crash and contains the count
    report_text = "\n".join(report)
    assert "Unique items:                       2" in report_text

def test_format_analysis_summary_extra_metrics():
    # Trigger line 195-196 (extra metrics loop)
    extra = {"Extra Metric": "Value123"}
    report = diff2typo._format_analysis_summary(
        raw_count=1,
        filtered_items=["a"],
        extra_metrics=extra,
        use_color=False
    )
    report_text = "\n".join(report)
    assert "Extra Metric:                       Value123" in report_text

def test_format_analysis_summary_retention():
    # Trigger line 173-178
    report = diff2typo._format_analysis_summary(
        raw_count=10,
        filtered_items=["a"] * 5,
        use_color=False
    )
    report_text = "\n".join(report)
    assert "Retention rate:                      50.0%" in report_text

def test_detect_format_from_extension_edge_cases():
    allowed = ['arrow', 'csv']
    # line 224: no extension
    assert diff2typo._detect_format_from_extension("file", allowed, "default") == "default"
    # line 240: unknown extension
    assert diff2typo._detect_format_from_extension("file.unknown", allowed, "default") == "default"

def test_main_summary_labels(tmp_path, monkeypatch):
    # Coverage for lines 998, 1002-1003 in main()
    monkeypatch.chdir(tmp_path)

    # Setup files
    diff_file = tmp_path / "diff.txt"
    diff_file.write_text("--- a/f\n+++ b/f\n@@ -1,1 +1,1 @@\n-teh\n+the\n")

    # Case 1: mode=audit (line 998)
    monkeypatch.setattr(sys, "argv", [
        "diff2typo.py",
        str(diff_file),
        "--mode", "audit",
        "--dictionary", "missing.csv",
        "--allowed", "missing.csv"
    ])

    # Mock summary to capture stderr
    with patch("sys.stderr", new=MagicMock()) as mock_stderr:
        # We need to mock sys.exit because audit might find nothing or something and main continues
        # but here we just want to see if it reaches the summary block
        try:
            diff2typo.main()
        except SystemExit:
            pass

        # Check if "Unique audit-items found:" was written to stderr
        all_calls = "".join(call.args[0] for call in mock_stderr.write.call_args_list)
        assert "Unique audit-items found:" in all_calls

    # Case 2: mode=both (lines 1002-1003)
    monkeypatch.setattr(sys, "argv", [
        "diff2typo.py",
        str(diff_file),
        "--mode", "both",
        "--dictionary", "missing.csv",
        "--allowed", "missing.csv"
    ])

    with patch("sys.stderr", new=MagicMock()) as mock_stderr:
        try:
            diff2typo.main()
        except SystemExit:
            pass

        all_calls = "".join(call.args[0] for call in mock_stderr.write.call_args_list)
        assert "Typos found:" in all_calls
        assert "Corrections found:" in all_calls
