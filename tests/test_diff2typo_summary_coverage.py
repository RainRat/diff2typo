import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Ensure the repository root is in the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import diff2typo

def test_should_enable_color():
    # Test NO_COLOR
    with patch.dict(os.environ, {"NO_COLOR": "1"}):
        assert diff2typo._should_enable_color(sys.stdout) is False

    # Test FORCE_COLOR
    with patch.dict(os.environ, {"FORCE_COLOR": "1"}):
        assert diff2typo._should_enable_color(sys.stdout) is True

    # Test isatty
    with patch.dict(os.environ, {}, clear=True):
        mock_stream = MagicMock()
        mock_stream.isatty.return_value = True
        assert diff2typo._should_enable_color(mock_stream) is True

        mock_stream.isatty.return_value = False
        assert diff2typo._should_enable_color(mock_stream) is False

def test_render_visual_bar():
    # Test 0%
    assert "█" not in diff2typo._render_visual_bar(0, max_bar=10)
    # Test 100%
    assert diff2typo._render_visual_bar(100, max_bar=10) == "█" * 10
    # Test 50%
    bar_50 = diff2typo._render_visual_bar(50, max_bar=10)
    assert bar_50.startswith("█████")
    # Test fractional (e.g., 55% of 10 blocks = 5.5 blocks)
    bar_55 = diff2typo._render_visual_bar(55, max_bar=10)
    assert "█" * 5 in bar_55
    assert len(bar_55) == 10

def test_format_analysis_summary_enhanced():
    # Test with raw_count > 0 and color
    items = ["typo1", "typo2"]
    report = diff2typo._format_analysis_summary(
        raw_count=10,
        filtered_items=items,
        item_label="typo",
        use_color=True,
        start_time=0.0 # Just to trigger branch, perf_counter will be subtracted from this
    )
    report_str = "\n".join(report)
    assert "Retention rate:" in report_str
    assert "\033[" in report_str  # Check for ANSI colors

    # Test with unhashable items
    unhashable_items = [["a"], ["b"]]
    report_unhashable = diff2typo._format_analysis_summary(
        raw_count=2,
        filtered_items=unhashable_items,
        item_label="item"
    )
    assert "Unique items:" in "\n".join(report_unhashable)

    # Test with extra_metrics
    extra = {"Custom Metric": "Value"}
    report_extra = diff2typo._format_analysis_summary(
        raw_count=1,
        filtered_items=["item"],
        extra_metrics=extra
    )
    assert "Custom Metric:" in "\n".join(report_extra)
    assert "Value" in "\n".join(report_extra)

def test_detect_format_from_extension_coverage():
    allowed = ["csv", "arrow", "table"]
    # Test no path
    assert diff2typo._detect_format_from_extension("", allowed, "default") == "default"
    # Test no extension
    assert diff2typo._detect_format_from_extension("filename", allowed, "default") == "default"
    # Test unsupported extension
    assert diff2typo._detect_format_from_extension("file.unknown", allowed, "default") == "default"
    # Test supported extension but not in allowed
    assert diff2typo._detect_format_from_extension("file.list", allowed, "default") == "default"

def test_main_audit_and_both_modes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    # Setup minimal files for main to run
    diff_file = tmp_path / "test.diff"
    diff_file.write_text("--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n")

    # Mode: audit
    monkeypatch.setattr(sys, "argv", [
        "diff2typo.py",
        "--input", str(diff_file),
        "--mode", "audit"
    ])

    # Mocking components to avoid full execution complexity
    with patch("diff2typo._read_diff_sources", return_value="--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n"), \
         patch("diff2typo.read_allowed_words", return_value=set()), \
         patch("diff2typo.read_words_mapping", return_value={}), \
         patch("diff2typo.process_audit_typos", return_value=["audit-typo"]), \
         patch("sys.stderr", new_callable=MagicMock) as mock_stderr:

        diff2typo.main()

        # Check if "audit-item" was used (captured in stderr)
        combined_stderr = "".join(call.args[0] for call in mock_stderr.write.call_args_list)
        # Note: it might be "audit-items" because of pluralization
        assert "audit-item" in combined_stderr or "audit-items" in combined_stderr

    # Mode: both
    monkeypatch.setattr(sys, "argv", [
        "diff2typo.py",
        "--input", str(diff_file),
        "--mode", "both"
    ])

    with patch("diff2typo._read_diff_sources", return_value="--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n"), \
         patch("diff2typo.read_allowed_words", return_value=set()), \
         patch("diff2typo.read_words_mapping", return_value={}), \
         patch("diff2typo.find_typos", return_value=["typo1 -> corr1"]), \
         patch("diff2typo.process_corrections_mode", return_value=["typo2 -> corr2"]), \
         patch("sys.stderr", new_callable=MagicMock) as mock_stderr:

        diff2typo.main()

        combined_stderr = "".join(call.args[0] for call in mock_stderr.write.call_args_list)
        assert "Typos found" in combined_stderr
        assert "Corrections found" in combined_stderr
