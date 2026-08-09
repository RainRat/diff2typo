import io
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos


def test_should_enable_color():
    # Test NO_COLOR environment variable
    with patch.dict(os.environ, {"NO_COLOR": "1"}):
        assert not gentypos._should_enable_color(sys.stderr)

    # Test FORCE_COLOR environment variable
    with patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=True):
        assert gentypos._should_enable_color(sys.stderr)

    # Test stream has no isatty
    with patch.dict(os.environ, {}, clear=True):
        mock_stream = MagicMock(spec=[])
        assert not gentypos._should_enable_color(mock_stream)

    # Test stream has isatty but returns False
    with patch.dict(os.environ, {}, clear=True):
        mock_stream = MagicMock()
        mock_stream.isatty.return_value = False
        assert not gentypos._should_enable_color(mock_stream)

    # Test stream has isatty and returns True
    with patch.dict(os.environ, {}, clear=True):
        mock_stream = MagicMock()
        mock_stream.isatty.return_value = True
        assert gentypos._should_enable_color(mock_stream)


def test_render_visual_bar():
    # 0% retention
    bar = gentypos._render_visual_bar(0.0)
    assert "█" not in bar
    assert len(bar) == 20

    # 100% retention
    bar = gentypos._render_visual_bar(100.0)
    assert bar == "█" * 20

    # 50% retention
    bar = gentypos._render_visual_bar(50.0)
    assert bar.startswith("█" * 10)
    assert len(bar) == 20


def test_format_analysis_summary():
    # Test format summary without color
    summary = gentypos._format_analysis_summary(
        raw_count=10,
        filtered_items=["a", "b", "c"],
        item_label="typo",
        start_time=time.perf_counter() - 0.5,
        use_color=False,
        extra_metrics={"Test Metric": "Value"},
        title="TEST SUMMARY",
        total_input_items=5
    )
    summary_text = "\n".join(summary)
    assert "TEST SUMMARY" in summary_text
    assert "Input words processed:" in summary_text
    assert "Unique typos generated:" in summary_text
    assert "Total typos after filtering:" in summary_text
    assert "Retention rate:" in summary_text
    assert "Test Metric:" in summary_text
    assert "Processing time:" in summary_text

    # Test format summary with color
    summary_colored = gentypos._format_analysis_summary(
        raw_count=10,
        filtered_items=["a", "b", "c"],
        item_label="typo",
        use_color=True,
        extra_metrics={"Test Metric": "Value"},
        title="TEST SUMMARY"
    )
    summary_colored_text = "\n".join(summary_colored)
    assert "\033[1;34m" in summary_colored_text


def test_main_with_reporting_summary(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gentypos.py", "hello", "--no-filter", "-f", "arrow"])

    # Mock os.path.exists to return False for config, input, and dictionary files
    monkeypatch.setattr("os.path.exists", lambda path: False)

    # Use a dummy word generation that succeeds
    with patch("gentypos._run_typo_generation", return_value={"hllo": "hello"}) as mock_gen:
        # Mock _run_typo_generation to have the attribute
        mock_gen.total_generated = 10
        with patch("logging.info") as mock_log_info:
            gentypos.main()

            captured = capsys.readouterr()
            # stdout should have the typo format
            assert "hllo -> hello" in captured.out
            # stderr should have the summary blocks
            assert "TYPO GENERATION SUMMARY" in captured.err
            assert "Retention rate:" in captured.err

            # Verify logging.info was called with the success message
            assert any("Wrote 1 line(s)" in call[0][0] for call in mock_log_info.call_args_list)
