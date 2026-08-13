import os
import sys
from unittest.mock import patch, MagicMock, mock_open
import pytest
from types import SimpleNamespace
import gentypos


def test_should_enable_color():
    mock_stream = MagicMock()
    mock_stream.isatty.return_value = True

    with patch.dict(os.environ, {"NO_COLOR": "1"}):
        assert gentypos._should_enable_color(mock_stream) is False

    with patch.dict(os.environ, {}, clear=True):
        if "NO_COLOR" in os.environ:
            del os.environ["NO_COLOR"]
        with patch.dict(os.environ, {"FORCE_COLOR": "1"}):
            assert gentypos._should_enable_color(mock_stream) is True

    with patch.dict(os.environ, {}, clear=True):
        if "NO_COLOR" in os.environ:
            del os.environ["NO_COLOR"]
        mock_stream.isatty.return_value = True
        assert gentypos._should_enable_color(mock_stream) is True

        mock_stream.isatty.return_value = False
        assert gentypos._should_enable_color(mock_stream) is False

        del mock_stream.isatty
        assert gentypos._should_enable_color(mock_stream) is False


def test_render_visual_bar():
    bar_100 = gentypos._render_visual_bar(100.0, max_bar=10)
    assert bar_100 == "█" * 10

    bar_0 = gentypos._render_visual_bar(0.0, max_bar=10)
    assert bar_0 == " " * 10

    bar_50 = gentypos._render_visual_bar(50.0, max_bar=10)
    assert bar_50.startswith("█" * 5)


def test_format_analysis_summary():
    report = gentypos._format_analysis_summary(
        raw_count=50,
        filtered_items=["one", "two"],
        item_label="typo",
        start_time=None,
        use_color=False,
        extra_metrics={"Custom Metric": "Custom Value"},
        title="TEST SUMMARY",
        total_input_items=10,
    )

    report_str = "\n".join(report)
    assert "TEST SUMMARY" in report_str
    assert "Total input words processed:" in report_str
    assert "Total typos generated:              50" in report_str
    assert "Unique typos after filtering:       2" in report_str
    assert "Custom Metric:                      Custom Value" in report_str
    assert "Retention rate:" in report_str


def test_main_displays_summary_and_logs(capsys, monkeypatch):
    test_args = ["gentypos.py", "hello", "--quiet"]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch("gentypos._format_analysis_summary", return_value=["SUMMARY HEADER", "SUMMARY BODY"]) as mock_summary, \
         patch("sys.stderr.write") as mock_stderr_write, \
         patch("logging.info") as mock_logging_info:

        gentypos.main()

        assert mock_summary.call_count == 0
        assert mock_stderr_write.call_count == 0

    test_args_verbose = ["gentypos.py", "hello"]
    monkeypatch.setattr(sys, "argv", test_args_verbose)

    with patch("sys.stderr.write") as mock_stderr_write, \
         patch("logging.info") as mock_logging_info:

        gentypos.main()

        assert mock_stderr_write.call_count >= 1
        written_stderr = "".join(call.args[0] for call in mock_stderr_write.call_args_list)
        assert "TYPO GENERATION SUMMARY" in written_stderr
        assert "Total input words processed" in written_stderr


def test_main_fallback_when_attributes_missing(capsys, monkeypatch):
    test_args = ["gentypos.py", "hello"]
    monkeypatch.setattr(sys, "argv", test_args)

    original_run_typo_generation = gentypos._run_typo_generation

    def mock_run_typo_generation(word_list, all_words, settings, adj_keys, custom_subs, quiet=False):
        return {"hallo": "hello"}

    monkeypatch.setattr(gentypos, "_run_typo_generation", mock_run_typo_generation)

    with patch("sys.stderr.write") as mock_stderr_write:
        gentypos.main()
        assert mock_stderr_write.call_count >= 1
        written_stderr = "".join(call.args[0] for call in mock_stderr_write.call_args_list)
        assert "TYPO GENERATION SUMMARY" in written_stderr

    monkeypatch.setattr(gentypos, "_run_typo_generation", original_run_typo_generation)
