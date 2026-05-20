import os
import sys
from unittest.mock import MagicMock, patch

import diff2typo

def test_should_enable_color_env_vars():
    with patch.dict(os.environ, {"NO_COLOR": "1"}):
        assert diff2typo._should_enable_color(sys.stderr) is False

    with patch.dict(os.environ, {"FORCE_COLOR": "1"}):
        assert diff2typo._should_enable_color(sys.stderr) is True

def test_render_visual_bar():
    assert diff2typo._render_visual_bar(0, 10) == "          "
    assert diff2typo._render_visual_bar(100, 10) == "██████████"
    assert diff2typo._render_visual_bar(50, 10) == "█████     "
    bar_55 = diff2typo._render_visual_bar(55, 10)
    assert bar_55.startswith("█████")
    assert len(bar_55) == 10

def test_format_analysis_summary_retention_and_metrics():
    summary_lines = diff2typo._format_analysis_summary(
        raw_count=100,
        filtered_items=["item1", "item2"],
        item_label="typo",
        use_color=False
    )
    summary_text = "\n".join(summary_lines)
    assert "Retention rate:" in summary_text
    assert "2.0%" in summary_text

    extra = {"Extra Metric": "Value"}
    summary_with_extra_lines = diff2typo._format_analysis_summary(
        raw_count=100,
        filtered_items=["item1"],
        item_label="typo",
        extra_metrics=extra,
        use_color=False
    )
    summary_with_extra_text = "\n".join(summary_with_extra_lines)
    assert "Extra Metric:" in summary_with_extra_text
    assert "Value" in summary_with_extra_text

def test_format_analysis_summary_unhashable():
    unhashable_items = [["list1"], ["list2"]]
    summary_lines = diff2typo._format_analysis_summary(
        raw_count=2,
        filtered_items=unhashable_items,
        item_label="typo",
        use_color=False
    )
    summary_text = "\n".join(summary_lines)
    assert "Unique typos:                       2" in summary_text

def test_detect_format_from_extension_edges():
    assert diff2typo._detect_format_from_extension("file", ["arrow"], "arrow") == "arrow"
    assert diff2typo._detect_format_from_extension("file.unknown", ["arrow"], "arrow") == "arrow"

def test_main_summary_labels_audit_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with patch("diff2typo._read_diff_sources", return_value=""), \
         patch("diff2typo.read_words_mapping", return_value={}), \
         patch("diff2typo.read_allowed_words", return_value=set()), \
         patch("diff2typo.process_audit_typos", return_value=["item1"]), \
         patch("diff2typo.smart_open_output") as mock_open:

        args = MagicMock()
        args.mode = "audit"
        args.quiet = False
        args.output_file = "-"
        args.output_format = "text"
        args.min_length = 2
        args.max_dist = 2
        args.dictionary_file = "words.csv"
        args.allowed_file = "allowed.csv"
        args.git = None
        args.input_files = []

        with patch("argparse.ArgumentParser.parse_args", return_value=args), \
             patch("diff2typo._format_analysis_summary") as mock_summary:

            with patch("diff2typo.find_typos", return_value=["a -> b"]):
                diff2typo.main()

            mock_summary.assert_called()
            kwargs = mock_summary.call_args.kwargs
            assert kwargs["item_label"] == "audit-item"

def test_main_summary_metrics_both_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with patch("diff2typo._read_diff_sources", return_value=""), \
         patch("diff2typo.read_words_mapping", return_value={}), \
         patch("diff2typo.read_allowed_words", return_value=set()), \
         patch("diff2typo.process_typos_mode", return_value=["typo1"]), \
         patch("diff2typo.process_corrections_mode", return_value=["corr1"]), \
         patch("diff2typo.smart_open_output") as mock_open:

        args = MagicMock()
        args.mode = "both"
        args.quiet = False
        args.output_file = "-"
        args.output_format = "text"
        args.min_length = 2
        args.max_dist = 2
        args.dictionary_file = "words.csv"
        args.allowed_file = "allowed.csv"
        args.git = None
        args.input_files = []

        with patch("argparse.ArgumentParser.parse_args", return_value=args), \
             patch("diff2typo._format_analysis_summary") as mock_summary:

            with patch("diff2typo.find_typos", return_value=["a -> b"]):
                diff2typo.main()

            mock_summary.assert_called()
            kwargs = mock_summary.call_args.kwargs
            assert "Typos found" in kwargs["extra_metrics"]
            assert "Corrections found" in kwargs["extra_metrics"]
            assert kwargs["extra_metrics"]["Typos found"] == 1
            assert kwargs["extra_metrics"]["Corrections found"] == 1
