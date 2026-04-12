import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

def test_format_analysis_summary_unhashable_items_fallback():
    unhashable_items = [["a"], ["b"], ["a"]]
    report = multitool._format_analysis_summary(3, unhashable_items, "item", None, False)

    full_report = "".join(report)
    assert "Unique items:" in full_report
    assert "3" in full_report.split("Unique items:")[1].split("\n")[0]

def test_format_analysis_summary_item_error_handling():
    class FailStr:
        def __str__(self):
            raise TypeError("str failed")

    bad_items = [FailStr()]
    report = multitool._format_analysis_summary(1, bad_items, "item", None, False)
    full_report = "".join(report)
    assert "Shortest item" not in full_report
    assert "Longest item" not in full_report

def test_format_analysis_summary_distance_calculation_error_handling():
    with patch("multitool.levenshtein_distance") as mock_lev:
        mock_lev.side_effect = Exception("Levenshtein failed")

        report = multitool._format_analysis_summary(2, [("a", "b")], "item", None, False)
        assert "Min/Max/Avg changes" not in "".join(report)

def test_print_processing_stats_respects_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    with patch("multitool._format_analysis_summary") as mock_format:
        mock_format.return_value = ["summary line"]
        multitool.print_processing_stats(10, ["item1"], "item", None)
        args, _ = mock_format.call_args
        assert args[4] is False

def test_stats_mode_respects_no_color(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    output_file = tmp_path / "stats.txt"
    input_file = tmp_path / "input.txt"
    input_file.write_text("word1 word2")

    with patch("multitool._format_analysis_summary") as mock_format:
        mock_format.return_value = ["summary line"]
        multitool.stats_mode(
            input_files=[str(input_file)],
            output_file=str(output_file),
            min_length=0,
            max_length=100,
            process_output=False,
            output_format='text'
        )
        args, _ = mock_format.call_args
        assert args[4] is False
