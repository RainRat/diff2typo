import sys
import os
import time
import logging
import importlib
import io
import json
import csv
from pathlib import Path
from unittest.mock import patch
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import typostats

def test_levenshtein_distance_empty_second_string():
    assert typostats.levenshtein_distance("abc", "") == 3

def test_minimal_formatter_colors():
    # Force colors off via NO_COLOR
    with patch("sys.stdout.isatty", return_value=True), patch.dict(os.environ, {"NO_COLOR": "1"}):
        importlib.reload(typostats)
        assert typostats.BLUE == ""

    # Force colors off via isatty=False
    with patch("sys.stdout.isatty", return_value=False), patch.dict(os.environ, {}, clear=True):
        importlib.reload(typostats)
        assert typostats.BLUE == ""

    # Enable colors
    with patch("sys.stdout.isatty", return_value=True), patch.dict(os.environ, {}, clear=True):
        importlib.reload(typostats)
        assert typostats.BLUE != ""

    # Reset
    importlib.reload(typostats)

def test_minimal_formatter_no_level_color():
    formatter = typostats.MinimalFormatter()
    # DEBUG is not in LEVEL_COLORS
    record = logging.LogRecord('name', logging.DEBUG, 'pathname', 10, 'debug msg', None, None)
    with patch("sys.stderr.isatty", return_value=True):
        formatted = formatter.format(record)
        assert "DEBUG: debug msg" in formatted

def test_format_analysis_summary_unhashable_items():
    # filtered_items=[[1]] is unhashable
    lines = typostats._format_analysis_summary(1, [[1]], use_color=False)
    # It should still report unique count as 1 (falling back to len(filtered_items))
    assert any("Unique items:" in line and " 1" in line for line in lines)

def test_format_analysis_summary_fractional_retention_bar():
    # 505/1000 = 50.5%
    # 0.505 * 20 = 10.1 blocks
    # 10 full blocks, 0.1 fraction
    lines = typostats._format_analysis_summary(1000, ["a"]*505, use_color=False)
    assert any("50.5%" in line for line in lines)
    # Check for some block character
    assert any("█" in line and "50.5%" in line for line in lines)

def test_format_analysis_summary_item_formatting_fallback():
    # 123 is not a tuple of length 2, so format_item should return str(123)
    lines = typostats._format_analysis_summary(1, [123], use_color=False)
    assert any("'123'" in line for line in lines)

def test_format_analysis_summary_str_exception():
    class BadStr:
        def __str__(self):
            raise TypeError("bad str")
        def __len__(self):
            return 5

    # Trigger line 177: except (ValueError, TypeError)
    lines = typostats._format_analysis_summary(1, [BadStr()], use_color=False)
    assert any("ANALYSIS SUMMARY" in line for line in lines)

def test_format_analysis_summary_distance_exception():
    class VeryBadStr:
        def __init__(self):
            self.count = 0
        def __str__(self):
            self.count += 1
            if self.count == 1: # First call in lengths calculation
                raise ValueError("first fail")
            raise RuntimeError("second fail") # Second call in distance calculation
        def __len__(self):
            return 4

    # Paired data distances block requires a tuple of length 2
    lines = typostats._format_analysis_summary(1, [(VeryBadStr(), "b")], use_color=False)
    assert any("ANALYSIS SUMMARY" in line for line in lines)

def test_format_analysis_summary_extra_metrics_and_time():
    start = time.perf_counter() - 1.0 # 1 second ago
    lines = typostats._format_analysis_summary(
        1, ["a"],
        extra_metrics={'ExtraLabel': 'ExtraValue'},
        start_time=start,
        use_color=False
    )
    assert any("ExtraLabel:" in line and "ExtraValue" in line for line in lines)
    assert any("Processing time:" in line for line in lines)

def test_format_analysis_summary_truthy_empty_gaps():
    class TruthyEmpty:
        def __bool__(self): return True
        def __iter__(self): return iter([])
        def __len__(self): return 0
        def __getitem__(self, i): raise IndexError()

    # This covers if lengths: False (157->165) and if distances: False (188->199)
    # For distances, it also needs to satisfy the tuple check.
    # We can use a TruthyEmpty that also acts like a sequence of tuples.

    class TruthyEmptyTuples(TruthyEmpty):
        def __getitem__(self, i):
            if i == 0: return ("a", "b")
            raise IndexError()
        def __len__(self): return 1 # Wait, if len is 1 it's not empty.
        # Let's try just len=1 but iter is empty.
        def __iter__(self): return iter([])

    lines = typostats._format_analysis_summary(1, TruthyEmptyTuples(), use_color=False)
    assert any("ANALYSIS SUMMARY" in line for line in lines)

def test_is_transposition_success():
    assert typostats.is_transposition("teh", "the") == [('he', 'eh')]

def test_is_transposition_not_transposition():
    # Two differences but not a swap
    assert typostats.is_transposition("ab", "cd") == []

def test_generate_report_no_results_quiet_true(capsys):
    typostats.generate_report({}, quiet=True)
    captured = capsys.readouterr()
    assert captured.err == ""

def test_generate_report_no_results_quiet_false(capsys):
    typostats.generate_report({}, quiet=False)
    captured = capsys.readouterr()
    assert "No replacements found" in captured.err

def test_generate_report_marker_m():
    # ('a', 'b'): len 1, len 1 -> [K] logic
    # ('m', 'rn'): len 1, len 2 -> [M] logic
    # ('abc', 'def'): len 3, len 3 -> neither [K], [T] nor [M] (covers 685 -> 687 branch)
    counts = {('a', 'b'): 1, ('m', 'rn'): 1, ('abc', 'def'): 1}
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        typostats.generate_report(counts, allow_1to2=True, keyboard=True, output_format='arrow')
        output = fake_out.getvalue()
        assert "[M]" in output
        assert "abc" in output

def test_generate_report_json_keyboard_logic():
    # Both adjacent, non-adjacent, and multi-char for coverage
    # ('he', 'eh'): len 2, len 2 -> [T] logic
    counts = {('q', 'w'): 1, ('q', 'p'): 1, ('m', 'rn'): 1, ('he', 'eh'): 1}
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        typostats.generate_report(counts, keyboard=True, allow_transposition=True, output_format='json')
        data = json.loads(fake_out.getvalue())
        # 'q' and 'w' are adjacent
        assert any(item["correct"] == 'q' and item["typo"] == 'w' and item["is_adjacent"] is True for item in data["replacements"])
        # 'q' and 'p' are not
        assert any(item["correct"] == 'q' and item["typo"] == 'p' and item["is_adjacent"] is False for item in data["replacements"])
        # 'm' and 'rn' are multi-char
        assert any(item["correct"] == 'm' and item["typo"] == 'rn' and item["is_adjacent"] is False for item in data["replacements"])

def test_generate_report_yaml_formatting():
    counts = {('a', 'b'): 1, ('a', 'c'): 1}
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        typostats.generate_report(counts, output_format='yaml')
        output = fake_out.getvalue()
        assert "  a:" in output
        assert '  - "b"' in output
        assert '  - "c"' in output

def test_generate_report_file_write_csv_no_extra_newline(tmp_path):
    # CSV already ends with a newline, so it should skip 739 (covering 738->740)
    report_file_csv = tmp_path / "report.csv"
    counts = {('a', 'b'): 1}
    typostats.generate_report(counts, output_file=str(report_file_csv), output_format='csv')
    content_csv = report_file_csv.read_text()
    assert content_csv.endswith('\n')
    # If it added another one, it would end with \n\n.
    assert not content_csv.endswith('\n\n')

def test_generate_report_file_write_success_logging(tmp_path, caplog):
    report_file = tmp_path / "report.txt"
    counts = {('a', 'b'): 1}
    with caplog.at_level(logging.INFO):
        typostats.generate_report(counts, output_file=str(report_file))
        assert f"Report successfully written to '{report_file}'." in caplog.text

def test_generate_report_file_write_failure_logging(caplog):
    counts = {('a', 'b'): 1}
    # Passing a directory path as output_file will cause OSError on open
    with caplog.at_level(logging.ERROR):
        typostats.generate_report(counts, output_file=".")
        assert "Failed to write report to '.'." in caplog.text

def test_generate_report_keyboard_multi_char():
    # Covering the False branch of 'if len(c) == 1 and len(t) == 1' at line 543
    # Actually, line 543 is probably inside keyboard loop
    counts = {('a', 'b'): 1, ('m', 'rn'): 1}
    typostats.generate_report(counts, keyboard=True, output_format='arrow')
