import sys
import logging
import io
import os
import importlib
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import typostats

def test_levenshtein_distance_base_case():
    # Cover line 68: if not s2: return len(s1)
    assert typostats.levenshtein_distance("abc", "") == 3

def test_minimal_formatter_color_coverage():
    formatter = typostats.MinimalFormatter()
    # Force a color into LEVEL_COLORS for testing if needed,
    # but WARNING (30) is already there as YELLOW.

    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=1,
        msg="warning message",
        args=None,
        exc_info=None
    )
    # record.levelname is set during creation based on level

    # Mock stderr.isatty to True AND ensure global YELLOW/RESET are not empty
    # We also need to mock MinimalFormatter.LEVEL_COLORS to ensure it uses the color we expect
    with patch("sys.stderr.isatty", return_value=True), \
         patch.dict(typostats.MinimalFormatter.LEVEL_COLORS, {logging.WARNING: "\033[1;33m"}), \
         patch("typostats.RESET", "\033[0m"):
        formatted = formatter.format(record)
        # WARNING should be colorized
        assert "\x1b[1;33mWARNING\x1b[0m" in formatted
        assert "warning message" in formatted

    # Test with a level that has no color defined
    record.levelno = logging.DEBUG
    record.levelname = "DEBUG"
    with patch("sys.stderr.isatty", return_value=True):
        formatted = formatter.format(record)
        # DEBUG should not be colorized (not in LEVEL_COLORS by default)
        assert "\033" not in formatted
        assert "DEBUG: warning message" == formatted

def test_format_analysis_summary_bar_padding():
    # Cover line 130: bar += " " * (max_bar - full_blocks - 1)
    # raw=100, filtered=50 => retention=50%. max_bar=20, total_blocks=10, full=10, frac=0.
    # frac_idx = 0 => blocks[0] = " ".
    report = typostats._format_analysis_summary(
        raw_count=100,
        filtered_items=["a"] * 50,
        use_color=False
    )
    # Find the retention rate line
    retention_line = [line for line in report if "Retention rate:" in line][0]
    # "█" * 10 + " " + " " * (20 - 10 - 1) = █x10 + " " + " "x9
    assert "█" * 10 in retention_line
    # Total bar length should be 20 chars after the percentage
    # Don't strip() because we want to count spaces
    bar_part = retention_line.split("%")[-1]
    # It might have a leading space from the formatting, so we find where the bar starts
    bar_start = bar_part.find("█")
    bar_content = bar_part[bar_start:]
    assert len(bar_content) == 20

def test_format_analysis_summary_unhashable():
    # Cover line 140: except (TypeError, ValueError): unique_count = len(filtered_items)
    # Lists are unhashable
    items = [["a"], ["b"]]
    report = typostats._format_analysis_summary(
        raw_count=2,
        filtered_items=items,
        use_color=False
    )
    unique_line = [line for line in report if "Unique items:" in line][0]
    assert "2" in unique_line

def test_format_analysis_summary_item_formatting_and_exceptions():
    # Use non-tuple items to cover line 153: return str(it)

    # Trigger 177-178 (except (ValueError, TypeError): pass)
    class ReallyBadItem:
        def __str__(self):
            raise TypeError("unstringable")

    # Use a generator that yields nothing to cover 157->165 (False branch for if lengths)
    # and 188->199 (False branch for if distances)
    # typostats needs subscripting for the distance block though: filtered_items[0]
    # So we'll use a custom class that is truthy but empty-ish?
    # Or just a list and we accept we can't hit 'if lengths' False branch easily if we are in 'if filtered_items'.
    # Actually, we can just use a list for filtered_items but mock len() to return 0? No.

    items = ["simple_string", ReallyBadItem()]
    report = typostats._format_analysis_summary(
        raw_count=2,
        filtered_items=items,
        use_color=False
    )
    # Check that it didn't crash
    assert any("ANALYSIS SUMMARY" in line for line in report)

    # Trigger paired data distance exception (line 195)
    # We can mock levenshtein_distance to raise an exception.
    # Note: typostats.levenshtein_distance is imported in _format_analysis_summary
    with patch("typostats.levenshtein_distance", side_effect=Exception("mocked failure")):
        report = typostats._format_analysis_summary(
            raw_count=1,
            filtered_items=[("a", "b")],
            use_color=False
        )
        # Should NOT have "Min/Max/Avg changes" because of exception
        assert not any("Min/Max/Avg changes:" in line for line in report)

    # To hit 157->165 and 188->199 we need filtered_items to be truthy but lengths/distances to be empty.
    # For distances, it checks 'if distances:' after [levenshtein_distance(...) for p in filtered_items]
    # If filtered_items = [("a", "b")] but levenshtein_distance returns None and we filter it out? No, it's not filtered.
    # If we mock levenshtein_distance to return nothing? No.

    # Let's try mocking lengths to be empty even if filtered_items is not
    with patch("typostats.min", side_effect=ValueError("mocked empty")):
        report = typostats._format_analysis_summary(
            raw_count=1,
            filtered_items=["a"],
            use_color=False
        )
        # Should trigger 'except (ValueError, TypeError): pass' at 177-178

    # To hit 157->165 and 188->199 we need filtered_items to be truthy but lengths/distances to be empty.
    # Alternative: use a filtered_items that is a generator but truthfully returns nothing?
    # lengths = [len(format_item(it)) for it in filtered_items]
    # If filtered_items is a custom object that returns True for bool() but is empty for iteration
    class FakeItems:
        def __bool__(self): return True
        def __iter__(self): return iter([])
        def __getitem__(self, i): return ("a", "b") # for isinstance(filtered_items[0], ...)
        def __len__(self): return 1 # so it's not empty

    report = typostats._format_analysis_summary(
        raw_count=1,
        filtered_items=FakeItems(),
        use_color=False
    )
    # This should hit if lengths (empty) and if distances (empty)

def test_is_transposition_success():
    # Cover line 239: return [(correction[i:j+1], typo[i:j+1])]
    assert typostats.is_transposition("teh", "the") == [("he", "eh")]

def test_is_transposition_not_transp():
    # Cover line 238->242 (False branch)
    # differences are adjacent but not a swap
    assert typostats.is_transposition("abc", "ade") == []

def test_generate_report_quiet_no_results(capsys):
    # Cover lines 631 and 651: if not quiet: sys.stderr.write(...)
    # and no results path
    typostats.generate_report(
        {},
        quiet=True
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""

    # Contrast with not quiet
    typostats.generate_report(
        {},
        quiet=False
    )
    captured = capsys.readouterr()
    assert "ANALYSIS SUMMARY" in captured.err
    assert "No replacements found matching the criteria." in captured.err

def test_generate_report_multi_letter_marker(capsys):
    # Cover line 568: extra_metrics["Multiple letters [M]"]
    # and line 686: marker = f"{c_bold}[M]{c_reset} "
    # Also cover 543->542 (False branch for single char check)
    # Also cover 685->687 (False branch for marker logic)
    counts = {
        ("abc", "a"): 1,  # [M] marker
        ("ab", "cd"): 1   # No marker, but show_attr is True
    }
    # We must enable one of the flags that sets show_attr = True
    typostats.generate_report(
        counts,
        include_deletions=True,
        allow_1to2=True,
        keyboard=True,
        quiet=False
    )
    captured = capsys.readouterr()
    assert "Multiple letters [M]:" in captured.err
    assert "[M]" in captured.out
    # Row for ("ab", "cd") should have no marker in ATTR column
    # ATTR column is before VISUAL column
    assert "ab │ cd   │     1 │  50.0% │      │" in captured.out or "ab │ cd   │     1 │  50.0% │      " in captured.out

def test_generate_report_json_adjacency(capsys):
    # Cover lines 705-709: JSON format with keyboard adjacency
    # Also cover 706->709 (False branch for single char check)
    counts = {
        ("a", "s"): 1,
        ("a", "p"): 1,
        ("ab", "cd"): 1
    }
    typostats.generate_report(
        counts,
        output_format="json",
        keyboard=True
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    repls = data["replacements"]
    # find "a" -> "s"
    as_repl = [r for r in repls if r["correct"] == "a" and r["typo"] == "s"][0]
    assert as_repl["is_adjacent"] is True
    # find "a" -> "p"
    ap_repl = [r for r in repls if r["correct"] == "a" and r["typo"] == "p"][0]
    assert ap_repl["is_adjacent"] is False

def test_generate_report_file_newline(tmp_path):
    # Cover lines 738-739: append newline if missing
    report_file = tmp_path / "report.txt"
    # Mock report content to NOT end with newline
    with patch("typostats.json.dumps", return_value='{"no_newline": true}'):
        typostats.generate_report(
            {},
            output_file=str(report_file),
            output_format="json"
        )
    content = report_file.read_text()
    assert content.endswith("\n")
    assert content.count("\n") == 1

    # Cover 738->740 (False branch: already has newline)
    with patch("typostats.json.dumps", return_value='{"has_newline": true}\n'):
        typostats.generate_report(
            {},
            output_file=str(report_file),
            output_format="json"
        )
    content = report_file.read_text()
    assert content.endswith("\n")
    assert content.count("\n") == 1

def test_color_initialization_coverage():
    # Cover line 36: if not sys.stdout.isatty() or os.environ.get('NO_COLOR'):
    # We need to test both True and False branches.

    # branch 36 taken (True): no color
    with patch("sys.stdout.isatty", return_value=False), \
         patch.dict(os.environ, {}, clear=True):
        importlib.reload(typostats)
        assert typostats.BLUE == ""

    # branch 36 not taken (False): color enabled
    # We must mock os.environ to NOT have NO_COLOR and isatty to be True
    with patch("sys.stdout.isatty", return_value=True), \
         patch.dict(os.environ, {}, clear=True):
        importlib.reload(typostats)
        assert typostats.BLUE != ""

    # Restore module state for other tests
    importlib.reload(typostats)
