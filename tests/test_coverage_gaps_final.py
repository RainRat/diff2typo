import sys
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import multitool
import typostats

def test_multitool_main_filenotfounderror(caplog, monkeypatch):
    """Test that multitool.main catches FileNotFoundError and logs it."""
    e = FileNotFoundError("missing_file.txt")
    e.filename = "missing_file.txt"
    mock_handler = MagicMock(side_effect=e)

    monkeypatch.setattr(sys, 'argv', ['multitool.py', 'count', 'dummy.txt'])

    with patch('multitool.count_mode', mock_handler):
        with caplog.at_level(logging.ERROR):
            # main() calls sys.exit(1) on FileNotFoundError
            with pytest.raises(SystemExit) as excinfo:
                multitool.main()
            assert excinfo.value.code == 1

    assert "File not found: 'missing_file.txt'" in caplog.text

def test_multitool_main_filenotfounderror_no_filename(caplog, monkeypatch):
    """Test that multitool.main catches FileNotFoundError without filename and logs it."""
    # This covers the 'else' branch in the FileNotFoundError handler
    mock_handler = MagicMock(side_effect=FileNotFoundError("Generic error"))

    monkeypatch.setattr(sys, 'argv', ['multitool.py', 'count', 'dummy.txt'])

    with patch('multitool.count_mode', mock_handler):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit) as excinfo:
                multitool.main()
            assert excinfo.value.code == 1

    assert "File not found: Generic error" in caplog.text

def test_multitool_help_subcommand(monkeypatch):
    """Test 'multitool.py help count' subcommand."""
    monkeypatch.setattr(sys, 'argv', ['multitool.py', 'help', 'count'])

    # show_mode_help calls parser.exit() which raises SystemExit
    with pytest.raises(SystemExit):
        multitool.main()

def test_multitool_help_subcommand_no_exit(monkeypatch):
    """Test 'multitool.py help count' subcommand without exit to cover return statement."""
    monkeypatch.setattr(sys, 'argv', ['multitool.py', 'help', 'count'])

    with patch('multitool.show_mode_help') as mock_help:
        # Should reach 'return' after show_mode_help
        multitool.main()
    mock_help.assert_called_once()

def test_levenshtein_distance_empty_s2():
    """Test levenshtein_distance with an empty second string."""
    assert typostats.levenshtein_distance("abc", "") == 3

def test_generate_report_unhashable_items():
    """Test _format_analysis_summary with unhashable items to trigger except block."""
    # The gap is in _format_analysis_summary at line 140
    # Let's mock set() to raise TypeError to trigger line 140
    with patch('builtins.set', side_effect=TypeError("unhashable")):
        lines = typostats._format_analysis_summary(10, ["item1"], use_color=False)
    assert any("Unique items:" in line for line in lines)

def test_generate_report_format_item_failure():
    """Test _format_analysis_summary format_item failure to trigger except block."""
    class BadItem:
        def __str__(self):
            raise TypeError("Cannot stringify")

    # Trigger except (ValueError, TypeError) on line 177
    lines = typostats._format_analysis_summary(1, [BadItem()], use_color=False)
    assert any("ANALYSIS SUMMARY" in line for line in lines)

def test_generate_report_distance_failure():
    """Test _format_analysis_summary distance calculation failure."""
    # line 187: distances = [levenshtein_distance(str(p[0]), str(p[1])) for p in filtered_items]

    with patch('typostats.levenshtein_distance', side_effect=Exception("Levenshtein failed")):
        # Must be a tuple of length 2 to enter the distance block
        lines = typostats._format_analysis_summary(1, [('a', 'b')], use_color=False)
    assert any("ANALYSIS SUMMARY" in line for line in lines)

def test_generate_report_all_summary_branches(capsys):
    """Trigger remaining branches in generate_report."""
    counts = {('a', 'b'): 1}
    typostats.generate_report(counts, quiet=True)
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "a │ b" in captured.out

def test_format_analysis_summary_retention_bar_edge():
    """Trigger the bar += blocks[frac_idx] branch in _format_analysis_summary."""
    lines = typostats._format_analysis_summary(3, ["item1"], use_color=False)
    # Check that the retention line has some bar characters
    retention_line = [line for line in lines if "Retention rate:" in line][0]
    assert "33.3%" in retention_line
    assert "█" in retention_line or "▏" in retention_line
