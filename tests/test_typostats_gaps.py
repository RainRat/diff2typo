import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import typostats

def test_is_transposition_different_lengths():
    """Cover line 33 in typostats.py: return [] if lengths differ."""
    assert typostats.is_transposition("a", "ab") == []

def test_process_typos_empty_line():
    """Cover line 160 in typostats.py: skip empty lines."""
    # Use words with only one character difference so they are detected
    lines = ["apple -> ample", "", "banana -> banaxa"]
    counts = typostats.process_typos(lines, allow_two_char=False)

    # apple -> ample: typo=apple, correction=ample => ('m', 'p')
    assert counts[('m', 'p')] == 1
    # banana -> banaxa: typo=banana, correction=banaxa => ('x', 'n')
    assert counts[('x', 'n')] == 1
    assert len(counts) == 2

def test_generate_report_write_failure(caplog):
    """Cover lines 371-372 in typostats.py: handle exception during report write."""
    counts = {('a', 'b'): 1}
    output_file = "failing_output.txt"

    # Mock open to raise an exception when writing
    with patch("builtins.open", side_effect=IOError("Permission denied")):
        with caplog.at_level(logging.ERROR):
            typostats.generate_report(counts, output_file=output_file)

    assert "Failed to write report" in caplog.text
    assert "Permission denied" in caplog.text

def test_main_no_input_files_stdin_fallback(monkeypatch, caplog):
    """Cover line 510 in typostats.py: default to stdin if no input files."""
    # Mock stdin.readlines to return empty list so it finishes quickly
    mock_stdin = MagicMock()
    mock_stdin.readlines.return_value = []
    monkeypatch.setattr(sys, "stdin", mock_stdin)

    # Mock sys.argv to have no input files
    monkeypatch.setattr(sys, "argv", ["typostats.py"])

    with caplog.at_level(logging.INFO):
        typostats.main()

    assert "Reading from stdin" in caplog.text

def test_generate_report_default_sort(capsys):
    """Cover the 'else' block in generate_report's sorting logic."""
    counts = {('b', 'y'): 1, ('a', 'z'): 5}
    # Default sort is by count descending
    typostats.generate_report(counts, sort_by='count', output_format='arrow', quiet=True)
    captured = capsys.readouterr().out
    lines = [l for l in captured.splitlines() if '->' in l]
    assert 'a' in lines[0] # a -> z has count 5
    assert 'b' in lines[1] # b -> y has count 1

def test_load_lines_from_file_none_if_missing(caplog):
    """Verify load_lines_from_file returns None for missing file (already tested but good to have)."""
    with caplog.at_level(logging.ERROR):
        result = typostats.load_lines_from_file("definitely_not_a_file_12345.txt")
    assert result is None
    assert "File not found" in caplog.text
