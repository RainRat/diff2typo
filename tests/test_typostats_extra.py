import sys
from pathlib import Path
import pytest
import logging
import json
import io
from unittest.mock import MagicMock, patch, mock_open

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import typostats

def test_get_adjacent_keys():
    # include_diagonals=True
    adj_diag = typostats.get_adjacent_keys(include_diagonals=True)
    # 'w' is surrounded by 'q', 'e', 'a', 's'
    # if diagonals are included: 'q', 'e', 'a', 's' + maybe some others depending on keyboard layout
    # Row 0: qwertyuiop
    # Row 1: asdfghjkl
    # 'w' is (0, 1).
    # Neighbours: (0, 0)->'q', (0, 2)->'e', (1, 0)->'a', (1, 1)->'s', (1, 2)->'d'
    assert 'q' in adj_diag['w']
    assert 'e' in adj_diag['w']
    assert 'a' in adj_diag['w']
    assert 's' in adj_diag['w']
    assert 'd' in adj_diag['w']

    # include_diagonals=False
    adj_no_diag = typostats.get_adjacent_keys(include_diagonals=False)
    # Neighbours: (0, 0)->'q', (0, 2)->'e', (1, 1)->'s'
    # (1, 0) and (1, 2) are diagonals
    assert 'q' in adj_no_diag['w']
    assert 'e' in adj_no_diag['w']
    assert 's' in adj_no_diag['w']
    assert 'a' not in adj_no_diag['w']
    assert 'd' not in adj_no_diag['w']

def test_is_one_letter_replacement_2to1_advanced():
    # ph -> f
    assert typostats.is_one_letter_replacement('f', 'ph', allow_2to1=True) == [('ph', 'f')]

    # include_deletions=False (default)
    # or -> o is a deletion because 'o' is in 'or'
    assert typostats.is_one_letter_replacement('o', 'or', allow_2to1=True) == []

    # include_deletions=True
    assert typostats.is_one_letter_replacement('o', 'or', allow_2to1=True, include_deletions=True) == [('or', 'o')]

def test_print_processing_stats_retention(caplog):
    caplog.set_level(logging.INFO)
    # 2 raw, 1 filtered -> 50.0% retention
    typostats.print_processing_stats(2, 1, item_label="replacement")
    assert "Total replacements encountered:     2" in caplog.text
    assert "Total replacements after filtering: 1" in caplog.text
    assert "Retention rate:                     50.0%" in caplog.text

    # 0 raw, 0 filtered -> should not divide by zero
    caplog.clear()
    typostats.print_processing_stats(0, 0, item_label="replacement")
    assert "No replacements passed the filtering criteria." in caplog.text

def test_generate_report_keyboard_arrow(capsys):
    counts = {('q', 'w'): 5} # 'q' and 'w' are adjacent
    typostats.generate_report(counts, keyboard=True, output_format='arrow', quiet=False)
    captured = capsys.readouterr()
    # Check stderr for the summary
    assert "Keyboard Adjacency" in captured.err
    assert "5/5" in captured.err
    assert "100.0%" in captured.err
    # Check stdout for the marker
    # The [K] might be colorized
    assert "[K]" in captured.out

def test_generate_report_keyboard_json():
    counts = {('q', 'w'): 5}
    with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
        typostats.generate_report(counts, keyboard=True, output_format='json')
        data = json.loads(fake_stdout.getvalue())
        assert data["replacements"][0]["is_adjacent"] is True

def test_generate_report_write_error(caplog):
    caplog.set_level(logging.ERROR)
    counts = {('a', 'b'): 1}
    with patch("builtins.open", side_effect=IOError("Permission denied")):
        typostats.generate_report(counts, output_file="unwritable.txt")
        assert "Failed to write report to 'unwritable.txt'. Error: Permission denied" in caplog.text

def test_detect_encoding_logic():
    with patch('typostats._CHARDET_AVAILABLE', True), \
         patch('typostats.chardet') as mock_chardet, \
         patch('builtins.open', mock_open(read_data=b"some data")):

        # High confidence
        mock_chardet.detect.return_value = {'encoding': 'utf-8', 'confidence': 0.9}
        assert typostats.detect_encoding("dummy.txt") == 'utf-8'

        # Low confidence
        mock_chardet.detect.return_value = {'encoding': 'utf-8', 'confidence': 0.4}
        assert typostats.detect_encoding("dummy.txt") is None

def test_load_lines_from_file_variants(monkeypatch):
    # Test stdin
    monkeypatch.setattr(sys.stdin, 'readlines', lambda: ["line1\n"])
    assert typostats.load_lines_from_file('-') == ["line1\n"]

    # Test fallback sequence: UTF-8 -> Detection -> Latin-1
    # 1. UTF-8 fails
    # 2. Detection returns 'ascii'
    # 3. Reading with 'ascii' fails
    # 4. Fallback to 'latin1' succeeds

    mock_files = {
        'dummy.txt': b'\xff' # Not valid UTF-8
    }

    def mocked_open(file, mode='r', encoding=None, **kwargs):
        if 'b' in mode:
            return io.BytesIO(mock_files[file])

        content = mock_files[file]
        if encoding == 'utf-8':
            raise UnicodeDecodeError('utf-8', content, 0, 1, 'invalid')
        elif encoding == 'ascii':
            raise UnicodeDecodeError('ascii', content, 0, 1, 'invalid')
        elif encoding == 'latin1':
            return io.StringIO(content.decode('latin1'))

        return io.StringIO(content.decode('utf-8'))

    with patch('builtins.open', side_effect=mocked_open), \
         patch('typostats.detect_encoding', return_value='ascii'):
        lines = typostats.load_lines_from_file('dummy.txt')
        assert lines == ['\xff'] # Latin1 decodes \xff as ÿ, but StringIO might represent it

def test_main_stdin_default():
    with patch('sys.argv', ['typostats.py']), \
         patch('typostats.load_lines_from_file', return_value=[]) as mock_load, \
         patch('typostats.generate_report'):
        typostats.main()
        mock_load.assert_called_with('-')

def test_main_allow_two_char_alias():
    # provide some lines so process_typos is called
    with patch('sys.argv', ['typostats.py', 'input.txt', '--allow-two-char']), \
         patch('typostats.load_lines_from_file', return_value=["tezt -> test"]), \
         patch('typostats.process_typos', return_value=({}, 0, 0)) as mock_process, \
         patch('typostats.generate_report'):
        typostats.main()
        # Verify that allow_1to2 and allow_2to1 are both True
        args, kwargs = mock_process.call_args
        assert kwargs['allow_1to2'] is True
        assert kwargs['allow_2to1'] is True

def test_minimal_formatter_warning():
    formatter = typostats.MinimalFormatter('%(levelname)s: %(message)s')

    # INFO level should not have prefix
    info_record = logging.LogRecord('name', logging.INFO, 'pathname', 10, 'info msg', None, None)
    assert formatter.format(info_record) == 'info msg'

    # WARNING level should have prefix
    warn_record = logging.LogRecord('name', logging.WARNING, 'pathname', 10, 'warn msg', None, None)
    assert formatter.format(warn_record) == 'WARNING: warn msg'

def test_generate_report_keyboard_with_limit(capsys):
    # Two adjacent pairs
    counts = {('q', 'w'): 5, ('a', 's'): 10}
    # Limit to 1, but keyboard summary should still show both (5+10 = 15)
    typostats.generate_report(counts, keyboard=True, limit=1, output_format='arrow', quiet=False)
    captured = capsys.readouterr()
    assert "Keyboard Adjacency" in captured.err
    assert "15/15" in captured.err
    assert "100.0%" in captured.err

def test_is_transposition_length_mismatch():
    # Covering line 73: length mismatch returns []
    assert typostats.is_transposition("abc", "ab") == []

def test_process_typos_empty_line():
    lines = ["", "tezt -> test"]
    counts, total, raw = typostats.process_typos(lines)
    # total_lines is incremented before checking if line is empty
    assert total == 2

def test_generate_report_with_file_and_keyboard():
    counts = {('q', 'w'): 5}
    # Mocking open to return a StringIO
    m = mock_open()
    with patch("builtins.open", m):
        typostats.generate_report(counts, output_file="test.txt", keyboard=True)

    # Check if write was called with expected content
    handle = m()
    written_content = "".join(call.args[0] for call in handle.write.call_args_list)
    assert "Keyboard Adjacency" in written_content

def test_generate_report_no_results_stderr(capsys):
    counts = {}
    typostats.generate_report(counts, quiet=False)
    captured = capsys.readouterr()
    assert "No replacements found matching the criteria." in captured.err

def test_minimal_formatter_color_with_tty():
    # Patch the colors in the class or global scope because they might be empty strings
    with patch("typostats.RED", "\033[31m"), patch("typostats.RESET", "\033[0m"):
        # We also need to patch LEVEL_COLORS because it was initialized with the old values
        new_colors = {logging.ERROR: "\033[31m"}
        with patch.object(typostats.MinimalFormatter, 'LEVEL_COLORS', new_colors):
            formatter = typostats.MinimalFormatter('%(levelname)s: %(message)s')
            record = logging.LogRecord('name', logging.ERROR, 'pathname', 10, 'error msg', None, None)
            with patch("typostats.sys.stderr.isatty", return_value=True):
                formatted = formatter.format(record)
                assert "\033[31m" in formatted
