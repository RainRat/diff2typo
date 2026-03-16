import sys
import logging
import io
import runpy
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import typostats

def test_generate_report_transposition_summary(capsys):
    counts = {('he', 'eh'): 1}
    # allow_transposition=True will trigger the transposition summary calculation
    typostats.generate_report(counts, allow_transposition=True, quiet=False)
    captured = capsys.readouterr()
    assert "Transpositions [T]:" in captured.err
    assert "1/1" in captured.err

def test_generate_report_criteria_summary_stderr(capsys):
    counts = {('a', 'b'): 1, ('c', 'd'): 1}
    # min_occurrences=2 will filter out everything, unique_filtered (0) != unique_total (2)
    typostats.generate_report(counts, min_occurrences=2, quiet=False)
    captured = capsys.readouterr()
    assert "Patterns matching criteria:" in captured.err

def test_generate_report_output_file_summaries(tmp_path):
    output_file = tmp_path / "report.txt"
    counts = {('he', 'eh'): 2, ('a', 'b'): 1}

    # Trigger criteria_summary (line 509) and transposition_summary (line 515)
    typostats.generate_report(
        counts,
        output_file=str(output_file),
        min_occurrences=2,
        allow_transposition=True
    )
    content = output_file.read_text()
    assert "Patterns matching criteria:" in content # line 509
    assert "Transpositions [T]:" in content # line 515

    # Trigger display_summary (line 518)
    typostats.generate_report(
        counts,
        output_file=str(output_file),
        limit=1
    )
    content = output_file.read_text()
    assert "Showing patterns:" in content # line 518

def test_generate_report_non_adjacent_keyboard(capsys):
    counts = {('q', 'p'): 1} # q and p are not adjacent
    typostats.generate_report(counts, keyboard=True, quiet=False)
    captured = capsys.readouterr()
    # Check that [K] is NOT in the row
    assert "[K]" not in captured.out
    # Coverage for line 559 (marker = "   ")

def test_detect_encoding_no_chardet(caplog):
    with patch('typostats._CHARDET_AVAILABLE', False):
        with caplog.at_level(logging.WARNING):
            assert typostats.detect_encoding("dummy.txt") is None
            assert "chardet not installed" in caplog.text

def test_load_lines_from_file_detection_success():
    mock_files = {
        'dummy.txt': b'\xff'
    }
    def mocked_open(file, mode='r', encoding=None, **kwargs):
        if 'b' in mode:
            return io.BytesIO(mock_files[file])
        if encoding == 'utf-8':
            raise UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')
        if encoding == 'latin1_detected':
            return io.StringIO("detected content")
        return io.StringIO("other content")

    with patch('builtins.open', side_effect=mocked_open), \
         patch('typostats.detect_encoding', return_value='latin1_detected'):
        lines = typostats.load_lines_from_file('dummy.txt')
        assert lines == ["detected content"]

def test_main_all_flag():
    with patch('sys.argv', ['typostats.py', '--all']), \
         patch('typostats.load_lines_from_file', return_value=[]), \
         patch('typostats.generate_report') as mock_report:
        typostats.main()
        _, kwargs = mock_report.call_args
        assert kwargs['allow_1to2'] is True
        assert kwargs['allow_2to1'] is True
        assert kwargs['include_deletions'] is True
        assert kwargs['allow_transposition'] is True
        assert kwargs['keyboard'] is True

def test_main_execution():
    with patch('sys.argv', ['typostats.py', '--help']):
        with pytest.raises(SystemExit):
            runpy.run_path('typostats.py', run_name='__main__')
