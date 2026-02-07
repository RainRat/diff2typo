import io
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import diff2typo

def test_format_typos_no_arrow():
    """Test format_typos when ' -> ' is not present in the typo string."""
    typos = ['malformed_typo']
    # Should fall back to filter_to_letters
    assert diff2typo.format_typos(typos, 'arrow') == ['malformedtypo']

def test_decode_with_fallback_latin1():
    """Test _decode_with_fallback when utf-8 fails but latin-1 succeeds."""
    # 0xE9 is 'é' in latin-1, invalid in utf-8
    latin1_data = b'caf\xe9'
    result = diff2typo._decode_with_fallback(latin1_data, "test data")
    assert result == 'café'

def test_read_stdin_text_already_string():
    """Test _read_stdin_text when sys.stdin.read() returns a string."""
    mock_stdin = MagicMock()
    del mock_stdin.buffer
    mock_stdin.read.return_value = "string data"
    with patch('sys.stdin', mock_stdin):
        result = diff2typo._read_stdin_text()
        assert result == "string data"

def test_read_diff_sources_no_input():
    """Test _read_diff_sources when no input files are provided (defaults to stdin)."""
    with patch('diff2typo._read_stdin_text') as mock_read_stdin:
        mock_read_stdin.return_value = "stdin content"
        result = diff2typo._read_diff_sources(None)
        assert result == "stdin content"

        result = diff2typo._read_diff_sources([])
        assert result == "stdin content"

def test_filter_known_typos_write_error(caplog):
    """Test filter_known_typos handles file write errors gracefully."""
    candidates = ['eror -> error']

    # Mock 'open' to raise an exception when opening 'candidates.txt'
    original_open = open
    def mock_open(file, *args, **kwargs):
        if 'candidates.txt' in str(file):
            raise Exception("Write failed")
        return original_open(file, *args, **kwargs)

    with patch('builtins.open', side_effect=mock_open):
        with caplog.at_level(logging.ERROR):
            result = diff2typo.filter_known_typos(candidates, 'typos')

    assert result == candidates
    assert "Error writing to temporary file" in caplog.text

def test_process_new_corrections_empty_mapping():
    """Test process_new_corrections with an empty words mapping."""
    candidates = ['eror -> error']
    result = diff2typo.process_new_corrections(candidates, {}, quiet=True)
    assert result == []

def test_process_new_corrections_standalone_only():
    """Test process_new_corrections with mapping containing only standalone words."""
    # Standalone words have empty sets as values in the mapping
    words_mapping = {'valid': set()}
    candidates = ['eror -> error']
    result = diff2typo.process_new_corrections(candidates, words_mapping, quiet=True)
    assert result == []

def test_main_both_mode(tmp_path, monkeypatch):
    """Integration test for main in 'both' mode."""
    monkeypatch.chdir(tmp_path)

    diff_file = tmp_path / "test.diff"
    diff_file.write_text("--- a/f\n+++ b/f\n@@\n-a eror b\n+a error b\n-c teh d\n+c the d\n")

    # 'eror' will be a new typo, 'teh' will be a new correction for existing typo
    dictionary_file = tmp_path / "words.csv"
    dictionary_file.write_text("teh,original\nvalid\n")

    allowed_file = tmp_path / "allowed.csv"
    allowed_file.write_text("")

    output_file = tmp_path / "output.txt"

    # Mock subprocess.run for typos tool (nothing known)
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = SimpleNamespace(stdout="", returncode=0)

        monkeypatch.setattr(sys, 'argv', [
            'diff2typo.py',
            '--input_file', str(diff_file),
            '--output_file', str(output_file),
            '--dictionary_file', str(dictionary_file),
            '--allowed_file', str(allowed_file),
            '--mode', 'both',
            '--quiet'
        ])

        diff2typo.main()

    content = output_file.read_text()
    assert "=== New Typos ===" in content
    assert "eror -> error" in content
    assert "=== New Corrections ===" in content
    assert "teh -> the" in content

def test_main_corrections_mode(tmp_path, monkeypatch):
    """Integration test for main in 'corrections' mode."""
    monkeypatch.chdir(tmp_path)

    diff_file = tmp_path / "test.diff"
    diff_file.write_text("--- a/f\n+++ b/f\n@@\n-a teh b\n+a the b\n")

    dictionary_file = tmp_path / "words.csv"
    dictionary_file.write_text("teh,original\n")

    allowed_file = tmp_path / "allowed.csv"
    allowed_file.write_text("")

    output_file = tmp_path / "output.txt"

    monkeypatch.setattr(sys, 'argv', [
        'diff2typo.py',
        '--input_file', str(diff_file),
        '--output_file', str(output_file),
        '--dictionary_file', str(dictionary_file),
        '--allowed_file', str(allowed_file),
        '--mode', 'corrections',
        '--quiet'
    ])

    diff2typo.main()

    content = output_file.read_text().strip()
    assert content == "teh -> the"

def test_filter_known_typos_subprocess_error(monkeypatch, tmp_path, caplog):
    """Test filter_known_typos handles subprocess errors."""
    candidates = ['eror -> error']

    # Create a dummy typos tool
    typos_tool = tmp_path / "typos"
    typos_tool.touch()

    def mock_run(*args, **kwargs):
        raise FileNotFoundError("Mocked file not found")

    monkeypatch.setattr('subprocess.run', mock_run)

    with caplog.at_level(logging.WARNING):
        result = diff2typo.filter_known_typos(candidates, str(typos_tool))

    assert result == candidates
    assert "Error running typos tool" in caplog.text

def test_tqdm_coverage():
    """Run functions that use tqdm with quiet=False to gain coverage."""
    candidates = ['eror -> error']
    filter_set = {'other'}

    mock_tqdm_obj = MagicMock()
    mock_tqdm_obj.__iter__.return_value = iter(candidates)

    with patch('diff2typo.tqdm', return_value=mock_tqdm_obj):
        diff2typo._filter_candidates_by_set(candidates, filter_set, "Test", quiet=False)
        assert mock_tqdm_obj.close.called

        mock_tqdm_obj.close.reset_mock()
        diff2typo.process_new_corrections(candidates, {'eror': {'error'}}, quiet=False)
        assert mock_tqdm_obj.close.called

def test_read_diff_file_not_found():
    """Test _read_diff_file exits when file is missing."""
    with pytest.raises(SystemExit):
        diff2typo._read_diff_file("nonexistent.diff")

def test_main_allowed_file_error(tmp_path, monkeypatch):
    """Test main exits when allowed file fails to read."""
    monkeypatch.chdir(tmp_path)
    allowed_file = tmp_path / "allowed.csv"
    allowed_file.touch()

    # Create words.csv so it doesn't exit early
    dictionary_file = tmp_path / "words.csv"
    dictionary_file.write_text("valid\n")

    with patch('diff2typo._read_diff_sources', return_value=""):
        # Trigger an exception during read_allowed_words
        with patch('diff2typo.read_allowed_words', side_effect=Exception("Read error")):
            monkeypatch.setattr(sys, 'argv', [
                'diff2typo.py',
                '--allowed_file', str(allowed_file),
                '--dictionary_file', str(dictionary_file),
                '--quiet'
            ])
            with pytest.raises(SystemExit):
                diff2typo.main()

def test_main_output_file_error(tmp_path, monkeypatch):
    """Test main exits when output file fails to write."""
    monkeypatch.chdir(tmp_path)

    diff_file = tmp_path / "test.diff"
    diff_file.write_text("--- a/f\n+++ b/f\n@@\n-a eror b\n+a error b\n")

    dictionary_file = tmp_path / "words.csv"
    dictionary_file.write_text("valid\n")

    output_file = tmp_path / "output.txt"

    original_open = open
    def mock_open(file, mode='r', **kwargs):
        if 'output.txt' in str(file) and 'w' in mode:
            raise Exception("Write error")
        return original_open(file, mode, **kwargs)

    with patch('builtins.open', side_effect=mock_open):
        monkeypatch.setattr(sys, 'argv', [
            'diff2typo.py',
            '--input_file', str(diff_file),
            '--output_file', str(output_file),
            '--dictionary_file', str(dictionary_file),
            '--quiet'
        ])
        with pytest.raises(SystemExit):
            diff2typo.main()

def test_read_diff_sources_directory(tmp_path):
    """Test _read_diff_sources exits when a matched path is a directory."""
    dir_path = tmp_path / "somedir"
    dir_path.mkdir()

    with pytest.raises(SystemExit):
        diff2typo._read_diff_sources([str(dir_path)])
