import os
import sys
import io
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import typostats

def test_detect_format_extension():
    allowed = ['arrow', 'json', 'csv', 'yaml']

    # Common extensions
    assert typostats._detect_format_from_extension('report.json', allowed, 'arrow') == 'json'
    assert typostats._detect_format_from_extension('report.csv', allowed, 'arrow') == 'csv'
    assert typostats._detect_format_from_extension('report.yaml', allowed, 'arrow') == 'yaml'
    assert typostats._detect_format_from_extension('report.yml', allowed, 'yaml') == 'yaml'
    assert typostats._detect_format_from_extension('report.txt', allowed, 'json') == 'arrow'
    assert typostats._detect_format_from_extension('report.arrow', allowed, 'json') == 'arrow'

    # Edge cases
    assert typostats._detect_format_from_extension('', allowed, 'arrow') == 'arrow'
    assert typostats._detect_format_from_extension(None, allowed, 'arrow') == 'arrow'
    assert typostats._detect_format_from_extension('-', allowed, 'arrow') == 'arrow'
    assert typostats._detect_format_from_extension('no_extension', allowed, 'arrow') == 'arrow'
    assert typostats._detect_format_from_extension('unknown.ext', allowed, 'arrow') == 'arrow'

    # Case insensitive
    assert typostats._detect_format_from_extension('REPORT.JSON', allowed, 'arrow') == 'json'

    # Not in allowed
    assert typostats._detect_format_from_extension('report.json', ['csv'], 'arrow') == 'arrow'

def test_color_env_vars():
    # NO_COLOR set
    with patch.dict(os.environ, {"NO_COLOR": "1"}):
        assert typostats._should_enable_color(MagicMock()) is False

    # FORCE_COLOR set (takes precedence if NO_COLOR is not set)
    with patch.dict(os.environ, {"FORCE_COLOR": "1"}):
        if "NO_COLOR" in os.environ:
            del os.environ["NO_COLOR"]
        assert typostats._should_enable_color(MagicMock()) is True

    # Neither set, check isatty
    with patch.dict(os.environ, {}, clear=True):
        mock_stream = MagicMock()
        mock_stream.isatty.return_value = True
        assert typostats._should_enable_color(mock_stream) is True

        mock_stream.isatty.return_value = False
        assert typostats._should_enable_color(mock_stream) is False

        # Missing isatty
        del mock_stream.isatty
        assert typostats._should_enable_color(mock_stream) is False

def test_robust_read_stdin_bytes():
    # Reset STDIN cache
    typostats._STDIN_CACHE = None

    mock_stdin = MagicMock()
    mock_buffer = MagicMock()
    mock_stdin.buffer = mock_buffer
    # Simulate binary data from stdin
    mock_buffer.read.return_value = b"bytes content\n"

    with patch('typostats.sys.stdin', mock_stdin):
        lines = typostats._read_file_lines_robust('-')
        assert lines == ["bytes content\n"]

    # Reset STDIN cache again
    typostats._STDIN_CACHE = None
    # Simulate binary data that fails UTF-8 but passes Latin-1
    mock_buffer.read.return_value = b"latin \xff content\n"
    with patch('typostats.sys.stdin', mock_stdin):
        lines = typostats._read_file_lines_robust('-')
        assert lines == ["latin \xff content\n"]

def test_robust_read_encoding_fallback_latin1(tmp_path):
    # Test the branch where UTF-8 fails, detection fails or returns None, and it falls back to latin-1
    f = tmp_path / "latin1.txt"
    # Content that is valid Latin-1 but not valid UTF-8
    f.write_bytes(b"latin1 \xff content")

    with patch('typostats.detect_encoding', return_value=None):
        lines = typostats._read_file_lines_robust(str(f))
        assert lines == ["latin1 \xff content"]

def test_robust_read_detected_encoding_success(tmp_path):
    # Test the branch where detected encoding succeeds (covers lines 436-437)
    f = tmp_path / "detected_success.txt"
    # UTF-16 content
    content = "some content"
    f.write_text(content, encoding='utf-16')

    with patch('typostats.detect_encoding', return_value='utf-16'):
        lines = typostats._read_file_lines_robust(str(f))
        assert lines == [content]

def test_robust_read_detected_encoding_failure(tmp_path):
    # Test the branch where detected encoding fails and it falls back to latin-1
    f = tmp_path / "fail_detected.txt"
    f.write_bytes(b"some content \xff")

    # We need to mock open to fail for the 'detected' encoding but work for others
    original_open = open
    def mocked_open(file, mode='r', encoding=None, **kwargs):
        if encoding == 'detected_enc':
            raise UnicodeDecodeError('detected_enc', b'', 0, 1, 'fake error')
        return original_open(file, mode=mode, encoding=encoding, **kwargs)

    with patch('typostats.detect_encoding', return_value='detected_enc'), \
         patch('builtins.open', side_effect=mocked_open):
        lines = typostats._read_file_lines_robust(str(f))
        assert lines == ["some content \xff"]

def test_is_one_letter_replacement_branches():
    # Test 1-to-2 where it's NOT an insertion and allow_1to2 is False (covers line 594)
    # To hit this, we need include_deletions=True so we enter the block, but allow_1to2=False
    # correction: 'm', typo: 'rn' -> 1-to-2 replacement, NOT an insertion
    assert typostats.is_one_letter_replacement('rn', 'm', allow_1to2=False, include_deletions=True) == []
    assert typostats.is_one_letter_replacement('rn', 'm', allow_1to2=True, include_deletions=False) == [('m', 'rn')]

    # Test 2-to-1 where it's NOT a deletion and allow_2to1 is False (covers line 614)
    # correction: 'ph', typo: 'f' -> 2-to-1 replacement, NOT a deletion
    assert typostats.is_one_letter_replacement('f', 'ph', allow_2to1=False, include_deletions=True) == []
    assert typostats.is_one_letter_replacement('f', 'ph', allow_2to1=True, include_deletions=False) == [('ph', 'f')]

def test_is_one_letter_replacement_deletions_false_branch():
    # Test 1-to-2 where it's an insertion but include_deletions is False
    assert typostats.is_one_letter_replacement('aa', 'a', allow_1to2=True, include_deletions=False) == []

    # Test 2-to-1 where it's a deletion but include_deletions is False
    assert typostats.is_one_letter_replacement('a', 'aa', allow_2to1=True, include_deletions=False) == []
