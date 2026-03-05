import logging
import sys
import os
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

def test_levenshtein_distance():
    # Identical strings
    assert multitool.levenshtein_distance("kitten", "kitten") == 0
    assert multitool.levenshtein_distance("", "") == 0

    # One empty string
    assert multitool.levenshtein_distance("kitten", "") == 6
    assert multitool.levenshtein_distance("", "kitten") == 6

    # Single edits
    assert multitool.levenshtein_distance("kitten", "sitten") == 1  # Substitution
    assert multitool.levenshtein_distance("kitten", "kittens") == 1 # Insertion
    assert multitool.levenshtein_distance("kitten", "kitte") == 1   # Deletion

    # Multiple edits
    assert multitool.levenshtein_distance("kitten", "sitting") == 3
    assert multitool.levenshtein_distance("flaw", "lawn") == 2

    # Length swap branch (if len(s1) < len(s2))
    assert multitool.levenshtein_distance("abc", "abcdef") == 3
    assert multitool.levenshtein_distance("abcdef", "abc") == 3

def test_smart_split():
    # Simple words
    assert multitool._smart_split("hello") == ["hello"]

    # CamelCase / PascalCase
    assert multitool._smart_split("camelCase") == ["camel", "Case"]
    assert multitool._smart_split("PascalCase") == ["Pascal", "Case"]

    # Acronyms
    assert multitool._smart_split("HTMLParser") == ["HTML", "Parser"]
    assert multitool._smart_split("simpleXML") == ["simple", "XML"]
    assert multitool._smart_split("HTTPRequest") == ["HTTP", "Request"]

    # Snake case and non-alphanumeric
    assert multitool._smart_split("snake_case_string") == ["snake", "case", "string"]
    assert multitool._smart_split("kebab-case-string") == ["kebab", "case", "string"]
    assert multitool._smart_split("multiple...dots") == ["multiple", "dots"]

    # Numbers
    assert multitool._smart_split("version123beta") == ["version", "123", "beta"]
    assert multitool._smart_split("123numbersFirst") == ["123", "numbers", "First"]

    # Mixed
    assert multitool._smart_split("JSON123Parser_v2") == ["JSON", "123", "Parser", "v", "2"]

    # Edge cases
    assert multitool._smart_split("") == []
    assert multitool._smart_split("   ") == []
    assert multitool._smart_split("!!!") == []

def test_detect_encoding_no_chardet(caplog):
    # Mock _CHARDET_AVAILABLE to False
    with patch("multitool._CHARDET_AVAILABLE", False):
        with caplog.at_level(logging.WARNING):
            result = multitool.detect_encoding("some_path")
            assert result is None
            assert "chardet not installed" in caplog.text

def test_detect_encoding_success(tmp_path, caplog):
    # Mock _CHARDET_AVAILABLE to True
    # We need to mock chardet as well since it might be installed
    with patch("multitool._CHARDET_AVAILABLE", True), \
         patch("multitool.chardet") as mock_chardet:

        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"hello")

        mock_chardet.detect.return_value = {"encoding": "utf-8", "confidence": 0.99}

        with caplog.at_level(logging.INFO):
            result = multitool.detect_encoding(str(test_file))
            assert result == "utf-8"
            assert "Detected encoding 'utf-8'" in caplog.text

def test_detect_encoding_low_confidence(tmp_path, caplog):
    with patch("multitool._CHARDET_AVAILABLE", True), \
         patch("multitool.chardet") as mock_chardet:

        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"hello")

        mock_chardet.detect.return_value = {"encoding": "utf-8", "confidence": 0.4}

        with caplog.at_level(logging.WARNING):
            result = multitool.detect_encoding(str(test_file))
            assert result is None
            assert "Failed to reliably detect encoding" in caplog.text

def test_read_file_lines_robust_utf8(tmp_path):
    test_file = tmp_path / "utf8.txt"
    content = "hello\nworld\n"
    test_file.write_text(content, encoding="utf-8")

    lines = multitool._read_file_lines_robust(str(test_file))
    assert lines == ["hello\n", "world\n"]

def test_read_file_lines_robust_latin1(tmp_path, caplog):
    test_file = tmp_path / "latin1.txt"
    # Create a file that is valid latin-1 but NOT valid utf-8
    content_bytes = "héllo\n".encode("latin-1")
    test_file.write_bytes(content_bytes)

    # We mock detect_encoding to return None to force latin-1 fallback
    with patch("multitool.detect_encoding", return_value=None):
        with caplog.at_level(logging.WARNING):
            lines = multitool._read_file_lines_robust(str(test_file))
            assert lines == ["héllo\n"]
            assert "Encoding detection failed. Fallback to latin-1" in caplog.text

def test_read_file_lines_robust_detected(tmp_path, caplog):
    test_file = tmp_path / "detected.txt"
    content_bytes = "héllo\n".encode("iso-8859-1")
    test_file.write_bytes(content_bytes)

    with patch("multitool.detect_encoding", return_value="iso-8859-1"):
        with caplog.at_level(logging.WARNING):
            lines = multitool._read_file_lines_robust(str(test_file))
            assert lines == ["héllo\n"]
            assert "Using detected encoding 'iso-8859-1'" in caplog.text

def test_read_file_lines_robust_detected_fails_fallback(tmp_path, caplog):
    test_file = tmp_path / "detected_fail.txt"
    content_bytes = "héllo\n".encode("latin-1")
    test_file.write_bytes(content_bytes)

    # Mock detection to return a WRONG encoding that will fail
    with patch("multitool.detect_encoding", return_value="ascii"):
        with caplog.at_level(logging.WARNING):
            lines = multitool._read_file_lines_robust(str(test_file))
            assert lines == ["héllo\n"]
            assert "Detected encoding 'ascii' failed" in caplog.text

def test_read_file_lines_robust_stdin(monkeypatch, caplog):
    # Mock sys.stdin.buffer to return bytes
    content = "stdin\nlines\n"
    mock_stdin = MagicMock()
    mock_stdin.buffer.read.return_value = content.encode("utf-8")
    monkeypatch.setattr(sys, "stdin", mock_stdin)

    # Reset cache before test
    multitool._STDIN_CACHE = None

    with caplog.at_level(logging.INFO):
        lines = multitool._read_file_lines_robust("-")
        assert lines == ["stdin\n", "lines\n"]
        assert "Reading from standard input..." in caplog.text

        # Test cache
        lines2 = multitool._read_file_lines_robust("-")
        assert lines2 == ["stdin\n", "lines\n"]
        assert "Using cached standard input..." in caplog.text

def test_read_file_lines_robust_stdin_latin1(monkeypatch, caplog):
    # Mock sys.stdin.buffer to return non-utf8 bytes
    content_bytes = "héllo\n".encode("latin-1")
    mock_stdin = MagicMock()
    mock_stdin.buffer.read.return_value = content_bytes
    monkeypatch.setattr(sys, "stdin", mock_stdin)

    # Reset cache before test
    multitool._STDIN_CACHE = None

    lines = multitool._read_file_lines_robust("-")
    assert lines == ["héllo\n"]
