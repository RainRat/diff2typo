import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import typostats

def test_detect_encoding_chardet_missing(caplog, monkeypatch, tmp_path):
    """Test detect_encoding returns None if chardet is missing."""
    f = tmp_path / "test.txt"
    f.write_text("dummy")

    monkeypatch.setattr(typostats, "_CHARDET_AVAILABLE", False)
    with caplog.at_level(logging.WARNING):
        assert typostats.detect_encoding(str(f)) is None
    assert "chardet not installed" in caplog.text

def test_detect_encoding_low_confidence(caplog, monkeypatch, tmp_path):
    """Test detect_encoding returns None if confidence is low."""
    f = tmp_path / "test.txt"
    f.write_text("dummy")

    monkeypatch.setattr(typostats, "_CHARDET_AVAILABLE", True)
    mock_chardet = MagicMock()
    mock_chardet.detect.return_value = {'encoding': 'utf-8', 'confidence': 0.3}
    monkeypatch.setattr(typostats, "chardet", mock_chardet)

    with caplog.at_level(logging.WARNING):
        assert typostats.detect_encoding(str(f)) is None
    assert "Failed to reliably detect encoding" in caplog.text

def test_detect_encoding_success(caplog, monkeypatch, tmp_path):
    """Test detect_encoding returns encoding if confidence is high."""
    f = tmp_path / "test.txt"
    f.write_text("dummy")

    monkeypatch.setattr(typostats, "_CHARDET_AVAILABLE", True)
    mock_chardet = MagicMock()
    mock_chardet.detect.return_value = {'encoding': 'shift_jis', 'confidence': 0.9}
    monkeypatch.setattr(typostats, "chardet", mock_chardet)

    with caplog.at_level(logging.INFO):
        assert typostats.detect_encoding(str(f)) == 'shift_jis'
    assert "Detected encoding: shift_jis" in caplog.text

def test_load_lines_from_file_stdin(monkeypatch, caplog):
    """Test loading from stdin."""
    mock_stdin = MagicMock()
    mock_stdin.readlines.return_value = ["line1\n", "line2\n"]
    monkeypatch.setattr(sys, "stdin", mock_stdin)

    with caplog.at_level(logging.INFO):
        lines = typostats.load_lines_from_file('-')

    assert lines == ["line1\n", "line2\n"]
    assert "Reading from stdin" in caplog.text

def test_load_lines_from_file_utf8_success(tmp_path):
    """Test loading a standard UTF-8 file."""
    f = tmp_path / "utf8.txt"
    f.write_text("café", encoding="utf-8")

    lines = typostats.load_lines_from_file(str(f))
    assert lines == ["café"]

def test_load_lines_from_file_utf8_fail_detect_success(tmp_path, caplog, monkeypatch):
    """Test UTF-8 failure, successful detection, and successful load with detected encoding."""
    f = tmp_path / "shift_jis.txt"
    # Use Japanese characters that are valid in Shift-JIS but invalid in UTF-8 when read as such
    content = "日本語".encode("shift_jis")
    f.write_bytes(content)

    monkeypatch.setattr(typostats, "_CHARDET_AVAILABLE", True)
    mock_chardet = MagicMock()
    mock_chardet.detect.return_value = {'encoding': 'shift_jis', 'confidence': 0.9}
    monkeypatch.setattr(typostats, "chardet", mock_chardet)

    with caplog.at_level(logging.WARNING):
        lines = typostats.load_lines_from_file(str(f))

    # "日本語" is read as a single line (or part of one)
    assert lines == ["日本語"]
    assert "UTF-8 decoding failed" in caplog.text
    assert "Using detected encoding: shift_jis" in caplog.text

def test_load_lines_from_file_fallback_latin1(tmp_path, caplog, monkeypatch):
    """Test UTF-8 failure, detection failure, and fallback to latin-1."""
    f = tmp_path / "unknown.txt"
    # Write bytes that are invalid UTF-8 but valid latin-1
    content = b"caf\xe9" # é in latin-1
    f.write_bytes(content)

    # Force detection failure
    monkeypatch.setattr(typostats, "_CHARDET_AVAILABLE", True)
    mock_chardet = MagicMock()
    mock_chardet.detect.return_value = {'encoding': None, 'confidence': 0.0}
    monkeypatch.setattr(typostats, "chardet", mock_chardet)

    with caplog.at_level(logging.WARNING):
        lines = typostats.load_lines_from_file(str(f))

    assert lines == ["café"]
    assert "UTF-8 decoding failed" in caplog.text
    assert "Fallback to latin1" in caplog.text

def test_load_lines_from_file_detected_encoding_fails_fallback(tmp_path, caplog, monkeypatch):
    """Test UTF-8 failure, detected encoding failure, and fallback to latin-1."""
    f = tmp_path / "tricky.txt"
    # Valid latin-1 bytes
    content = b"\xff\xfe\xfd"
    f.write_bytes(content)

    # Mock detection returning something that will fail to decode these bytes (e.g., utf-8 again)
    monkeypatch.setattr(typostats, "_CHARDET_AVAILABLE", True)
    mock_chardet = MagicMock()
    mock_chardet.detect.return_value = {'encoding': 'utf-8', 'confidence': 0.6}
    monkeypatch.setattr(typostats, "chardet", mock_chardet)

    with caplog.at_level(logging.WARNING):
        lines = typostats.load_lines_from_file(str(f))

    # In latin-1: \xff=ÿ, \xfe=þ, \xfd=ý
    assert lines == ["ÿþý"]
    assert "Detected encoding utf-8 failed" in caplog.text
    assert "Fallback to latin1" in caplog.text

def test_load_lines_from_file_not_found(tmp_path, caplog):
    """Test handling of missing file."""
    f = tmp_path / "missing.txt"

    with caplog.at_level(logging.ERROR):
        lines = typostats.load_lines_from_file(str(f))

    assert lines is None
    assert "File not found" in caplog.text
