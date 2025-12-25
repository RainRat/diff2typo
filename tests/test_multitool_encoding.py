import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

# Replicate the disable_tqdm fixture as it's used in multitool tests
@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_load_and_clean_file_with_latin1_content(tmp_path, caplog):
    """
    Test that a file with latin-1 content is correctly loaded (either via chardet detection or fallback).
    This replaces the fragile test_load_and_clean_file_encoding_fallback.
    """
    data_file = tmp_path / "latin1.txt"
    data_file.write_bytes("café\nnaïve\n".encode("latin-1"))

    caplog.set_level(logging.WARNING)

    # We don't mock detect_encoding here; we let it run (or fail if chardet missing)
    # The goal is to verify the END RESULT is correct loading.

    raw_items, cleaned_items, unique_items = multitool._load_and_clean_file(
        str(data_file),
        1,
        10,
    )

    assert raw_items == ["café", "naïve"]
    assert cleaned_items == ["caf", "nave"]
    assert unique_items == ["caf", "nave"]

    # Check that we logged *something* about encoding if UTF-8 failed.
    # We expect "UTF-8 decoding failed"
    assert "UTF-8 decoding failed" in caplog.text

def test_load_and_clean_file_detection_failure_fallback(tmp_path, caplog):
    """
    Test the fallback to latin-1 when encoding detection explicitly returns None.
    This covers the 'else' block in _load_and_clean_file's detection logic.
    """
    data_file = tmp_path / "unknown_encoding.txt"
    data_file.write_bytes("café\nnaïve\n".encode("latin-1"))

    caplog.set_level(logging.WARNING)

    # Mock detect_encoding to return None
    with patch('multitool.detect_encoding', return_value=None):
        raw_items, cleaned_items, unique_items = multitool._load_and_clean_file(
            str(data_file),
            1,
            10,
        )

    assert raw_items == ["café", "naïve"]
    assert "Encoding detection failed. Fallback to latin-1" in caplog.text

def test_load_and_clean_file_detected_encoding_fails(tmp_path, caplog):
    """
    Test the fallback to latin-1 when the detected encoding still fails to decode.
    This covers the nested 'except UnicodeDecodeError' block.
    """
    data_file = tmp_path / "fake_utf8.txt"
    # Write invalid UTF-8 bytes that are valid latin-1
    data_file.write_bytes(b"\xff\xfe\xfd")

    caplog.set_level(logging.WARNING)

    # Mock detect_encoding to return 'utf-8' (which will fail to decode these bytes)
    # We use 'utf-8' because the code tries to open with detected_encoding.
    # Note: The code first tries 'utf-8' naturally. If that fails, it calls detect_encoding.
    # We need detect_encoding to return something that ALSO fails.
    # Let's say detect_encoding returns 'utf-8' (maybe falsely confident).
    # Then the code tries opening with 'utf-8' again inside the if block, which fails again.

    with patch('multitool.detect_encoding', return_value='utf-8'):
        # We need to ensure the FIRST open (standard utf-8) fails. It will because of the bytes.
        raw_items, _, _ = multitool._load_and_clean_file(
            str(data_file),
            1,
            10,
        )

    # Since the input bytes are \xff\xfe\xfd, in latin-1 they are "ÿþý"
    assert raw_items == ["ÿþý"]
    assert "Detected encoding 'utf-8' failed" in caplog.text
    assert "Fallback to latin-1" in caplog.text
