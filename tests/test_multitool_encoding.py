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

def test_detect_encoding(caplog, monkeypatch, tmp_path):
    # Create a dummy file
    f = tmp_path / "test.txt"
    f.write_text("dummy")

    # Case 1: chardet not available
    monkeypatch.setattr(multitool, "_CHARDET_AVAILABLE", False)
    with caplog.at_level(logging.WARNING):
        assert multitool.detect_encoding(str(f)) is None
    assert "chardet not installed" in caplog.text
    caplog.clear()

    # Case 2: chardet available, low confidence
    monkeypatch.setattr(multitool, "_CHARDET_AVAILABLE", True)
    mock_chardet = MagicMock()
    mock_chardet.detect.return_value = {'encoding': 'utf-8', 'confidence': 0.3}
    monkeypatch.setattr(multitool, "chardet", mock_chardet)

    with caplog.at_level(logging.WARNING):
        assert multitool.detect_encoding(str(f)) is None
    assert "Failed to reliably detect" in caplog.text
    caplog.clear()

    # Case 3: chardet available, high confidence
    mock_chardet.detect.return_value = {'encoding': 'utf-8', 'confidence': 0.9}
    with caplog.at_level(logging.INFO):
        assert multitool.detect_encoding(str(f)) == 'utf-8'
    assert "Detected encoding 'utf-8'" in caplog.text

def test_load_and_clean_file_stdin_success(monkeypatch):
    """Test reading from stdin successfully."""
    mock_stdin = MagicMock()
    mock_stdin.readlines.return_value = ["line1\n", "line2\n"]
    mock_stdin.encoding = 'utf-8'
    monkeypatch.setattr(sys, "stdin", mock_stdin)

    raw, cleaned, unique = multitool._load_and_clean_file('-', 1, 100)
    assert raw == ["line1", "line2"]
    assert cleaned == ["line", "line"]
    assert unique == ["line"]

def test_load_and_clean_file_stdin_unicode_error(monkeypatch, caplog):
    """Test reading from stdin with UnicodeDecodeError."""
    mock_stdin = MagicMock()
    # Simulate readlines raising UnicodeDecodeError
    mock_stdin.readlines.side_effect = UnicodeDecodeError('utf-8', b'', 0, 1, 'bad')
    monkeypatch.setattr(sys, "stdin", mock_stdin)

    with caplog.at_level(logging.WARNING):
        raw, cleaned, unique = multitool._load_and_clean_file('-', 1, 100)

    assert raw == []
    assert cleaned == []
    assert unique == []
    assert "Reading from stdin failed with encoding errors" in caplog.text


def test_arrow_mode_latin1(tmp_path):
    input_file = tmp_path / "latin1_arrow.txt"
    input_file.write_bytes(b"caf\xe9 -> coffee\n")
    output_file = tmp_path / "output.txt"

    # Use clean_items=False to preserve the accent during verification
    multitool.arrow_mode([str(input_file)], str(output_file), 1, 100, False, clean_items=False)
    assert output_file.read_text(encoding="utf-8").strip() == "café"


def test_line_mode_latin1(tmp_path):
    input_file = tmp_path / "latin1_line.txt"
    input_file.write_bytes(b"caf\xe9\n")
    output_file = tmp_path / "output.txt"

    multitool.line_mode(
        [str(input_file)], str(output_file), 1, 100, False, clean_items=False
    )
    assert output_file.read_text(encoding="utf-8").strip() == "café"


def test_csv_mode_latin1(tmp_path):
    input_file = tmp_path / "latin1.csv"
    input_file.write_bytes(b"caf\xe9,coffee\n")
    output_file = tmp_path / "output.txt"

    multitool.csv_mode(
        [str(input_file)], str(output_file), 1, 100, False, first_column=True, clean_items=False
    )
    assert output_file.read_text(encoding="utf-8").strip() == "café"


def test_yaml_mode_latin1(tmp_path):
    input_file = tmp_path / "latin1.yaml"
    input_file.write_bytes(b"key: caf\xe9\n")
    output_file = tmp_path / "output.txt"

    multitool.yaml_mode(
        [str(input_file)], str(output_file), 1, 100, False, key="key", clean_items=False
    )
    assert output_file.read_text(encoding="utf-8").strip() == "café"


def test_count_mode_latin1(tmp_path):
    input_file = tmp_path / "latin1_count.txt"
    input_file.write_bytes(b"caf\xe9 caf\xe9 coffee\n")
    output_file = tmp_path / "output.txt"

    multitool.count_mode(
        [str(input_file)], str(output_file), 1, 100, False, clean_items=False
    )
    content = output_file.read_text(encoding="utf-8")
    assert "café: 2" in content


def test_regex_mode_latin1(tmp_path):
    input_file = tmp_path / "latin1_regex.txt"
    input_file.write_bytes(b"caf\xe9 coffee\n")
    output_file = tmp_path / "output.txt"

    multitool.regex_mode([str(input_file)], str(output_file), 1, 100, False, pattern=r"caf.")
    assert output_file.read_text(encoding="utf-8").strip() == "café"


def test_check_mode_latin1(tmp_path):
    input_file = tmp_path / "latin1_check.csv"
    input_file.write_bytes(b"caf\xe9,caf\xe9\n")
    output_file = tmp_path / "output.txt"

    multitool.check_mode(
        [str(input_file)], str(output_file), 1, 100, False, clean_items=False
    )
    assert output_file.read_text(encoding="utf-8").strip() == "café"


def test_map_mode_latin1(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("café", encoding="utf-8")
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_bytes(b"caf\xe9,coffee\n")
    output_file = tmp_path / "output.txt"

    multitool.map_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        clean_items=False,
    )
    assert output_file.read_text(encoding="utf-8").strip() == "coffee"


def test_json_mode_latin1(tmp_path):
    input_file = tmp_path / "latin1.json"
    input_file.write_bytes(b'{"key": "caf\xe9"}')
    output_file = tmp_path / "output.txt"

    multitool.json_mode(
        [str(input_file)], str(output_file), 1, 100, False, key="key", clean_items=False
    )
    assert output_file.read_text(encoding="utf-8").strip() == "café"


def test_backtick_mode_latin1(tmp_path):
    input_file = tmp_path / "latin1_backtick.txt"
    input_file.write_bytes(b"error: `caf\xe9` should be `coffee`\n")
    output_file = tmp_path / "output.txt"

    multitool.backtick_mode(
        [str(input_file)], str(output_file), 1, 100, False, clean_items=False
    )
    # Both 'café' and 'coffee' should be extracted as they follow the 'error:' marker
    assert output_file.read_text(encoding="utf-8").strip() == "café\ncoffee"
