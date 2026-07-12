import sys
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add root to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Ensure global state is reset before and after each test."""
    multitool._STDIN_CACHE = None
    if hasattr(multitool.detect_encoding, "_warning_shown"):
        delattr(multitool.detect_encoding, "_warning_shown")
    yield
    multitool._STDIN_CACHE = None
    if hasattr(multitool.detect_encoding, "_warning_shown"):
        delattr(multitool.detect_encoding, "_warning_shown")

def test_detect_encoding_basic(tmp_path):
    """Verifies detect_encoding with a simple UTF-8 file."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world", encoding="utf-8")

    # Ensure chardet is "available" for this test
    with patch("multitool._CHARDET_AVAILABLE", True), \
         patch("multitool.chardet") as mock_chardet:
        mock_chardet.detect.return_value = {'encoding': 'utf-8', 'confidence': 0.99}
        encoding = multitool.detect_encoding(str(test_file))
        assert encoding == "utf-8"

def test_detect_encoding_no_chardet(tmp_path):
    """Verifies detect_encoding when chardet is not available."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello", encoding="utf-8")

    with patch("multitool._CHARDET_AVAILABLE", False):
        encoding = multitool.detect_encoding(str(test_file))
        assert encoding is None
        assert getattr(multitool.detect_encoding, "_warning_shown", False) is True

def test_detect_encoding_failed(tmp_path):
    """Verifies detect_encoding when chardet fails or confidence is low."""
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"\xff\xfe\xfd") # Some random bytes

    with patch("multitool._CHARDET_AVAILABLE", True), \
         patch("multitool.chardet") as mock_chardet:
        # Case 1: No encoding detected
        mock_chardet.detect.return_value = {'encoding': None, 'confidence': 0.0}
        assert multitool.detect_encoding(str(test_file)) is None

        # Case 2: Low confidence
        mock_chardet.detect.return_value = {'encoding': 'utf-8', 'confidence': 0.4}
        assert multitool.detect_encoding(str(test_file)) is None

def test_read_file_lines_robust_stdin_cache(monkeypatch):
    """Verifies that multiple calls with '-' use the global _STDIN_CACHE."""
    input_data = "line1\nline2\n"

    mock_stdin = MagicMock()
    mock_stdin.buffer.read.return_value = input_data.encode('utf-8')
    monkeypatch.setattr(sys, "stdin", mock_stdin)

    # First call
    lines1 = multitool._read_file_lines_robust('-')
    assert lines1 == ["line1\n", "line2\n"]
    assert mock_stdin.buffer.read.call_count == 1

    # Second call should use cache
    lines2 = multitool._read_file_lines_robust('-')
    assert lines2 == ["line1\n", "line2\n"]
    assert mock_stdin.buffer.read.call_count == 1 # Still 1

def test_read_file_lines_robust_stdin_binary_utf8(monkeypatch):
    """Verifies successful UTF-8 decoding of binary stdin."""
    input_data = "hello \u2713".encode('utf-8')

    mock_stdin = MagicMock()
    mock_stdin.buffer.read.return_value = input_data
    monkeypatch.setattr(sys, "stdin", mock_stdin)

    lines = multitool._read_file_lines_robust('-')
    assert lines == ["hello \u2713"]

def test_read_file_lines_robust_stdin_binary_latin1(monkeypatch):
    """Verifies fallback to Latin-1 for binary stdin when UTF-8 fails."""
    # Invalid UTF-8 but valid Latin-1
    input_data = b"hello \xff"

    mock_stdin = MagicMock()
    mock_stdin.buffer.read.return_value = input_data
    monkeypatch.setattr(sys, "stdin", mock_stdin)

    lines = multitool._read_file_lines_robust('-')
    assert lines == ["hello \xff"]

def test_read_file_lines_robust_stdin_str(monkeypatch):
    """Verifies handling when sys.stdin.buffer is not available and read() returns str."""
    input_data = "line1\nline2"

    # Mock stdin WITHOUT buffer
    mock_stdin = MagicMock(spec=['read', 'encoding'])
    mock_stdin.read.return_value = input_data
    mock_stdin.encoding = 'utf-8'
    monkeypatch.setattr(sys, "stdin", mock_stdin)

    lines = multitool._read_file_lines_robust('-')
    assert lines == ["line1\n", "line2"]

def test_read_file_lines_robust_missing_file():
    """Verifies that it exits with code 1 when a file is missing."""
    with pytest.raises(SystemExit) as cm:
        multitool._read_file_lines_robust("non_existent_file_xyz.txt")
    assert cm.value.code == 1

def test_read_file_lines_robust_directory(tmp_path):
    """Verifies that it returns an empty list and warns when path is a directory."""
    test_dir = tmp_path / "some_dir"
    test_dir.mkdir()

    lines = multitool._read_file_lines_robust(str(test_dir))
    assert lines == []

def test_read_file_lines_robust_encoding_fallback(tmp_path):
    """Verifies the fallback logic (UTF-8 -> Detection -> Latin-1)."""
    test_file = tmp_path / "fallback.txt"
    # Write Latin-1 content
    test_file.write_bytes(b"hello \xff")

    # 1. UTF-8 fails, Detection succeeds
    with patch("multitool.detect_encoding") as mock_detect:
        mock_detect.return_value = "latin-1"
        lines = multitool._read_file_lines_robust(str(test_file))
        assert lines == ["hello \xff"]
        mock_detect.assert_called_once()

    # 2. UTF-8 fails, Detection fails, Fallback to latin-1
    with patch("multitool.detect_encoding") as mock_detect:
        mock_detect.return_value = None
        lines = multitool._read_file_lines_robust(str(test_file))
        assert lines == ["hello \xff"]
        mock_detect.assert_called_once()

    # 3. UTF-8 fails, Detection returns WRONG encoding, final fallback to latin-1
    with patch("multitool.detect_encoding") as mock_detect:
        mock_detect.return_value = "ascii" # This will fail for \xff
        lines = multitool._read_file_lines_robust(str(test_file))
        assert lines == ["hello \xff"]
        mock_detect.assert_called_once()
