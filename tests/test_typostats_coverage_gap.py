import sys
from unittest.mock import MagicMock, patch
from typostats import (
    _should_enable_color,
    is_one_letter_replacement,
    _read_file_lines_robust,
    main
)

def test_should_enable_color(monkeypatch):
    # Test NO_COLOR
    monkeypatch.setenv("NO_COLOR", "1")
    assert _should_enable_color(sys.stdout) is False
    monkeypatch.delenv("NO_COLOR")

    # Test FORCE_COLOR
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert _should_enable_color(sys.stdout) is True
    monkeypatch.delenv("FORCE_COLOR")

    # Test stream with isatty
    mock_stream = MagicMock()
    mock_stream.isatty.return_value = True
    assert _should_enable_color(mock_stream) is True

    mock_stream.isatty.return_value = False
    assert _should_enable_color(mock_stream) is False

def test_typostats_main_format_detection_various(tmp_path):
    input_file = tmp_path / "input.csv"
    input_file.write_text("typo,correction\nteh,the")

    # Mapping of output files and expected detected formats
    cases = [
        ("output.json", "json"),
        ("output.csv", "csv"),
        ("output.yaml", "yaml"),
        ("output.yml", "yaml"),
        ("output.arrow", "arrow"),
        ("output.txt", "arrow"),
        ("output.unknown", "arrow"),
        ("outputfile", "arrow"),
        ("-", "arrow"),
        ("", "arrow"),
    ]

    for output_name, expected_format in cases:
        out_path = str(tmp_path / output_name) if output_name and output_name != "-" else output_name
        with patch("sys.argv", ["typostats.py", str(input_file), "-o", out_path]), \
             patch("typostats.generate_report") as mock_report:
            try:
                main()
            except SystemExit:
                pass
            mock_report.assert_called_once()
            assert mock_report.call_args.kwargs["output_format"] == expected_format

def test_is_one_letter_replacement_disallowed_patterns():
    # To hit line 594: (allow_1to2 or include_deletions) must be True, but allow_1to2 False.
    # 'a' -> 'bc' is a 1-to-2 replacement.
    assert is_one_letter_replacement("bc", "a", allow_1to2=False, include_deletions=True) == []

    # To hit line 614: (allow_2to1 or include_deletions) must be True, but allow_2to1 False.
    # 'ab' -> 'c' is a 2-to-1 replacement.
    assert is_one_letter_replacement("c", "ab", allow_2to1=False, include_deletions=True) == []

def test_read_file_lines_robust_stdin_string(monkeypatch):
    import typostats
    # Clear cache
    monkeypatch.setattr(typostats, "_STDIN_CACHE", None)

    mock_stdin = MagicMock()
    if hasattr(mock_stdin, 'buffer'):
        del mock_stdin.buffer
    mock_stdin.read.return_value = "line1\nline2\n"

    with patch("sys.stdin", mock_stdin):
        lines = _read_file_lines_robust("-")
        assert lines == ["line1\n", "line2\n"]

def test_read_file_lines_robust_stdin_binary_fallback(monkeypatch):
    import typostats
    monkeypatch.setattr(typostats, "_STDIN_CACHE", None)

    mock_stdin = MagicMock()
    mock_buffer = MagicMock()
    # Binary data that is valid latin-1 but invalid utf-8
    # 0xE9 is é in latin-1
    mock_buffer.read.return_value = b"\xe9\n"
    mock_stdin.buffer = mock_buffer

    with patch("sys.stdin", mock_stdin):
        lines = _read_file_lines_robust("-")
        assert lines == ["\xe9\n"]

def test_read_file_lines_robust_directory(tmp_path):
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()

    lines = _read_file_lines_robust(str(dir_path))
    assert lines == []

def test_read_file_lines_robust_file_encoding_fallback(tmp_path):
    # Create a file with latin-1 encoding
    file_path = tmp_path / "latin1.txt"
    with open(file_path, "wb") as f:
        f.write(b"\xe9\n")

    # Branch 1: detect_encoding succeeds
    with patch("typostats.detect_encoding", return_value="latin-1"):
        lines = _read_file_lines_robust(str(file_path))
        assert lines == ["\xe9\n"]

    # Branch 2: detect_encoding fails or itself leads to error
    with patch("typostats.detect_encoding", return_value=None):
        lines = _read_file_lines_robust(str(file_path))
        assert lines == ["\xe9\n"]

def test_read_file_lines_robust_file_detect_encoding_fails_midway(tmp_path):
    file_path = tmp_path / "latin1_v2.txt"
    with open(file_path, "wb") as f:
        f.write(b"\xe9\n")

    with patch("typostats.detect_encoding", return_value="utf-8"): # Wrong encoding detected
        # This will trigger UnicodeDecodeError again in the try block at line 435
        # and fall back to latin-1 at line 444
        lines = _read_file_lines_robust(str(file_path))
        assert lines == ["\xe9\n"]

def test_typostats_main_basic(tmp_path):
    input_file = tmp_path / "input.csv"
    input_file.write_text("typo,correction\nteh,the")

    with patch("sys.argv", ["typostats.py", str(input_file), "--format", "json"]):
        try:
            main()
        except SystemExit:
            pass
