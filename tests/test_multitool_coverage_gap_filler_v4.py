
import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_read_file_lines_robust_not_found(caplog):
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as cm:
            multitool._read_file_lines_robust("nonexistent_file_12345.txt")
    assert cm.value.code == 1
    assert "Input file 'nonexistent_file_12345.txt' not found." in caplog.text

def test_main_missing_secondary_file_zip(monkeypatch, caplog):
    monkeypatch.setattr(sys, "argv", ["multitool.py", "zip"])
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as cm:
            multitool.main()
    assert cm.value.code == 1
    assert "Zip mode requires a secondary file" in caplog.text

def test_main_missing_mapping_map(monkeypatch, caplog):
    monkeypatch.setattr(sys, "argv", ["multitool.py", "map"])
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as cm:
            multitool.main()
    assert cm.value.code == 1
    assert "Map mode requires a mapping file" in caplog.text

def test_main_missing_query_search(monkeypatch, caplog):
    monkeypatch.setattr(sys, "argv", ["multitool.py", "search"])
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as cm:
            multitool.main()
    assert cm.value.code == 1
    assert "Search mode requires a search query" in caplog.text

def test_main_empty_delimiter_normalization(monkeypatch, tmp_path):
    input_file = tmp_path / "input.csv"
    input_file.write_text("apple,banana\ncherry,date")
    output_file = tmp_path / "output.txt"

    monkeypatch.setattr(sys, "argv", [
        "multitool.py", "csv", str(input_file),
        "--output", str(output_file),
        "--delimiter", "",
        "--first-column",
        "--min-length", "1"
    ])

    multitool._STDIN_CACHE = None
    multitool.main()
    assert output_file.read_text().splitlines() == ["apple", "cherry"]

def test_extract_json_items_empty_content(tmp_path):
    f = tmp_path / "empty.json"
    f.write_text("   ")
    items = list(multitool._extract_json_items(str(f), "key"))
    assert items == []

def test_parse_markdown_table_row_no_trailing_pipe():
    row = "| a | b"
    result = multitool._parse_markdown_table_row(row)
    assert result == ["a", "b"]

def test_main_file2_fallback_stdin(monkeypatch, tmp_path):
    f2 = tmp_path / "f2.txt"
    f2.write_text("second")

    mock_stdin = MagicMock()
    mock_stdin.buffer.read.return_value = b"first\n"
    mock_stdin.read.return_value = "first\n"
    monkeypatch.setattr(sys, "stdin", mock_stdin)
    multitool._STDIN_CACHE = None

    output_file = tmp_path / "out.txt"

    monkeypatch.setattr(sys, "argv", [
        "multitool.py", "zip", str(f2), "--output", str(output_file), "--min-length", "1"
    ])

    multitool.main()
    assert "first -> second" in output_file.read_text()

def test_main_mapping_fallback_stdin(monkeypatch, tmp_path):
    m = tmp_path / "map.csv"
    m.write_text("typo,fix")

    mock_stdin = MagicMock()
    mock_stdin.read.return_value = "typo\n"
    mock_stdin.buffer.read.return_value = b"typo\n"
    monkeypatch.setattr(sys, "stdin", mock_stdin)
    multitool._STDIN_CACHE = None

    output_file = tmp_path / "out.txt"

    monkeypatch.setattr(sys, "argv", [
        "multitool.py", "scrub", str(m), "--output", str(output_file), "--min-length", "1"
    ])

    multitool.main()
    assert output_file.read_text().strip() == "fix"

def test_main_query_fallback_stdin(monkeypatch, tmp_path):
    mock_stdin = MagicMock()
    mock_stdin.read.return_value = "apple\nbanana\n"
    mock_stdin.buffer.read.return_value = b"apple\nbanana\n"
    monkeypatch.setattr(sys, "stdin", mock_stdin)
    multitool._STDIN_CACHE = None

    output_file = tmp_path / "out.txt"

    monkeypatch.setattr(sys, "argv", [
        "multitool.py", "search", "apple", "--output", str(output_file), "--min-length", "1"
    ])

    multitool.main()
    assert "apple" in output_file.read_text()
    assert "banana" not in output_file.read_text()
