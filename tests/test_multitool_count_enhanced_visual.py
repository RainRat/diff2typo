import json
import csv
import sys
from pathlib import Path
import pytest
from io import StringIO

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_count_mode_smart_splitting(tmp_path):
    input_file = tmp_path / "input.txt"
    # CamelCase: Camel(1), Case(1)
    input_file.write_text("CamelCase")
    output_file = tmp_path / "output.txt"

    # With smart splitting
    multitool.count_mode([str(input_file)], str(output_file), 1, 10, False, smart=True, min_count=1)
    lines = output_file.read_text().splitlines()
    assert "camel: 1" in lines
    assert "case: 1" in lines

def test_count_mode_custom_delimiter(tmp_path):
    input_file = tmp_path / "input.txt"
    # Semicolon delimiter: a(1), b(1), c(1)
    input_file.write_text("a;b;c")
    output_file = tmp_path / "output.txt"

    # With semicolon delimiter
    multitool.count_mode([str(input_file)], str(output_file), 1, 10, False, delimiter=';', min_count=1)
    lines = output_file.read_text().splitlines()
    assert "a: 1" in lines
    assert "b: 1" in lines
    assert "c: 1" in lines

def test_count_mode_visual_report_content(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple banana apple")
    output_file = tmp_path / "output.txt"

    # Mock isatty to False to avoid ANSI colors in verification if needed,
    # but we'll just check for content.
    # Actually, we want to check for the table structure.

    multitool.count_mode([str(input_file)], str(output_file), 1, 10, False, output_format='arrow', min_count=1)
    content = output_file.read_text()

    assert "ITEM" in content
    assert "COUNT" in content
    assert "VISUAL" in content
    assert "apple" in content
    assert "banana" in content
    assert "66.7%" in content
    assert "33.3%" in content
    # Check for Unicode block characters (at least one)
    assert any(c in content for c in ["█", "▏", "▎", "▍", "▌", "▋", "▊", "▉"])

def test_count_mode_raw_preservation(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("Apple APPLE")
    output_file = tmp_path / "output.txt"

    # With raw=True (clean_items=False)
    multitool.count_mode([str(input_file)], str(output_file), 1, 10, False, clean_items=False, min_count=1)
    lines = output_file.read_text().splitlines()
    assert "Apple: 1" in lines
    assert "APPLE: 1" in lines

    # Without raw (default)
    multitool.count_mode([str(input_file)], str(output_file), 1, 10, False, clean_items=True, min_count=1)
    lines = output_file.read_text().splitlines()
    assert "apple: 2" in lines
