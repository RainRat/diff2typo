import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable=None, *_, **__: iterable if iterable is not None else MagicMock())

def test_count_mode_by_file_words(tmp_path):
    """Test count --by-file with individual words."""
    file1 = tmp_path / "file1.txt"
    file1.write_text("apple banana apple")
    file2 = tmp_path / "file2.txt"
    file2.write_text("apple cherry")

    output_file = tmp_path / "output.csv"

    multitool.count_mode(
        input_files=[str(file1), str(file2)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=False,
        by_file=True,
        output_format='csv'
    )

    content = output_file.read_text().splitlines()
    assert "item,files" in content
    assert "apple,2" in content
    assert "banana,1" in content
    assert "cherry,1" in content

def test_count_mode_by_file_lines_raw(tmp_path):
    """Test count --by-file --lines with raw=True (clean_items=False)."""
    file1 = tmp_path / "file1.txt"
    file1.write_text("line one\nline two")
    file2 = tmp_path / "file2.txt"
    file2.write_text("line one\nline three")

    output_file = tmp_path / "output.txt"

    multitool.count_mode(
        input_files=[str(file1), str(file2)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=False,
        lines=True,
        by_file=True,
        output_format='line',
        clean_items=False
    )

    content = output_file.read_text().splitlines()
    assert "line one: 2" in content
    assert "line two: 1" in content
    assert "line three: 1" in content

def test_count_mode_by_file_chars(tmp_path):
    """Test count --by-file --chars."""
    file1 = tmp_path / "file1.txt"
    file1.write_text("abc")
    file2 = tmp_path / "file2.txt"
    file2.write_text("abd")

    output_file = tmp_path / "output.txt"

    multitool.count_mode(
        input_files=[str(file1), str(file2)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        chars=True,
        by_file=True,
        output_format='line'
    )

    content = output_file.read_text().splitlines()
    assert "a: 2" in content
    assert "b: 2" in content
    assert "c: 1" in content
    assert "d: 1" in content

def test_count_mode_by_file_arrow_extra_metrics(tmp_path):
    """Test count --by-file --format arrow to cover extra_metrics logic."""
    file1 = tmp_path / "file1.txt"
    file1.write_text("apple")

    output_file = tmp_path / "output.txt"

    multitool.count_mode(
        input_files=[str(file1)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=False,
        by_file=True,
        output_format='arrow'
    )

    content = output_file.read_text()
    assert "Total files processed:" in content
    assert "1" in content
