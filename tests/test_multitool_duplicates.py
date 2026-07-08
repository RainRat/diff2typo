import os
import json
import yaml
import pytest
import multitool
from pathlib import Path
import io

def test_duplicates_mode_basic(tmp_path):
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file3 = tmp_path / "file3.txt"

    content = "identical content"
    file1.write_text(content)
    file2.write_text(content)
    file3.write_text("different content")

    output = tmp_path / "output.json"
    multitool.duplicates_mode(
        input_files=[str(file1), str(file2), str(file3)],
        output_file=str(output),
        min_length=0,
        max_length=10**9,
        output_format='json',
        quiet=True
    )

    with open(output, 'r') as f:
        data = json.load(f)

    # In duplicates mode, the first file is the "original"
    # output maps Duplicate -> Original + Size
    assert str(file2) in data
    assert data[str(file2)].startswith(str(file1))
    assert "17 bytes" in data[str(file2)]
    assert str(file3) not in data

def test_duplicates_mode_filters(tmp_path):
    file1 = tmp_path / "small.txt"
    file2 = tmp_path / "small_dup.txt"
    file3 = tmp_path / "large.txt"
    file4 = tmp_path / "large_dup.txt"

    file1.write_text("small")
    file2.write_text("small")
    file3.write_text("this is a larger file content")
    file4.write_text("this is a larger file content")

    output = tmp_path / "output.json"

    # Filter for large files only
    multitool.duplicates_mode(
        input_files=[str(file1), str(file2), str(file3), str(file4)],
        output_file=str(output),
        min_length=10,
        max_length=100,
        output_format='json',
        quiet=True
    )

    with open(output, 'r') as f:
        data = json.load(f)

    assert str(file4) in data
    assert str(file2) not in data

def test_duplicates_mode_stdin(tmp_path, monkeypatch):
    file1 = tmp_path / "file1.txt"
    content = "stdin content"
    file1.write_text(content)

    # Mock stdin
    monkeypatch.setattr('sys.stdin', io.StringIO(content))

    # Reset STDIN cache
    multitool._STDIN_CACHE = None

    output = tmp_path / "output.json"
    multitool.duplicates_mode(
        input_files=[str(file1), '-'],
        output_file=str(output),
        min_length=0,
        max_length=10**9,
        output_format='json',
        quiet=True
    )

    with open(output, 'r') as f:
        data = json.load(f)

    # '-' is considered a duplicate of file1
    assert '-' in data
    assert data['-'].startswith(str(file1))

def test_duplicates_mode_formats(tmp_path):
    file1 = tmp_path / "f1.txt"
    file2 = tmp_path / "f2.txt"
    content = "content"
    file1.write_text(content)
    file2.write_text(content)

    formats = ['arrow', 'line', 'markdown', 'table', 'md-table', 'aligned', 'yaml', 'csv']
    for fmt in formats:
        output = tmp_path / f"output.{fmt}"
        multitool.duplicates_mode(
            input_files=[str(file1), str(file2)],
            output_file=str(output),
            min_length=0,
            max_length=10**9,
            output_format=fmt,
            quiet=True
        )
        assert output.exists()
        assert output.stat().st_size > 0

        # Basic content check
        text = output.read_text()
        if fmt == 'csv':
            assert str(file2) in text
            assert str(file1) in text
        elif fmt == 'md-table':
            assert "| Duplicate | Original | Size |" in text
            assert str(file2) in text
        elif fmt == 'line':
            # default separator is ' -> '
            assert f"{file2} -> {file1}" in text

def test_duplicates_mode_limit(tmp_path):
    file1 = tmp_path / "f1.txt"
    file2 = tmp_path / "f2.txt"
    file3 = tmp_path / "f3.txt"
    content = "content"
    file1.write_text(content)
    file2.write_text(content)
    file3.write_text(content)

    output = tmp_path / "output.json"
    multitool.duplicates_mode(
        input_files=[str(file1), str(file2), str(file3)],
        output_file=str(output),
        min_length=0,
        max_length=10**9,
        output_format='json',
        quiet=True,
        limit=1
    )

    with open(output, 'r') as f:
        data = json.load(f)

    assert len(data) == 1

def test_duplicates_mode_inaccessible(tmp_path, capsys):
    file1 = tmp_path / "f1.txt"
    file1.write_text("content")

    # Use a directory as if it were a file
    dir1 = tmp_path / "dir1"
    dir1.mkdir()

    multitool.duplicates_mode(
        input_files=[str(file1), str(dir1), "non_existent.txt"],
        output_file="-",
        min_length=0,
        max_length=10**9,
        output_format='line',
        quiet=True
    )
    # Should not crash and should not find duplicates
    out, err = capsys.readouterr()
    assert out == ""

def test_compute_file_hash_error(monkeypatch, tmp_path):
    # Test error handling in _compute_file_hash
    file1 = tmp_path / "f1.txt"
    file1.write_text("content")

    original_open = open
    def mock_open(path, *args, **kwargs):
        if str(path) == str(file1):
            raise IOError("Permission denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open)
    assert multitool._compute_file_hash(str(file1)) == ""

def test_duplicates_mode_os_error_getsize(tmp_path, monkeypatch, capsys):
    file1 = tmp_path / "f1.txt"
    file1.write_text("content")

    def mock_getsize(path):
        raise OSError("Simulated error")

    monkeypatch.setattr(os.path, "getsize", mock_getsize)

    multitool.duplicates_mode(
        input_files=[str(file1)],
        output_file="-",
        min_length=0,
        max_length=10**9,
        output_format='line',
        quiet=True
    )
    # Should skip the file gracefully
    out, err = capsys.readouterr()
    assert out == ""
