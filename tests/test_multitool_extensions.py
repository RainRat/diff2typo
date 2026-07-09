import os
import json
import csv
import sys
from io import StringIO
import pytest
from unittest.mock import patch
import multitool
from multitool import main

def setup_module(module):
    # Ensure standard input cache is reset before tests
    multitool._STDIN_CACHE = None

@pytest.fixture
def temp_files(tmp_path):
    # Create some test files with different extensions and sizes
    d = tmp_path / "test_dir"
    d.mkdir()

    f1 = d / "file1.txt"
    f1.write_text("Hello world") # 11 bytes

    f2 = d / "file2.txt"
    f2.write_text("Test file") # 9 bytes

    f3 = d / "script.py"
    f3.write_text("print('test')") # 13 bytes

    f4 = d / "no_ext"
    f4.write_text("Some data") # 9 bytes

    return d

def test_extensions_json_output(temp_files, capsys):
    test_args = ["multitool.py", "extensions", str(temp_files), "-f", "json"]
    with patch.object(sys, 'argv', test_args):
        main()

    captured = capsys.readouterr()
    output = captured.out

    data = json.loads(output)
    # 3 unique extensions: .py, .txt, (no extension)
    assert len(data) == 3

    # Sort by extension for consistent testing
    data.sort(key=lambda x: x["extension"])

    # (no extension) comes before .py because '(' < '.'
    assert data[0]["extension"] == "(no extension)"
    assert data[0]["count"] == 1
    assert data[0]["size"] == 9

    assert data[1]["extension"] == ".py"
    assert data[1]["count"] == 1
    assert data[1]["size"] == 13

    assert data[2]["extension"] == ".txt"
    assert data[2]["count"] == 2
    assert data[2]["size"] == 20

def test_extensions_csv_output(temp_files, capsys):
    test_args = ["multitool.py", "extensions", str(temp_files), "-f", "csv"]
    with patch.object(sys, 'argv', test_args):
        main()

    captured = capsys.readouterr()
    output = captured.out

    reader = csv.DictReader(StringIO(output))
    rows = list(reader)

    assert len(rows) == 3

    # Check for presence of expected extensions
    exts = {row["extension"] for row in rows}
    assert ".py" in exts
    assert ".txt" in exts
    assert "(no extension)" in exts

def test_extensions_arrow_output(temp_files, capsys):
    test_args = ["multitool.py", "extensions", str(temp_files), "-f", "arrow"]
    with patch.object(sys, 'argv', test_args):
        main()

    captured = capsys.readouterr()
    output = captured.out

    assert "Extension" in output
    assert "Files" in output
    assert "Size" in output
    assert ".py" in output
    assert ".txt" in output
    assert "(no extension)" in output
    assert "EXTENSIONS ANALYSIS SUMMARY" in output
    assert "Total files analyzed:               4" in output

def test_extensions_limit(temp_files, capsys):
    test_args = ["multitool.py", "extensions", str(temp_files), "-f", "json", "-L", "1"]
    with patch.object(sys, 'argv', test_args):
        main()

    captured = capsys.readouterr()
    output = captured.out

    data = json.loads(output)
    assert len(data) == 1
    # The largest should be .txt (20 bytes)
    assert data[0]["extension"] == ".txt"

def test_extensions_no_files(tmp_path, capsys):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    test_args = ["multitool.py", "extensions", str(empty_dir), "-f", "json"]
    with patch.object(sys, 'argv', test_args):
        main()

    captured = capsys.readouterr()
    # Structured data helper outputs [] for empty list
    assert captured.out.strip() == "[]"

def test_extensions_help(capsys):
    test_args = ["multitool.py", "help", "extensions"]
    with pytest.raises(SystemExit):
        with patch.object(sys, 'argv', test_args):
            main()

    captured = capsys.readouterr()
    # show_mode_help outputs to stderr via parser.exit
    help_text = captured.err
    assert "Analyzes disk usage by extension" in help_text
    assert "USAGE:" in help_text
    assert "python multitool.py extensions [FILES...] [OPTIONS]" in help_text
