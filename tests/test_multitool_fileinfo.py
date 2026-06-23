import os
import json
import csv
import io
from unittest.mock import patch, MagicMock, contextlib
import pytest
from multitool import fileinfo_mode, main

# Helper to mock smart_open_output and prevent closing the StringIO object
class MockSmartOpen:
    def __init__(self, stream):
        self.stream = stream
    def __enter__(self):
        return self.stream
    def __exit__(self, *args):
        pass

def test_fileinfo_mode_basic(tmp_path):
    f1 = tmp_path / "test1.txt"
    f1.write_text("hello world\nsecond line", encoding="utf-8")

    output = io.StringIO()
    with patch('multitool.smart_open_output', return_value=MockSmartOpen(output)):
        fileinfo_mode([str(f1)], "-", output_format='json', quiet=True)

    data = json.loads(output.getvalue())
    assert len(data) == 1
    assert data[0]['file'] == str(f1)
    assert data[0]['lines'] == 2
    assert data[0]['words'] == 4
    assert data[0]['size'] == os.path.getsize(f1)

def test_fileinfo_mode_multiple_files(tmp_path):
    f1 = tmp_path / "test1.txt"
    f1.write_text("one two", encoding="utf-8")
    f2 = tmp_path / "test2.txt"
    f2.write_text("three\nfour\nfive", encoding="utf-8")

    output = io.StringIO()
    with patch('multitool.smart_open_output', return_value=MockSmartOpen(output)):
        fileinfo_mode([str(f1), str(f2)], "-", output_format='json', quiet=True)

    data = json.loads(output.getvalue())
    assert len(data) == 2
    assert data[0]['words'] == 2
    assert data[1]['lines'] == 3

def test_fileinfo_mode_arrow_format(tmp_path):
    f1 = tmp_path / "test1.txt"
    f1.write_text("hello world", encoding="utf-8")

    output = io.StringIO()
    with patch('multitool.smart_open_output', return_value=MockSmartOpen(output)), \
         patch('multitool._should_enable_color', return_value=False):
        fileinfo_mode([str(f1)], "-", output_format='arrow', quiet=True)

    content = output.getvalue()
    assert "File" in content
    assert "Size" in content
    assert "Lines" in content
    assert "Words" in content
    assert "Encoding" in content
    assert str(f1) in content

def test_fileinfo_mode_csv_format(tmp_path):
    f1 = tmp_path / "test1.txt"
    f1.write_text("hello world", encoding="utf-8")

    output = io.StringIO()
    with patch('multitool.smart_open_output', return_value=MockSmartOpen(output)):
        fileinfo_mode([str(f1)], "-", output_format='csv', quiet=True)

    content = output.getvalue()
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]['file'] == str(f1)
    assert int(rows[0]['lines']) == 1
    assert int(rows[0]['words']) == 2

def test_fileinfo_mode_limit(tmp_path):
    f1 = tmp_path / "test1.txt"
    f1.write_text("one", encoding="utf-8")
    f2 = tmp_path / "test2.txt"
    f2.write_text("two", encoding="utf-8")

    output = io.StringIO()
    with patch('multitool.smart_open_output', return_value=MockSmartOpen(output)):
        fileinfo_mode([str(f1), str(f2)], "-", output_format='json', quiet=True, limit=1)

    data = json.loads(output.getvalue())
    assert len(data) == 1
    assert data[0]['file'] == str(f1)

def test_fileinfo_cli_interactive(tmp_path):
    f1 = tmp_path / "test1.txt"
    f1.write_text("hello", encoding="utf-8")

    with patch('sys.argv', ['multitool.py', 'fileinfo', str(f1)]), \
         patch('sys.stdout.isatty', return_value=True), \
         patch('multitool.fileinfo_mode') as mock_mode:
        main()

        args, kwargs = mock_mode.call_args
        assert kwargs['output_format'] == 'arrow'

def test_fileinfo_cli_piped(tmp_path):
    f1 = tmp_path / "test1.txt"
    f1.write_text("hello", encoding="utf-8")

    with patch('sys.argv', ['multitool.py', 'fileinfo', str(f1)]), \
         patch('sys.stdout.isatty', return_value=False), \
         patch('multitool.fileinfo_mode') as mock_mode:
        main()

        args, kwargs = mock_mode.call_args
        assert kwargs['output_format'] == 'line'
