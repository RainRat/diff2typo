import os
import subprocess
import json
import csv
import pytest

def test_fileinfo_mode_visual(tmp_path):
    # Create a test file
    test_file = tmp_path / "test.txt"
    content = "Hello world\nThis is a test file.\nThird line."
    test_file.write_text(content, encoding='utf-8')

    # Run multitool.py fileinfo
    result = subprocess.run(
        ['python3', 'multitool.py', 'fileinfo', str(test_file), '-f', 'arrow'],
        capture_output=True, text=True, check=True
    )

    # Assertions
    output = result.stdout
    assert "test.txt" in output
    assert "Lines" in output
    assert "Words" in output
    assert "Size" in output
    assert "3" in output # Lines
    assert "9" in output # Words (Hello, world, This, is, a, test, file, Third, line) - wait, splitting by line.split()
    # "Hello world" -> 2
    # "This is a test file." -> 5
    # "Third line." -> 2
    # Total = 9

def test_fileinfo_mode_json(tmp_path):
    test_file = tmp_path / "test.txt"
    content = "One two three"
    test_file.write_text(content, encoding='utf-8')

    result = subprocess.run(
        ['python3', 'multitool.py', 'fileinfo', str(test_file), '-f', 'json'],
        capture_output=True, text=True, check=True
    )

    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]['file'] == str(test_file)
    assert data[0]['lines'] == 1
    assert data[0]['words'] == 3
    assert data[0]['size'] == len(content)

def test_fileinfo_mode_csv(tmp_path):
    test_file = tmp_path / "test.txt"
    content = "Line 1\nLine 2"
    test_file.write_text(content, encoding='utf-8')

    csv_file = tmp_path / "output.csv"
    subprocess.run(
        ['python3', 'multitool.py', 'fileinfo', str(test_file), '-f', 'csv', '-o', str(csv_file)],
        check=True
    )

    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]['file'] == str(test_file)
        assert int(rows[0]['lines']) == 2
        assert int(rows[0]['words']) == 4
