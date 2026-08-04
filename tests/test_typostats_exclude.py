import sys
import os
import json
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import typostats


def test_is_file_excluded_basic():
    assert typostats._is_file_excluded("foo.json", ["*.json"]) is True
    assert typostats._is_file_excluded("foo.txt", ["*.json"]) is False
    assert typostats._is_file_excluded("dir/foo.json", ["*.json"]) is True
    assert typostats._is_file_excluded("dir/foo.json", ["dir/*"]) is True
    assert typostats._is_file_excluded("", ["*.json"]) is False
    assert typostats._is_file_excluded("foo.json", None) is False
    assert typostats._is_file_excluded("foo.json", []) is False


def test_main_with_single_file_exclusion(tmp_path):
    f1 = tmp_path / "f1.txt"
    f1.write_text("a -> b\n", encoding="utf-8")
    f2 = tmp_path / "f2.txt"
    f2.write_text("c -> d\n", encoding="utf-8")

    with patch('sys.argv', ['typostats.py', str(f1), str(f2), '--exclude', '*f2*']), \
         patch('typostats.generate_report') as mock_report:
        typostats.main()
        assert mock_report.call_args[1]['total_pairs'] == 1


def test_main_with_recursive_directory_exclusion(tmp_path):
    sub1 = tmp_path / "subdir1"
    sub1.mkdir()
    f1 = sub1 / "file1.txt"
    f1.write_text("a -> b\n", encoding="utf-8")

    sub2 = tmp_path / "subdir2"
    sub2.mkdir()
    f2 = sub2 / "file2.txt"
    f2.write_text("c -> d\n", encoding="utf-8")

    with patch('sys.argv', ['typostats.py', str(tmp_path), '--exclude', '*/subdir2/*']), \
         patch('typostats.generate_report') as mock_report:
        typostats.main()
        assert mock_report.call_args[1]['total_pairs'] == 1


def test_main_with_extension_exclusion(tmp_path):
    f1 = tmp_path / "f1.txt"
    f1.write_text("a -> b\n", encoding="utf-8")
    f2 = tmp_path / "f2.csv"
    f2.write_text("c,d\n", encoding="utf-8")

    with patch('sys.argv', ['typostats.py', str(f1), str(f2), '-e', '*.csv']), \
         patch('typostats.generate_report') as mock_report:
        typostats.main()
        assert mock_report.call_args[1]['total_pairs'] == 1
