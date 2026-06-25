import os
import pytest
from multitool import main
from unittest.mock import patch
import sys

def test_rename_regex_basic(tmp_path):
    # Setup files
    file1 = tmp_path / "data_2023.txt"
    file2 = tmp_path / "data_2024.txt"
    file1.write_text("1")
    file2.write_text("2")

    # Rename data_(\d+) to archive_\1
    args = [
        "multitool.py", "rename", str(tmp_path),
        "--regex",
        "--add", r"data_(\d+):archive_\1",
        "--in-place"
    ]

    with patch.object(sys, 'argv', args):
        main()

    assert (tmp_path / "archive_2023.txt").exists()
    assert (tmp_path / "archive_2024.txt").exists()
    assert not (tmp_path / "data_2023.txt").exists()
    assert not (tmp_path / "data_2024.txt").exists()

def test_rename_regex_groups_swap(tmp_path):
    # Setup files
    file1 = tmp_path / "report_jan.txt"
    file1.write_text("report")

    # Swap report and month: (\w+)_(\w+) to \2_\1
    args = [
        "multitool.py", "rename", str(file1),
        "--regex",
        "--add", r"(\w+)_(\w+):\2_\1",
        "--in-place"
    ]

    with patch.object(sys, 'argv', args):
        main()

    assert (tmp_path / "jan_report.txt").exists()
    assert not (tmp_path / "report_jan.txt").exists()

def test_rename_regex_smart_case(tmp_path):
    # Setup files
    file1 = tmp_path / "TehData.txt"
    file2 = tmp_path / "teh_data.txt"
    file3 = tmp_path / "TEH_LOUD.txt"
    for f in [file1, file2, file3]:
        f.write_text("content")

    # Rename teh to the with smart case, using inline ignore case to ensure matches
    args = [
        "multitool.py", "rename", str(tmp_path),
        "--regex",
        "--add", r"(?i)teh:the",
        "--smart-case",
        "--in-place"
    ]

    with patch.object(sys, 'argv', args):
        main()

    assert (tmp_path / "TheData.txt").exists()
    assert (tmp_path / "the_data.txt").exists()
    assert (tmp_path / "THE_LOUD.txt").exists()
    assert not file1.exists()
    assert not file2.exists()
    assert not file3.exists()

def test_rename_regex_invalid_pattern(tmp_path, caplog):
    file1 = tmp_path / "file.txt"
    file1.write_text("1")

    args = [
        "multitool.py", "rename", str(file1),
        "--regex",
        "--add", r"[invalid:target",
        "--in-place"
    ]

    with patch.object(sys, 'argv', args):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1

    assert "Invalid regular expression pattern" in caplog.text

def test_rename_regex_no_smart_case(tmp_path):
    # Setup files
    file1 = tmp_path / "TehData.txt"
    file1.write_text("1")

    # Rename teh to the WITHOUT smart case, using inline ignore case
    args = [
        "multitool.py", "rename", str(file1),
        "--regex",
        "--add", r"(?i)teh:the",
        "--in-place"
    ]

    with patch.object(sys, 'argv', args):
        main()

    # Should be "theData.txt" because replacement is "the" and smart-case is off
    assert (tmp_path / "theData.txt").exists()
    assert not file1.exists()
