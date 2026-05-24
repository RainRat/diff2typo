import pytest
from multitool import _get_total_line_count
from unittest.mock import patch

def test_get_total_line_count_nonexistent_file():
    # Attempting to count lines in a non-existent file should return 0 and not raise OSError
    assert _get_total_line_count(["nonexistent_file.txt"]) == 0

def test_get_total_line_count_oserror_mock():
    # Mocking open to raise OSError should return 0
    with patch("builtins.open", side_effect=OSError("Access Denied")):
        assert _get_total_line_count(["some_file.txt"]) == 0

def test_get_total_line_count_mixed_files(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("line1\nline2\n")
    # One valid file, one non-existent
    assert _get_total_line_count([str(f), "nonexistent.txt"]) == 2
