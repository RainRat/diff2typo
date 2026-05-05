import os
import subprocess
import pytest

def test_align_mode_default_separator(tmp_path):
    """Verify align mode with default separator."""
    input_file = tmp_path / "typos.csv"
    input_file.write_text("teh,the\nabcde,abc\n", encoding="utf-8")

    # Run align mode
    result = subprocess.run(
        ["python3", "multitool.py", "align", str(input_file)],
        capture_output=True,
        text=True
    )

    # Expected output (aligned)
    # teh   -> the
    # abcde -> abc
    # Max length of left column is 5 ('abcde')
    expected = "teh   -> the\nabcde -> abc\n"
    assert result.stdout == expected

def test_align_mode_custom_separator(tmp_path):
    """Verify align mode with custom separator."""
    input_file = tmp_path / "typos.csv"
    input_file.write_text("teh,the\nabcde,abc\n", encoding="utf-8")

    # Run align mode with custom separator
    result = subprocess.run(
        ["python3", "multitool.py", "align", str(input_file), "--sep", " | "],
        capture_output=True,
        text=True
    )

    # Expected output (aligned with custom separator)
    expected = "teh   | the\nabcde | abc\n"
    assert result.stdout == expected

def test_align_mode_with_cleaning(tmp_path):
    """Verify align mode with default cleaning (filter_to_letters)."""
    input_file = tmp_path / "typos.csv"
    # 'teh1' should become 'teh'
    input_file.write_text("teh1,the\n", encoding="utf-8")

    result = subprocess.run(
        ["python3", "multitool.py", "align", str(input_file)],
        capture_output=True,
        text=True
    )

    expected = "teh -> the\n"
    assert result.stdout == expected
