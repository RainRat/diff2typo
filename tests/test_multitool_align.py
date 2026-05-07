import io
from unittest.mock import patch
import pytest
import multitool

def test_align_mode_default_separator(tmp_path):
    """Verify align mode with default separator."""
    input_file = tmp_path / "typos.csv"
    input_file.write_text("teh,the\nabcde,abc\n", encoding="utf-8")

    # Run align mode
    with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
        with patch("sys.argv", ["multitool.py", "align", str(input_file)]):
            multitool.main()
        output = mock_stdout.getvalue()

    # Expected output (aligned)
    # teh   -> the
    # abcde -> abc
    # Max length of left column is 5 ('abcde')
    expected = "teh   -> the\nabcde -> abc\n"
    assert output == expected

def test_align_mode_custom_separator(tmp_path):
    """Verify align mode with custom separator."""
    input_file = tmp_path / "typos.csv"
    input_file.write_text("teh,the\nabcde,abc\n", encoding="utf-8")

    # Run align mode with custom separator
    with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
        with patch("sys.argv", ["multitool.py", "align", str(input_file), "--sep", " | "]):
            multitool.main()
        output = mock_stdout.getvalue()

    # Expected output (aligned with custom separator)
    expected = "teh   | the\nabcde | abc\n"
    assert output == expected

def test_align_mode_with_cleaning(tmp_path):
    """Verify align mode with default cleaning (filter_to_letters)."""
    input_file = tmp_path / "typos.csv"
    # 'teh1' should become 'teh'
    input_file.write_text("teh1,the\n", encoding="utf-8")

    with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
        with patch("sys.argv", ["multitool.py", "align", str(input_file)]):
            multitool.main()
        output = mock_stdout.getvalue()

    expected = "teh -> the\n"
    assert output == expected

def test_align_mode_min_max_length(tmp_path):
    """Verify align mode with min and max length filtering."""
    input_file = tmp_path / "typos.csv"
    input_file.write_text("a,b\nab,cd\nabc,def\nabcde,fghij\n", encoding="utf-8")

    # Test with min-length 3 and max-length 4
    with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
        with patch("sys.argv", ["multitool.py", "align", str(input_file), "--min-length", "3", "--max-length", "4"]):
            multitool.main()
        output = mock_stdout.getvalue()

    # Only 'abc,def' matches (length 3)
    expected = "abc -> def\n"
    assert output == expected

def test_align_mode_process_output(tmp_path):
    """Verify align mode with --process-output (sorting and deduplication)."""
    input_file = tmp_path / "typos.csv"
    # Duplicate and out of order
    input_file.write_text("zzz,aaa\nabc,def\nabc,def\n", encoding="utf-8")

    with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
        with patch("sys.argv", ["multitool.py", "align", str(input_file), "--process-output"]):
            multitool.main()
        output = mock_stdout.getvalue()

    # Should be sorted and unique
    expected = "abc -> def\nzzz -> aaa\n"
    assert output == expected

def test_align_mode_limit(tmp_path):
    """Verify align mode with --limit."""
    input_file = tmp_path / "typos.csv"
    input_file.write_text("a,b\nc,d\ne,f\n", encoding="utf-8")

    with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
        with patch("sys.argv", ["multitool.py", "align", str(input_file), "--limit", "2"]):
            multitool.main()
        output = mock_stdout.getvalue()

    # Should only show first 2 pairs
    expected = "a -> b\nc -> d\n"
    assert output == expected

def test_align_mode_raw(tmp_path):
    """Verify align mode with --raw flag (skipping cleaning)."""
    input_file = tmp_path / "typos.csv"
    # 'teh1' should stay 'teh1' with --raw
    input_file.write_text("teh1,the\n", encoding="utf-8")

    with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
        with patch("sys.argv", ["multitool.py", "align", str(input_file), "--raw"]):
            multitool.main()
        output = mock_stdout.getvalue()

    expected = "teh1 -> the\n"
    assert output == expected

def test_align_mode_empty_after_cleaning(tmp_path):
    """Verify align mode skips items that are empty after cleaning."""
    input_file = tmp_path / "typos.csv"
    # '1' will become '' after cleaning, so it should be skipped
    input_file.write_text("1,the\nabc,def\n", encoding="utf-8")

    with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
        with patch("sys.argv", ["multitool.py", "align", str(input_file)]):
            multitool.main()
        output = mock_stdout.getvalue()

    # '1' is skipped, only 'abc -> def' remains
    expected = "abc -> def\n"
    assert output == expected
