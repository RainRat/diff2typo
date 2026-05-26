import pytest
from unittest.mock import patch
import multitool
import random

def create_input_file(filepath):
    content = """alpha
beta
gamma
delta
epsilon
zeta
eta
theta
iota
kappa
"""
    with open(filepath, 'w') as f:
        f.write(content)

def test_shuffle_mode_basic(tmp_path):
    """Test that shuffle mode reorders lines."""
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    create_input_file(input_file)

    # Use a fixed seed for reproducibility in this test
    random.seed(42)

    test_args = [
        "multitool.py", "shuffle", str(input_file),
        "--output", str(output_file)
    ]
    with patch("sys.argv", test_args):
        multitool.main()

    with open(output_file, 'r') as f:
        lines = [line.strip() for line in f]

    assert len(lines) == 10

    with open(input_file, 'r') as f:
        source_lines = [line.strip() for line in f]

    # Verify all original lines are present
    assert sorted(lines) == sorted(source_lines)

    # Verify the order is different (with seed 42, it should be)
    assert lines != source_lines

def test_shuffle_mode_empty_input(tmp_path):
    """Test shuffle mode with empty input."""
    input_file = tmp_path / "empty.txt"
    input_file.write_text("")
    output_file = tmp_path / "output.txt"

    test_args = [
        "multitool.py", "shuffle", str(input_file),
        "--output", str(output_file)
    ]
    with patch("sys.argv", test_args):
        multitool.main()

    with open(output_file, 'r') as f:
        lines = f.readlines()
    assert len(lines) == 0

def test_shuffle_mode_limit(tmp_path):
    """Test shuffle mode with --limit flag."""
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    create_input_file(input_file)

    test_args = [
        "multitool.py", "shuffle", str(input_file),
        "--output", str(output_file),
        "--limit", "5"
    ]
    with patch("sys.argv", test_args):
        multitool.main()

    with open(output_file, 'r') as f:
        lines = f.readlines()

    assert len(lines) == 5

def test_shuffle_mode_min_length(tmp_path):
    """Test shuffle mode with --min-length filter."""
    input_file = tmp_path / "input.txt"
    # Mixed lengths
    content = "a\nbb\nccc\ndddd\n"
    input_file.write_text(content)
    output_file = tmp_path / "output.txt"

    test_args = [
        "multitool.py", "shuffle", str(input_file),
        "--output", str(output_file),
        "--min-length", "3"
    ]
    with patch("sys.argv", test_args):
        multitool.main()

    with open(output_file, 'r') as f:
        lines = [line.strip() for line in f]

    assert len(lines) == 2
    assert "ccc" in lines
    assert "dddd" in lines
    assert "a" not in lines
    assert "bb" not in lines

def test_shuffle_mode_process_output(tmp_path):
    """Test shuffle mode with --process flag (should sort and dedup)."""
    input_file = tmp_path / "input.txt"
    # Create input with duplicates and out of order
    content = "zebra\napple\napple\nbanana\n"
    input_file.write_text(content)
    output_file = tmp_path / "output.txt"

    test_args = [
        "multitool.py", "shuffle", str(input_file),
        "--output", str(output_file),
        "--process"
    ]
    with patch("sys.argv", test_args):
        multitool.main()

    with open(output_file, 'r') as f:
        lines = [line.strip() for line in f]

    # Shuffling followed by process (sort/dedup) should result in alphabetical unique list
    assert lines == ["apple", "banana", "zebra"]
