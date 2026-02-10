import pytest
import os
import sys
from unittest.mock import patch
import multitool

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

def test_sample_mode_n(tmp_path):
    """Test sampling a specific number of lines."""
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    create_input_file(input_file)

    # Run multitool with sample mode requesting 3 lines
    test_args = [
        "multitool.py", "sample", str(input_file),
        "--output", str(output_file),
        "--n", "3"
    ]
    with patch("sys.argv", test_args):
        multitool.main()

    with open(output_file, 'r') as f:
        lines = f.readlines()

    assert len(lines) == 3

    # Verify lines are from the source (cleaned versions)
    with open(input_file, 'r') as f:
        source_lines = set(l.strip() for l in f)

    for line in lines:
        assert line.strip() in source_lines

def test_sample_mode_percent(tmp_path):
    """Test sampling a percentage of lines."""
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    create_input_file(input_file)

    # Run multitool with sample mode requesting 50% lines (5 lines)
    test_args = [
        "multitool.py", "sample", str(input_file),
        "--output", str(output_file),
        "--percent", "50"
    ]
    with patch("sys.argv", test_args):
        multitool.main()

    with open(output_file, 'r') as f:
        lines = f.readlines()

    assert len(lines) == 5

def test_sample_mode_invalid_args(tmp_path):
    """Test that specifying both n and percent or neither fails."""
    input_file = tmp_path / "input.txt"
    create_input_file(input_file)

    # Neither - argparse will raise SystemExit because mode requires one of them in the group
    test_args_neither = ["multitool.py", "sample", str(input_file)]
    with patch("sys.argv", test_args_neither):
        with pytest.raises(SystemExit):
            multitool.main()

    # Both - argparse will raise SystemExit due to mutual exclusivity
    test_args_both = [
        "multitool.py", "sample", str(input_file),
        "--n", "2", "--percent", "50"
    ]
    with patch("sys.argv", test_args_both):
        with pytest.raises(SystemExit):
            multitool.main()

def test_sample_mode_empty_input(tmp_path):
    """Test sampling from an empty file."""
    input_file = tmp_path / "empty.txt"
    input_file.write_text("")
    output_file = tmp_path / "output.txt"

    test_args = [
        "multitool.py", "sample", str(input_file),
        "--output", str(output_file),
        "--n", "5"
    ]
    with patch("sys.argv", test_args):
        multitool.main()

    with open(output_file, 'r') as f:
        lines = f.readlines()
    assert len(lines) == 0

def test_sample_mode_n_too_large(tmp_path):
    """Test sampling more lines than available."""
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    create_input_file(input_file) # 10 lines

    test_args = [
        "multitool.py", "sample", str(input_file),
        "--output", str(output_file),
        "--n", "20"
    ]
    with patch("sys.argv", test_args):
        multitool.main()

    with open(output_file, 'r') as f:
        lines = f.readlines()
    assert len(lines) == 10

def test_sample_mode_percent_edge_cases(tmp_path):
    """Test 0% and 100% sampling."""
    input_file = tmp_path / "input.txt"
    create_input_file(input_file) # 10 lines

    # 0%
    output_file_0 = tmp_path / "output_0.txt"
    test_args_0 = [
        "multitool.py", "sample", str(input_file),
        "--output", str(output_file_0),
        "--percent", "0"
    ]
    with patch("sys.argv", test_args_0):
        multitool.main()
    with open(output_file_0, 'r') as f:
        assert len(f.readlines()) == 0

    # 100%
    output_file_100 = tmp_path / "output_100.txt"
    test_args_100 = [
        "multitool.py", "sample", str(input_file),
        "--output", str(output_file_100),
        "--percent", "100"
    ]
    with patch("sys.argv", test_args_100):
        multitool.main()
    with open(output_file_100, 'r') as f:
        assert len(f.readlines()) == 10

def test_sample_mode_raw(tmp_path):
    """Test sampling with --raw flag (clean_items=False)."""
    input_file = tmp_path / "input.txt"
    # Create input with some trailing spaces that would be cleaned normally
    content = "line1   \nline2   \nline3   \n"
    input_file.write_text(content)
    output_file = tmp_path / "output.txt"

    test_args = [
        "multitool.py", "sample", str(input_file),
        "--output", str(output_file),
        "--n", "2",
        "--raw"
    ]
    with patch("sys.argv", test_args):
        multitool.main()

    with open(output_file, 'r') as f:
        lines = f.readlines()

    assert len(lines) == 2
    for line in lines:
        # If raw, it should keep the spaces before the newline
        # Actually, multitool line items are stripped of \n but maybe not trailing spaces if raw?
        # Let's check multitool.py _extract_line_items
        assert line.endswith("   \n")

def test_sample_mode_process_output(tmp_path):
    """Test sampling with --process flag."""
    input_file = tmp_path / "input.txt"
    # Create input with duplicates
    content = "apple\napple\norange\norange\n"
    input_file.write_text(content)
    output_file = tmp_path / "output.txt"

    # Sample 100% with process (which should unique and sort)
    test_args = [
        "multitool.py", "sample", str(input_file),
        "--output", str(output_file),
        "--percent", "100",
        "--process"
    ]
    with patch("sys.argv", test_args):
        multitool.main()

    with open(output_file, 'r') as f:
        lines = f.readlines()

    assert len(lines) == 2
    assert lines == ["apple\n", "orange\n"]
