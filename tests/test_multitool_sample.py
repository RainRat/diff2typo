import pytest
import subprocess
import os
import sys

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
    subprocess.run([
        sys.executable, "multitool.py", "sample", str(input_file),
        "--output", str(output_file),
        "--n", "3"
    ], check=True)

    with open(output_file, 'r') as f:
        lines = f.readlines()

    assert len(lines) == 3

    # Verify lines are from the source (cleaned versions)
    # alpha -> alpha (length 5, min is 3)
    # The clean_and_filter reduces 'alpha\n' to 'alpha'.
    # The output file has 'alpha\n'.
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
    subprocess.run([
        sys.executable, "multitool.py", "sample", str(input_file),
        "--output", str(output_file),
        "--percent", "50"
    ], check=True)

    with open(output_file, 'r') as f:
        lines = f.readlines()

    assert len(lines) == 5

def test_sample_mode_invalid_args(tmp_path):
    """Test that specifying both n and percent or neither fails."""
    input_file = tmp_path / "input.txt"
    create_input_file(input_file)

    # Neither
    result = subprocess.run([
        sys.executable, "multitool.py", "sample", str(input_file)
    ], capture_output=True, text=True)
    assert result.returncode != 0

    # Both
    result = subprocess.run([
        sys.executable, "multitool.py", "sample", str(input_file), "--n", "2", "--percent", "50"
    ], capture_output=True, text=True)
    assert result.returncode != 0
