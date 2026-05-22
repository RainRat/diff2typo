import os
import pytest
from multitool import sort_mode

def test_sort_mode_alpha(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("banana\napple\ncherry\n")
    output_file = tmp_path / "output.txt"

    sort_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        by='alpha',
        reverse=False,
        unique=False,
        output_format='line',
        quiet=True,
        clean_items=False
    )

    assert output_file.read_text().splitlines() == ["apple", "banana", "cherry"]

def test_sort_mode_length(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a\nbbb\ncc\n")
    output_file = tmp_path / "output.txt"

    sort_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        by='length',
        reverse=False,
        unique=False,
        output_format='line',
        quiet=True,
        clean_items=False
    )

    assert output_file.read_text().splitlines() == ["a", "cc", "bbb"]

def test_sort_mode_numeric(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("10\n2\n20\n1\n")
    output_file = tmp_path / "output.txt"

    sort_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        by='numeric',
        reverse=False,
        unique=False,
        output_format='line',
        quiet=True,
        clean_items=False
    )

    assert output_file.read_text().splitlines() == ["1", "2", "10", "20"]

def test_sort_mode_reverse_unique(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple\nbanana\napple\ncherry\n")
    output_file = tmp_path / "output.txt"

    sort_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        by='alpha',
        reverse=True,
        unique=True,
        output_format='line',
        quiet=True,
        clean_items=False
    )

    assert output_file.read_text().splitlines() == ["cherry", "banana", "apple"]
