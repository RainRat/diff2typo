import pytest
from multitool import unique_mode
import io
import contextlib
import os

def test_unique_mode_order_preservation(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("banana\napple\nbanana\ncherry\napple\n")
    output_file = tmp_path / "output.txt"

    unique_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='line',
        quiet=True,
        clean_items=False
    )

    content = output_file.read_text()
    assert content == "banana\napple\ncherry\n"

def test_unique_mode_with_sort(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("banana\napple\nbanana\ncherry\napple\n")
    output_file = tmp_path / "output.txt"

    unique_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        output_format='line',
        quiet=True,
        clean_items=False
    )

    content = output_file.read_text()
    assert content == "apple\nbanana\ncherry\n"
