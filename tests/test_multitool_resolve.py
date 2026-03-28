import pytest
from multitool import resolve_mode
import io
import os

def test_resolve_mode_basic(tmp_path):
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    input_file.write_text("a -> b\nb -> c\nx -> y\ny -> z\nz -> w\np -> q\n")

    resolve_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        output_format='line',
        quiet=True,
        clean_items=True
    )

    expected = "a -> c\nb -> c\np -> q\nx -> w\ny -> w\nz -> w\n"
    assert output_file.read_text() == expected

def test_resolve_mode_cycle(tmp_path):
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    # m -> n -> m is a cycle. Terminal value detection should stop.
    input_file.write_text("m -> n\nn -> m\na -> b\nb -> c\n")

    resolve_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        output_format='line',
        quiet=True,
        clean_items=True
    )

    # m -> n, n -> m
    # a -> c, b -> c
    expected = "a -> c\nb -> c\nm -> n\nn -> m\n"
    assert output_file.read_text() == expected

def test_resolve_mode_empty_input(tmp_path):
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    input_file.write_text("")

    resolve_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        output_format='line',
        quiet=True,
        clean_items=True
    )

    assert output_file.read_text() == ""

def test_resolve_mode_long_chain(tmp_path):
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    # Using letters only to avoid issues with default cleaning
    import string
    letters = string.ascii_lowercase
    content = "\n".join([f"{letters[i]} -> {letters[i+1]}" for i in range(10)])
    input_file.write_text(content)

    resolve_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        output_format='line',
        quiet=True,
        clean_items=True
    )

    output = output_file.read_text().splitlines()
    # 'a' should resolve to 'k' (index 0 to index 10)
    assert "a -> k" in output
    # 'j' should resolve to 'k' (index 9 to index 10)
    assert "j -> k" in output
    assert len(output) == 10
