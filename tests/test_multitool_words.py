import pytest
from multitool import words_mode
import io
import contextlib
import sys

def test_words_mode_basic(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello world\nthis is a test")
    output_file = tmp_path / "output.txt"

    words_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        clean_items=False
    )

    assert output_file.read_text().splitlines() == ["hello", "world", "this", "is", "a", "test"]

def test_words_mode_delimiter(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple,banana,cherry\ndate,elderberry")
    output_file = tmp_path / "output.txt"

    words_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        delimiter=",",
        clean_items=False
    )

    assert output_file.read_text().splitlines() == ["apple", "banana", "cherry", "date", "elderberry"]

def test_words_mode_cleaning_and_filtering(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("Hello World! 123\nShort a")
    output_file = tmp_path / "output.txt"

    words_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=True,
        clean_items=True
    )

    # "Hello" -> "hello", "World!" -> "world", "123" -> filtered out by clean_items (only a-z)
    # "Short" -> "short", "a" -> too short (min_length=3)
    # process_output=True means sorted and unique
    assert output_file.read_text().splitlines() == ["hello", "short", "world"]
