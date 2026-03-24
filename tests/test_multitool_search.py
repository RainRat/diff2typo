import pytest
import sys
import os
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from multitool import search_mode
import io
import os
import logging

def test_search_mode_basic(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello world\nthis is a test\nanother line")

    output_file = tmp_path / "output.txt"

    # Search for "hello"
    search_mode([str(input_file)], "hello", str(output_file), 1, 100, False)

    assert output_file.read_text().strip() == "hello world"

def test_search_mode_fuzzy(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello world\nhelo world\nhillo world")

    output_file = tmp_path / "output.txt"

    # Search for "hello" with max_dist=1
    search_mode([str(input_file)], "hello", str(output_file), 1, 100, False, max_dist=1)

    results = output_file.read_text().strip().split('\n')
    assert "hello world" in results
    assert "helo world" in results
    assert "hillo world" in results

def test_search_mode_smart(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("myCamelCaseVar\nmy_snake_case_var\nnormal")

    output_file = tmp_path / "output.txt"

    # Search for "camel" with smart=True
    search_mode([str(input_file)], "camel", str(output_file), 1, 100, False, smart=True)
    assert output_file.read_text().strip() == "myCamelCaseVar"

    # Search for "snake" with smart=True
    search_mode([str(input_file)], "snake", str(output_file), 1, 100, False, smart=True)
    assert output_file.read_text().strip() == "my_snake_case_var"

def test_search_mode_line_numbers(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("first\nsecond\nthird")

    output_file = tmp_path / "output.txt"

    # Search for "second" with line_numbers=True
    search_mode([str(input_file)], "second", str(output_file), 1, 100, False, line_numbers=True)

    # We expect color codes if in a terminal, but here we just check for the number
    content = output_file.read_text()
    # Strip potential ANSI codes for comparison
    clean_content = re.sub(r'\x1b\[[0-9;]*m', '', content)
    assert "2:second" in clean_content

def test_search_mode_multiple_files(tmp_path):
    f1 = tmp_path / "f1.txt"
    f1.write_text("match in f1")
    f2 = tmp_path / "f2.txt"
    f2.write_text("match in f2")

    output_file = tmp_path / "output.txt"

    search_mode([str(f1), str(f2)], "match", str(output_file), 1, 100, False)

    content = output_file.read_text()
    # Strip potential ANSI codes for comparison
    clean_content = re.sub(r'\x1b\[[0-9;]*m', '', content)
    assert "f1.txt:match in f1" in clean_content
    assert "f2.txt:match in f2" in clean_content

def test_search_mode_mapping_file(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("one\ntwo\nthree")

    mapping_file = tmp_path / "map.csv"
    mapping_file.write_text("one,1\ntwo,2")

    output_file = tmp_path / "output.txt"

    search_mode([str(input_file)], str(mapping_file), str(output_file), 1, 100, False)

    results = output_file.read_text().strip().split('\n')
    assert "one" in results
    assert "two" in results
    assert "three" not in results
