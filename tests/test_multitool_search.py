import sys
import os
import pytest
import io
import re
from unittest.mock import patch

# Add the repository root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multitool import search_mode

def strip_ansi(text):
    """Remove ANSI escape sequences from a string."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def test_search_exact_match(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("Hello world\nThis is a test\nSearching for something", encoding='utf-8')

    output_file = tmp_path / "output.txt"

    search_mode(
        input_files=[str(input_file)],
        query="test",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    # Strip ANSI colors if present
    results = [strip_ansi(r) for r in results]
    assert len(results) == 1
    assert "This is a test" in results[0]

def test_search_fuzzy_match(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("Hello world\nThhe quick brown fox\nThis is a test", encoding='utf-8')

    output_file = tmp_path / "output.txt"

    # Search for 'the' with max-dist 1 should find 'Thhe'
    search_mode(
        input_files=[str(input_file)],
        query="the",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        max_dist=1,
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    results = [strip_ansi(r) for r in results]
    assert any("Thhe quick brown fox" in r for r in results)
    assert len(results) == 1

def test_search_smart_match(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("myVariableName\nother_variable\nnormal words", encoding='utf-8')

    output_file = tmp_path / "output.txt"

    # Search for 'variable' with smart mode should find 'myVariableName' and 'other_variable'
    search_mode(
        input_files=[str(input_file)],
        query="variable",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        smart=True,
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    results = [strip_ansi(r) for r in results]
    assert len(results) == 2
    assert any("myVariableName" in r for r in results)
    assert any("other_variable" in r for r in results)

def test_search_line_numbers(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("Line one\nLine two\nThird line", encoding='utf-8')

    output_file = tmp_path / "output.txt"

    search_mode(
        input_files=[str(input_file)],
        query="two",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        line_numbers=True,
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    results = [strip_ansi(r) for r in results]
    assert len(results) == 1
    # Standard grep behavior: no filename for single file unless forced
    assert "2: Line two" in results[0]

def test_search_limit(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("match 1\nmatch 2\nmatch 3", encoding='utf-8')

    output_file = tmp_path / "output.txt"

    search_mode(
        input_files=[str(input_file)],
        query="match",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        limit=2,
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    assert len(results) == 2

def test_search_no_matches(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("Hello world", encoding='utf-8')

    output_file = tmp_path / "output.txt"

    search_mode(
        input_files=[str(input_file)],
        query="nonexistent",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    assert len(results) == 0

def test_search_length_filter(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("a short word and alongword", encoding='utf-8')

    output_file = tmp_path / "output.txt"

    # Search for 'word' with min-length 5 should miss 'a' but find 'alongword' if it matches
    # Actually, the logic applies length filter to the CANDIDATE words in the file.

    # If we search for 'long' in 'alongword' with smart mode and min-length 9
    search_mode(
        input_files=[str(input_file)],
        query="long",
        output_file=str(output_file),
        min_length=9,
        max_length=100,
        process_output=False,
        smart=True
    )
    results = output_file.read_text(encoding='utf-8').splitlines()
    assert len(results) == 1
    assert "alongword" in results[0]

    # Now with min-length 20, it should find nothing
    search_mode(
        input_files=[str(input_file)],
        query="long",
        output_file=str(output_file),
        min_length=20,
        max_length=100,
        process_output=False,
        smart=True
    )
    results = output_file.read_text(encoding='utf-8').splitlines()
    assert len(results) == 0
