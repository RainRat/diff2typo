import sys
import os
import re
import pytest

# Add the repository root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multitool import search_mode, scan_mode

def strip_ansi(text):
    """Remove ANSI escape sequences from a string."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

@pytest.fixture
def sample_file(tmp_path):
    f = tmp_path / "sample.txt"
    content = "\n".join([
        "line 1",
        "line 2",
        "line 3 (MATCH)",
        "line 4",
        "line 5",
        "line 6",
        "line 7 (MATCH)",
        "line 8",
        "line 9",
        "line 10"
    ])
    f.write_text(content, encoding='utf-8')
    return f

def test_search_context_after(sample_file, tmp_path):
    output = tmp_path / "output.txt"
    search_mode(
        input_files=[str(sample_file)],
        query="MATCH",
        output_file=str(output),
        min_length=1,
        max_length=100,
        process_output=False,
        after_context=2,
        line_numbers=True,
        with_filename=True
    )

    results = output.read_text(encoding='utf-8').splitlines()
    # Expectation for first match (idx 2): line 3, 4, 5
    # Expectation for second match (idx 6): line 7, 8, 9
    # Non-contiguous, so separator "--" expected

    clean_results = [strip_ansi(r) for r in results]

    # Prefix format: filename:line: for match, filename-line- for context
    fname = str(sample_file)
    expected = [
        f"{fname}:3: line 3 (MATCH)",
        f"{fname}-4- line 4",
        f"{fname}-5- line 5",
        "--",
        f"{fname}:7: line 7 (MATCH)",
        f"{fname}-8- line 8",
        f"{fname}-9- line 9"
    ]

    assert clean_results == expected

def test_search_context_before(sample_file, tmp_path):
    output = tmp_path / "output.txt"
    search_mode(
        input_files=[str(sample_file)],
        query="MATCH",
        output_file=str(output),
        min_length=1,
        max_length=100,
        process_output=False,
        before_context=1,
        line_numbers=True,
        with_filename=True
    )

    results = output.read_text(encoding='utf-8').splitlines()
    clean_results = [strip_ansi(r) for r in results]

    fname = str(sample_file)
    expected = [
        f"{fname}-2- line 2",
        f"{fname}:3: line 3 (MATCH)",
        "--",
        f"{fname}-6- line 6",
        f"{fname}:7: line 7 (MATCH)"
    ]

    assert clean_results == expected

def test_search_context_both(sample_file, tmp_path):
    output = tmp_path / "output.txt"
    # Merge context overlap: match at 2 and 6.
    # With -C 2:
    #   Block 1: 0, 1, 2, 3, 4
    #   Block 2: 4, 5, 6, 7, 8
    # They touch at index 4, so they should be merged without separator.
    search_mode(
        input_files=[str(sample_file)],
        query="MATCH",
        output_file=str(output),
        min_length=1,
        max_length=100,
        process_output=False,
        before_context=2,
        after_context=2,
        line_numbers=True,
        with_filename=True
    )

    results = output.read_text(encoding='utf-8').splitlines()
    clean_results = [strip_ansi(r) for r in results]

    assert "--" not in clean_results
    assert len(clean_results) == 9 # indices 0 through 8
    fname = str(sample_file)
    assert f"{fname}:3: line 3 (MATCH)" == clean_results[2]
    assert f"{fname}:7: line 7 (MATCH)" == clean_results[6]

def test_scan_context(sample_file, tmp_path):
    output = tmp_path / "output.txt"
    scan_mode(
        input_files=[str(sample_file)],
        mapping_file=None,
        ad_hoc=["MATCH:FIX"],
        output_file=str(output),
        min_length=1,
        max_length=100,
        process_output=False,
        after_context=1,
        line_numbers=True,
        with_filename=True
    )

    results = output.read_text(encoding='utf-8').splitlines()
    clean_results = [strip_ansi(r) for r in results]

    fname = str(sample_file)
    expected = [
        f"{fname}:3: line 3 (MATCH)",
        f"{fname}-4- line 4",
        "--",
        f"{fname}:7: line 7 (MATCH)",
        f"{fname}-8- line 8"
    ]

    assert clean_results == expected
