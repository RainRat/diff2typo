import sys
import os
import re

# Add the repository root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multitool
from multitool import search_mode, scan_mode

def strip_ansi(text):
    """Remove ANSI escape sequences from a string."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def test_search_context_before(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("line 1\nline 2\nline 3\nMATCH\nline 5", encoding='utf-8')
    output_file = tmp_path / "output.txt"

    search_mode(
        input_files=[str(input_file)],
        query="MATCH",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        before_context=2
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    assert len(results) == 3
    assert results[0] == "line 2"
    assert results[1] == "line 3"
    assert results[2] == "MATCH"

def test_search_context_after(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("line 1\nMATCH\nline 3\nline 4\nline 5", encoding='utf-8')
    output_file = tmp_path / "output.txt"

    search_mode(
        input_files=[str(input_file)],
        query="MATCH",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        after_context=2
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    assert len(results) == 3
    assert results[0] == "MATCH"
    assert results[1] == "line 3"
    assert results[2] == "line 4"

def test_search_context_both(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("line 1\nline 2\nMATCH\nline 4\nline 5", encoding='utf-8')
    output_file = tmp_path / "output.txt"

    search_mode(
        input_files=[str(input_file)],
        query="MATCH",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        before_context=1,
        after_context=1
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    assert len(results) == 3
    assert results[0] == "line 2"
    assert results[1] == "MATCH"
    assert results[2] == "line 4"

def test_search_context_overlap(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("line 1\nMATCH1\nline 3\nMATCH2\nline 5", encoding='utf-8')
    output_file = tmp_path / "output.txt"

    # Both matches share line 3 as context
    search_mode(
        input_files=[str(input_file)],
        query="MATCH",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        before_context=1,
        after_context=1
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    # line 1 (context for MATCH1)
    # MATCH1
    # line 3 (context for both)
    # MATCH2
    # line 5 (context for MATCH2)
    assert len(results) == 5
    assert results[0] == "line 1"
    assert results[1] == "MATCH1"
    assert results[2] == "line 3"
    assert results[3] == "MATCH2"
    assert results[4] == "line 5"

def test_search_context_separator(tmp_path):
    input_file = tmp_path / "test.txt"
    # MATCH1 at line 2, MATCH2 at line 10. Gap in between.
    lines = [f"line {i}" for i in range(1, 15)]
    lines[1] = "MATCH1"
    lines[9] = "MATCH2"
    input_file.write_text("\n".join(lines), encoding='utf-8')
    output_file = tmp_path / "output.txt"

    search_mode(
        input_files=[str(input_file)],
        query="MATCH",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        before_context=1,
        after_context=1
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    # Expected:
    # line 1
    # MATCH1
    # line 3
    # --
    # line 9
    # MATCH2
    # line 11
    assert "--" in results
    idx = results.index("--")
    assert results[idx-1] == "line 3"
    assert results[idx+1] == "line 9"

def test_search_context_line_numbers(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("line 1\nMATCH\nline 3", encoding='utf-8')
    output_file = tmp_path / "output.txt"

    search_mode(
        input_files=[str(input_file)],
        query="MATCH",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        line_numbers=True,
        before_context=1,
        after_context=1
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    assert len(results) == 3
    assert results[0] == "1- line 1"
    assert results[1] == "2: MATCH"
    assert results[2] == "3- line 3"

def test_scan_context(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("line 1\nTYPO\nline 3", encoding='utf-8')
    output_file = tmp_path / "output.txt"

    scan_mode(
        input_files=[str(input_file)],
        mapping_file=None,
        ad_hoc=["TYPO:CORRECTION"],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        line_numbers=True,
        before_context=1,
        after_context=1
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    assert len(results) == 3
    assert results[0] == "1- line 1"
    assert results[1] == "2: TYPO"
    assert results[2] == "3- line 3"

def test_format_search_line_with_color(monkeypatch):
    # Mock color constants
    monkeypatch.setattr(multitool, "BOLD", "<b>")
    monkeypatch.setattr(multitool, "BLUE", "<blue>")
    monkeypatch.setattr(multitool, "RESET", "<reset>")

    from multitool import _format_search_line

    # With filename and line number
    formatted = _format_search_line("file.txt", 10, ":", "content", use_color=True)
    assert formatted == "<b><blue>file.txt:10:<reset> content"

    # Only line number
    formatted = _format_search_line(None, 10, "-", "content", use_color=True)
    assert formatted == "<b><blue>10-<reset> content"

    # No prefix
    formatted = _format_search_line(None, None, ":", "content", use_color=True)
    assert formatted == "content"

    # No color
    formatted = _format_search_line("file.txt", 10, ":", "content", use_color=False)
    assert formatted == "file.txt:10: content"
