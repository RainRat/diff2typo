import sys
import os
import logging
import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

def test_extract_frontmatter_empty_file(tmp_path):
    """Cover line 1666: empty file in _extract_frontmatter."""
    input_file = tmp_path / "empty.md"
    input_file.write_text("", encoding='utf-8')

    results = list(multitool._extract_frontmatter(str(input_file)))
    assert results == []

def test_extract_frontmatter_unclosed(tmp_path):
    """Cover line 1681: unclosed frontmatter in _extract_frontmatter."""
    input_file = tmp_path / "unclosed.md"
    input_file.write_text("---\ntitle: unclosed\nNo end separator", encoding='utf-8')

    results = list(multitool._extract_frontmatter(str(input_file)))
    assert results == []

def test_extract_frontmatter_no_pyyaml(tmp_path):
    """Cover lines 1688-1690: missing PyYAML in _extract_frontmatter."""
    input_file = tmp_path / "test.md"
    input_file.write_text("---\ntitle: test\n---\nbody", encoding='utf-8')

    with patch.dict('sys.modules', {'yaml': None}):
        with patch('logging.error') as mock_log:
            results = list(multitool._extract_frontmatter(str(input_file)))
            assert results == []
            assert any("PyYAML is required" in call[0][0] for call in mock_log.call_args_list)

def test_convert_mode_single_result(tmp_path):
    """Cover lines 2067-2068: single result in convert_mode."""
    input_file = tmp_path / "input.json"
    input_file.write_text('{"key": "value"}')
    output_file = tmp_path / "output.json"

    multitool.convert_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        output_format='json',
        quiet=True
    )

    content = output_file.read_text()
    assert '"key": "value"' in content
    assert content.strip().startswith('{')

def test_convert_mode_no_results(tmp_path):
    """Cover lines 2062-2063: no results in convert_mode."""
    input_file = tmp_path / "empty.json"
    input_file.write_text('')
    output_file = tmp_path / "output.json"

    with patch('logging.warning') as mock_log:
        multitool.convert_mode(
            input_files=[str(input_file)],
            output_file=str(output_file),
            output_format='json',
            quiet=True
        )
        assert any("No data found" in call[0][0] for call in mock_log.call_args_list)

def test_extract_path_items_stdin():
    """Cover line 4222: path is '-' in _extract_path_items."""
    results = list(multitool._extract_path_items("-"))
    assert results == []

def test_extract_path_items_no_flags():
    """Cover line 4235: default to full path."""
    results = list(multitool._extract_path_items("some/file.txt"))
    assert results == ["some/file.txt"]

def test_search_mode_headings_blank_line(tmp_path):
    """Cover line 4591: blank line between file groups in search_mode."""
    file1 = (tmp_path / "file1.txt").resolve()
    file1.write_text("match1")
    file2 = (tmp_path / "file2.txt").resolve()
    file2.write_text("match2")
    output_file = tmp_path / "output.txt"

    multitool.search_mode(
        input_files=[str(file1), str(file2)],
        query="match",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        heading=True,
        quiet=True
    )

    content = output_file.read_text()
    assert str(file1) in content
    assert str(file2) in content
    # Blank line between file1 results and file2 header
    assert "match1\n\n" in content

def test_scan_mode_headings_blank_line(tmp_path):
    """Cover line 6162: blank line between file groups in scan_mode."""
    file1 = (tmp_path / "file1.txt").resolve()
    file1.write_text("apple")
    file2 = (tmp_path / "file2.txt").resolve()
    file2.write_text("banana")
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text('{"apple": "APPLE", "banana": "BANANA"}')
    output_file = tmp_path / "output.txt"

    multitool.scan_mode(
        input_files=[str(file1), str(file2)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        heading=True,
        quiet=True
    )

    content = output_file.read_text()
    assert str(file1) in content
    assert str(file2) in content
    assert "apple\n\n" in content

def test_standardize_transitive_fuzzy_and_casing(tmp_path):
    """Cover lines 5606-5607 and 5622-5624."""
    input_file = tmp_path / "input.txt"
    content = ["apple"] * 1000 + ["Apple"] * 50 + ["apply"] * 100 + ["applys"] * 10
    input_file.write_text("\n".join(content))
    output_file = tmp_path / "output.txt"

    multitool.standardize_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        threshold=3.0,
        fuzzy=1,
        quiet=True
    )

    content = output_file.read_text()
    assert "applys" not in content
    assert "apply" not in content
    assert "Apple" not in content
    assert content.count("apple") == 1160

def test_standardize_casing_only_branch(tmp_path):
    """Cover line 5622-5624 more specifically."""
    input_file = tmp_path / "input.txt"
    content = ["Test"] * 10 + ["TEST"] * 5
    input_file.write_text("\n".join(content))
    output_file = tmp_path / "output.txt"

    multitool.standardize_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        quiet=True
    )
    content = output_file.read_text()
    assert "Test" in content
    assert "TEST" not in content
