import logging
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    monkeypatch.setattr(multitool, "tqdm", lambda iterable=None, *_, **__: iterable if iterable is not None else MagicMock())

def test_extract_frontmatter_empty_file(tmp_path):
    empty_file = tmp_path / "empty.md"
    empty_file.write_text("")
    results = list(multitool._extract_frontmatter(str(empty_file)))
    assert results == []

def test_extract_frontmatter_unclosed_markers(tmp_path):
    unclosed_file = tmp_path / "unclosed.md"
    unclosed_file.write_text("---\ntitle: test\n")
    results = list(multitool._extract_frontmatter(str(unclosed_file)))
    assert results == []

def test_extract_frontmatter_pyyaml_missing_fallback(tmp_path, caplog):
    test_file = tmp_path / "test.md"
    test_file.write_text("---\ntitle: test\n---\n")
    with patch.dict(sys.modules, {'yaml': None}):
        with caplog.at_level(logging.ERROR):
            results = list(multitool._extract_frontmatter(str(test_file)))
            assert results == []
            assert "PyYAML is required" in caplog.text

def test_convert_mode_no_matching_data_warning(tmp_path, caplog):
    input_file = tmp_path / "test.json"
    input_file.write_text('{"foo": "bar"}')
    with caplog.at_level(logging.WARNING):
        multitool.convert_mode(
            input_files=[str(input_file)],
            output_file=str(tmp_path / "out.xml"),
            output_format="xml",
            key="nonexistent"
        )
    assert "No data found to convert." in caplog.text

def test_extract_path_items_stdin_early_return():
    results = list(multitool._extract_path_items("-"))
    assert results == []

def test_format_search_line_no_prefix_rendering():
    content = "plain content"
    result = multitool._format_search_line("file.txt", 0, content, True, False, False, False)
    assert result == content

def test_format_search_line_no_prefix_rendering_with_color():
    content = "color content"
    result = multitool._format_search_line("file.txt", 0, content, True, False, False, True)
    assert result == content

def test_search_mode_file_group_blank_line_separation(tmp_path):
    file1 = tmp_path / "file1.txt"
    file1.write_text("match1\n")
    file2 = tmp_path / "file2.txt"
    file2.write_text("match2\n")
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
    content = output_file.read_text().splitlines()
    assert any("file1.txt" in line for line in content)
    assert "match1" in content
    assert "" in content
    assert any("file2.txt" in line for line in content)
    assert "match2" in content

def test_standardize_mode_transitive_fuzzy_mapping_closure(tmp_path):
    content = "apple\n" * 100 + "apples\n" * 40 + "appless\n" * 10
    test_file = tmp_path / "transitive.txt"
    test_file.write_text(content)
    output_file = tmp_path / "output.txt"
    multitool.standardize_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        quiet=True,
        clean_items=True,
        fuzzy=1,
        threshold=2
    )
    result = output_file.read_text().splitlines()
    assert all(line == "apple" for line in result)

def test_standardize_mode_casing_only_within_fuzzy_block(tmp_path):
    content = "Word\n" * 10 + "word\n" * 5 + "Werd\n" * 2
    test_file = tmp_path / "fuzzy_casing.txt"
    test_file.write_text(content)
    output_file = tmp_path / "output.txt"
    multitool.standardize_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        quiet=True,
        clean_items=True,
        fuzzy=1,
        threshold=2
    )
    result = output_file.read_text()
    assert result.count("Word") == 17
    assert "word" not in result
    assert "Werd" not in result

def test_scan_mode_file_group_blank_line_separation(tmp_path):
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("word\n")
    file1 = tmp_path / "file1.txt"
    file1.write_text("word\n")
    file2 = tmp_path / "file2.txt"
    file2.write_text("word\n")
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
    content = output_file.read_text().splitlines()
    assert any("file1.txt" in line for line in content)
    assert "" in content
    assert any("file2.txt" in line for line in content)
