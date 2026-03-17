import sys
import json
import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_extract_pairs_markdown_table_and_colon(tmp_path):
    f = tmp_path / "pairs.txt"
    f.write_text(
        "| typo | correction |\n"
        "| :--- | :--- |\n"
        "|  teh  |  the  |\n"
        "| edge | case |\n"
        "apple: red\n"
    )

    # _extract_pairs is called by pairs_mode
    out = tmp_path / "out.txt"
    multitool.pairs_mode([str(f)], str(out), min_length=1, max_length=100, process_output=False)

    content = out.read_text().splitlines()
    assert "teh -> the" in content
    assert "edge -> case" in content
    assert "apple -> red" in content
    assert "typo -> correction" not in content

def test_words_mode_smart_split():
    input_text = "CamelCase snake_case"
    with patch("multitool._read_file_lines_robust", return_value=[input_text]):
        mock_outfile = MagicMock()
        with patch("multitool.smart_open_output", return_value=MagicMock(__enter__=lambda s: mock_outfile)):
            multitool.words_mode(["mock"], "out", 1, 100, True, smart=True, clean_items=False)

    written_items = [call.args[0].strip() for call in mock_outfile.write.call_args_list]
    assert "Camel" in written_items
    assert "Case" in written_items
    assert "snake" in written_items
    assert "case" in written_items

def test_stats_mode_markdown_with_pairs(tmp_path):
    f = tmp_path / "input.txt"
    # Added empty lines to cover line 1114 (in multitool.py)
    f.write_text("teh -> the\n\n  \n")
    out = tmp_path / "stats.md"

    multitool.stats_mode([str(f)], str(out), min_length=1, max_length=100, process_output=False, include_pairs=True, output_format='markdown', clean_items=True)

    content = out.read_text()
    assert "### ANALYSIS SUMMARY" in content
    assert "### PAIRED DATA STATISTICS" in content
    # Simple pair to ensure dist 1
    f.write_text("a -> b\n")
    multitool.stats_mode([str(f)], str(out), min_length=1, max_length=100, process_output=False, include_pairs=True, output_format='markdown', clean_items=True)
    content = out.read_text()
    assert "| Min character changes | 1 |" in content

def test_near_duplicates_mode_optimizations(tmp_path):
    f = tmp_path / "words.txt"
    f.write_text("cat\nhat\ncattle\n")
    out = tmp_path / "out.txt"

    multitool.near_duplicates_mode([str(f)], str(out), 1, 100, process_output=True, max_dist=1, show_dist=True)

    content = out.read_text()
    assert "cat -> hat (changes: 1)" in content
    assert "cattle" not in content

def test_fuzzymatch_mode_optimizations(tmp_path):
    f1 = tmp_path / "list1.txt"
    f1.write_text("cat\n")
    f2 = tmp_path / "list2.txt"
    f2.write_text("a\nbat\ndoggy\n")
    out = tmp_path / "out.txt"

    multitool.fuzzymatch_mode([str(f1)], str(f2), str(out), 1, 100, process_output=True, max_dist=1, show_dist=True)

    content = out.read_text()
    assert "cat -> bat (changes: 1)" in content
    assert "doggy" not in content

def test_discovery_mode_optimizations(tmp_path):
    f = tmp_path / "text.txt"
    content = "cat " + "hat "*5 + "bat "*5 + "doggy "*5 + "a "*5
    f.write_text(content)
    out = tmp_path / "out.txt"

    multitool.discovery_mode([str(f)], str(out), 1, 100, process_output=True, freq_min=5, max_dist=1, show_dist=True)

    content = out.read_text()
    assert "cat -> bat (changes: 1)" in content
    assert "cat -> hat (changes: 1)" in content
    assert "doggy" not in content
    assert "-> a" not in content

def test_md_table_mode_columns(tmp_path):
    f = tmp_path / "table.md"
    f.write_text(
        "| col0 | col1 | col2 |\n"
        "| --- | --- | --- |\n"
        "| val0 | val1 | val2 |\n"
    )
    out = tmp_path / "out.txt"
    # Test --column option to cover lines 775-777
    # Use clean_items=False to preserve 'val0'
    multitool.md_table_mode([str(f)], str(out), 1, 100, True, columns=[0, 2], clean_items=False)
    content = out.read_text().splitlines()
    assert "val0" in content
    assert "val2" in content
    assert "val1" not in content

def test_write_output_yaml_no_module(tmp_path, monkeypatch):
    # Mock ImportError for yaml to cover lines 324-326
    import builtins
    real_import = builtins.__import__
    def mocked_import(name, *args, **kwargs):
        if name == 'yaml':
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)

    items = ["apple", "banana"]
    out = tmp_path / "out.yaml"
    multitool.write_output(items, str(out), output_format='yaml')
    content = out.read_text()
    assert "- apple\n- banana\n" in content
