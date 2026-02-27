import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def setup_logging():
    # Ensure INFO logs are captured
    logging.getLogger().setLevel(logging.INFO)

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_stats_mode_items_json(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("apple banana apple cherry\n")
    out = tmp_path / "stats.json"

    multitool.stats_mode([str(f)], str(out), min_length=1, max_length=100, process_output=False, output_format='json')

    with open(out) as j:
        stats = json.load(j)

    assert stats["items"]["total_encountered"] == 4
    assert stats["items"]["total_filtered"] == 4
    assert stats["items"]["unique_count"] == 3
    assert stats["items"]["min_length"] == 5
    assert stats["items"]["max_length"] == 6
    assert stats["items"]["shortest"] == "apple"
    assert stats["items"]["longest"] == "banana"

def test_stats_mode_pairs_json(tmp_path):
    f = tmp_path / "pairs.txt"
    f.write_text("teh -> the\nteh -> tea\napple -> apple\n")
    out = tmp_path / "stats.json"

    multitool.stats_mode([str(f)], str(out), min_length=1, max_length=100, process_output=False, include_pairs=True, output_format='json')

    with open(out) as j:
        stats = json.load(j)

    assert stats["pairs"]["total_extracted"] == 3
    assert stats["pairs"]["total_filtered"] == 3
    assert stats["pairs"]["unique_pairs"] == 3
    assert stats["pairs"]["conflicts"] == 1 # teh -> the, tea
    assert stats["pairs"]["overlaps"] == 1 # apple is both typo and correction
    assert stats["pairs"]["min_dist"] == 0 # apple -> apple
    assert stats["pairs"]["max_dist"] == 2 # teh -> tea

def test_stats_mode_yaml(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("apple\n")
    out = tmp_path / "stats.yaml"

    # Test with yaml module if available
    multitool.stats_mode([str(f)], str(out), min_length=1, max_length=100, process_output=False, output_format='yaml')
    content = out.read_text()
    assert "total_encountered: 1" in content

def test_stats_mode_yaml_no_module(tmp_path, monkeypatch):
    # Mock ImportError for yaml
    import builtins
    real_import = builtins.__import__
    def mocked_import(name, *args, **kwargs):
        if name == 'yaml':
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)

    f = tmp_path / "input.txt"
    f.write_text("apple\n")
    out = tmp_path / "stats.yaml"

    multitool.stats_mode([str(f)], str(out), min_length=1, max_length=100, process_output=False, include_pairs=True, output_format='yaml')
    content = out.read_text()
    assert "items:" in content
    assert "total_encountered: 1" in content
    assert "pairs:" in content

def test_stats_mode_markdown(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("apple\n")
    out = tmp_path / "stats.md"

    multitool.stats_mode([str(f)], str(out), min_length=1, max_length=100, process_output=False, include_pairs=True, output_format='markdown')
    content = out.read_text()
    assert "### ANALYSIS STATISTICS" in content
    assert "### PAIRED DATA STATISTICS" in content
    assert "| Total items encountered | 1 |" in content

def test_stats_mode_human_readable(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("apple banana\n")
    out = tmp_path / "stats.txt"

    # Mock isatty to False for no colors
    mock_file = MagicMock()
    mock_file.isatty.return_value = False

    with patch("multitool.smart_open_output") as mock_smart_open:
        mock_smart_open.return_value.__enter__.return_value = mock_file
        multitool.stats_mode([str(f)], str(out), min_length=1, max_length=100, process_output=False, include_pairs=True, output_format='line')

    # Collect written calls
    written_content = "".join(call.args[0] for call in mock_file.write.call_args_list)
    assert "ANALYSIS STATISTICS" in written_content
    assert "PAIRED DATA STATISTICS" in written_content
    assert "\033" not in written_content # No ANSI colors

def test_stats_mode_human_readable_tty(tmp_path, monkeypatch):
    f = tmp_path / "input.txt"
    f.write_text("apple\n")
    out = tmp_path / "stats.txt"

    # Force color constants to have values for the test
    monkeypatch.setattr(multitool, "BOLD", "\033[1m")
    monkeypatch.setattr(multitool, "GREEN", "\033[1;32m")
    monkeypatch.setattr(multitool, "YELLOW", "\033[1;33m")
    monkeypatch.setattr(multitool, "RESET", "\033[0m")

    # Mock isatty to True for colors
    mock_file = MagicMock()
    mock_file.isatty.return_value = True

    with patch("multitool.smart_open_output") as mock_smart_open:
        mock_smart_open.return_value.__enter__.return_value = mock_file
        multitool.stats_mode([str(f)], str(out), min_length=1, max_length=100, process_output=False, output_format='line')

    written_content = "".join(call.args[0] for call in mock_file.write.call_args_list)
    assert "\033" in written_content # ANSI colors present

def test_stats_mode_empty_input(tmp_path):
    f = tmp_path / "empty.txt"
    f.touch()
    out = tmp_path / "stats.json"

    multitool.stats_mode([str(f)], str(out), min_length=1, max_length=100, process_output=False, output_format='json')

    with open(out) as j:
        stats = json.load(j)
    assert stats["items"]["total_encountered"] == 0
    assert "min_length" not in stats["items"]

def test_stats_mode_filtering(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("a bb ccc\n")
    out = tmp_path / "stats.json"

    # Only "bb" should pass length filter 2-2
    multitool.stats_mode([str(f)], str(out), min_length=2, max_length=2, process_output=False, output_format='json')

    with open(out) as j:
        stats = json.load(j)
    assert stats["items"]["total_encountered"] == 3
    assert stats["items"]["total_filtered"] == 1
    assert stats["items"]["unique_count"] == 1

def test_stats_mode_pairs_filtering(tmp_path):
    f = tmp_path / "pairs.txt"
    f.write_text("a -> bb\ncc -> dddd\n")
    out = tmp_path / "stats.json"

    # min_length=2, max_length=3 -> Only "cc" on left?
    # Wait, both sides must meet criteria in stats_mode for pairs
    multitool.stats_mode([str(f)], str(out), min_length=2, max_length=3, process_output=False, include_pairs=True, output_format='json')

    with open(out) as j:
        stats = json.load(j)
    # pair 1: a (len 1) -> bb (len 2) -> filtered out because of 'a'
    # pair 2: cc (len 2) -> dddd (len 4) -> filtered out because of 'dddd'
    assert stats["pairs"]["total_extracted"] == 2
    assert stats["pairs"]["total_filtered"] == 0

def test_stats_mode_pairs_empty_after_clean(tmp_path):
    f = tmp_path / "pairs.txt"
    f.write_text("123 -> !!!\n")
    out = tmp_path / "stats.json"

    multitool.stats_mode([str(f)], str(out), min_length=1, max_length=100, process_output=False, include_pairs=True, output_format='json', clean_items=True)

    with open(out) as j:
        stats = json.load(j)
    # Both cleaned to empty strings, should be skipped
    assert stats["pairs"]["total_filtered"] == 0
