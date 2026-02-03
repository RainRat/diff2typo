import pytest
from pathlib import Path
import multitool
import sys

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_load_mapping_file_csv(tmp_path):
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\nrecieve,receive\n")
    mapping = multitool._load_mapping_file(str(mapping_file))
    assert mapping == {"teh": "the", "recieve": "receive"}

def test_load_mapping_file_arrow(tmp_path):
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("teh -> the\nrecieve -> receive\n")
    mapping = multitool._load_mapping_file(str(mapping_file))
    assert mapping == {"teh": "the", "recieve": "receive"}

def test_map_mode_basic(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh\nquick\nbrown\nfox\n")
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\n")
    output_file = tmp_path / "output.txt"

    multitool.map_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
    )

    assert output_file.read_text().splitlines() == ["the", "quick", "brown", "fox"]

def test_map_mode_drop_missing(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh\nquick\nbrown\nfox\n")
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\n")
    output_file = tmp_path / "output.txt"

    multitool.map_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        drop_missing=True
    )

    assert output_file.read_text().splitlines() == ["the"]

def test_map_mode_raw(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("TeH\nQuick\nBrown\nFox\n")
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("TeH,The\n")
    output_file = tmp_path / "output.txt"

    multitool.map_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        clean_items=False
    )

    assert output_file.read_text().splitlines() == ["The", "Quick", "Brown", "Fox"]

def test_map_mode_length_filter_bug(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a\nbb\nccc\ndddd\n")
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("bb,BB\n")
    output_file = tmp_path / "output.txt"

    multitool.map_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=False,
        clean_items=False
    )

    lines = output_file.read_text().splitlines()
    # Currently, 'a' and 'BB' WILL be in lines due to the bug.
    # We want them to be NOT in lines.
    assert "a" not in lines
    assert "BB" not in lines
    assert lines == ["ccc", "dddd"]

def test_map_mode_process_output(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh\nquick\nquick\n")
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\n")
    output_file = tmp_path / "output.txt"

    multitool.map_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
    )

    # Should be sorted and unique
    assert output_file.read_text().splitlines() == ["quick", "the"]

def test_map_mode_mapping_cleaning(tmp_path):
    """Verify that mapping keys are cleaned when clean_items=True."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh\n")
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("TeH!,the\n") # Key has punctuation and caps
    output_file = tmp_path / "output.txt"

    multitool.map_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        clean_items=True
    )

    # teh cleaned is 'teh'. TeH! cleaned is 'teh'. Should match.
    assert output_file.read_text().strip() == "the"
