import json
import logging
import pytest
from pathlib import Path
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_load_mapping_file_json_flat(tmp_path):
    mapping_file = tmp_path / "mapping.json"
    data = {"teh": "the", "recieve": "receive"}
    mapping_file.write_text(json.dumps(data))
    mapping = multitool._load_mapping_file(str(mapping_file))
    assert mapping == {"teh": "the", "recieve": "receive"}

def test_load_mapping_file_json_replacements(tmp_path):
    mapping_file = tmp_path / "mapping.json"
    data = {
        "replacements": [
            {"typo": "teh", "correct": "the"},
            {"typo": "recieve", "correct": "receive"}
        ]
    }
    mapping_file.write_text(json.dumps(data))
    mapping = multitool._load_mapping_file(str(mapping_file))
    assert mapping == {"teh": "the", "recieve": "receive"}

def test_load_mapping_file_json_list(tmp_path):
    mapping_file = tmp_path / "mapping.json"
    data = [
        {"typo": "teh", "correct": "the"},
        {"typo": "recieve", "correct": "receive"}
    ]
    mapping_file.write_text(json.dumps(data))
    mapping = multitool._load_mapping_file(str(mapping_file))
    assert mapping == {"teh": "the", "recieve": "receive"}

def test_load_mapping_file_yaml(tmp_path):
    mapping_file = tmp_path / "mapping.yaml"
    mapping_file.write_text("teh: the\nrecieve: receive\n")
    mapping = multitool._load_mapping_file(str(mapping_file))
    assert mapping == {"teh": "the", "recieve": "receive"}

def test_load_mapping_file_table(tmp_path):
    mapping_file = tmp_path / "mapping.txt"
    # Table format: typo = "correction"
    mapping_file.write_text('teh = "the"\nrecieve = "receive"\n')
    mapping = multitool._load_mapping_file(str(mapping_file))
    assert mapping == {"teh": "the", "recieve": "receive"}

def test_load_mapping_file_comments_and_empty(tmp_path):
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text(
        "# This is a comment\n"
        "\n"
        "teh -> the\n"
        "  \n"
        "# Another comment\n"
        "recieve -> receive\n"
    )
    mapping = multitool._load_mapping_file(str(mapping_file))
    assert mapping == {"teh": "the", "recieve": "receive"}

def test_load_mapping_file_json_error(tmp_path, caplog):
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text("{invalid json")
    with caplog.at_level(logging.ERROR):
        mapping = multitool._load_mapping_file(str(mapping_file))
    assert "Failed to parse JSON mapping" in caplog.text
    assert mapping == {}

def test_map_mode_output_length_filter_bug(tmp_path):
    """
    Verify that transformed items are subject to length filtering.
    Currently this test fails because map_mode doesn't re-apply filters to mapped values.
    """
    input_file = tmp_path / "input.txt"
    input_file.write_text("the\nquick\n")
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("the,a\n") # 'the' -> 'a' (shorter than min_length=3)
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
    # If the bug is present, 'a' will be in lines.
    # If fixed, 'a' should be filtered out because len('a') < 3.
    assert "a" not in lines
    assert lines == ["quick"]
