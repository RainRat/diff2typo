import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

# Replicate the disable_tqdm fixture
@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)


def test_json_mode_simple_key(tmp_path):
    input_file = tmp_path / "input.json"
    data = [{"name": "alice"}, {"name": "bob"}]
    input_file.write_text(json.dumps(data))
    output_file = tmp_path / "output.txt"

    multitool.json_mode(
        [str(input_file)],
        str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        key="name",
    )

    result = output_file.read_text().splitlines()
    assert sorted(result) == ["alice", "bob"]


def test_json_mode_nested_key(tmp_path):
    input_file = tmp_path / "input.json"
    data = [
        {"meta": {"id": "1", "user": "alice"}},
        {"meta": {"id": "2", "user": "bob"}},
    ]
    input_file.write_text(json.dumps(data))
    output_file = tmp_path / "output.txt"

    multitool.json_mode(
        [str(input_file)],
        str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        key="meta.user",
    )

    result = output_file.read_text().splitlines()
    assert sorted(result) == ["alice", "bob"]


def test_json_mode_list_handling(tmp_path):
    # Tests where the key points to a list or traverses through a list implicitly
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.txt"

    # Case: Root is object, key points to list of strings
    # We use distinct strings to verify that all items in the list are extracted
    data = {"tags": ["alpha", "beta"]}
    input_file.write_text(json.dumps(data))

    multitool.json_mode(
        [str(input_file)],
        str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        key="tags",
    )
    result = output_file.read_text().splitlines()
    assert sorted(result) == ["alpha", "beta"]


def test_json_mode_missing_key(tmp_path):
    input_file = tmp_path / "input.json"
    data = [{"name": "alice"}, {"age": 30}]
    input_file.write_text(json.dumps(data))
    output_file = tmp_path / "output.txt"

    multitool.json_mode(
        [str(input_file)],
        str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        key="name",
    )

    # Should only find "alice"
    result = output_file.read_text().splitlines()
    assert sorted(result) == ["alice"]


def test_json_mode_malformed_json(tmp_path, caplog):
    input_file = tmp_path / "input.json"
    input_file.write_text("{invalid json")
    output_file = tmp_path / "output.txt"

    with caplog.at_level(logging.ERROR):
        multitool.json_mode(
            [str(input_file)],
            str(output_file),
            min_length=1,
            max_length=100,
            process_output=True,
            key="name",
        )

    assert "Failed to parse JSON" in caplog.text
    # Output file should be created but empty (or contains nothing from this file)
    assert output_file.exists()
    assert output_file.read_text() == ""


def test_json_mode_empty_file(tmp_path):
    input_file = tmp_path / "empty.json"
    input_file.write_text("")
    output_file = tmp_path / "output.txt"

    multitool.json_mode(
        [str(input_file)],
        str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        key="name",
    )

    assert output_file.read_text() == ""


def test_json_mode_deeply_nested_list(tmp_path):
    # Data: List of objects, each has a 'items' list of objects, each has 'value'
    input_file = tmp_path / "input.json"
    data = [
        {
            "items": [
                {"value": "one"},
                {"value": "two"}
            ]
        },
        {
            "items": [
                {"value": "three"}
            ]
        }
    ]
    input_file.write_text(json.dumps(data))
    output_file = tmp_path / "output.txt"

    multitool.json_mode(
        [str(input_file)],
        str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        key="items.value",
    )

    result = output_file.read_text().splitlines()
    assert sorted(result) == ["one", "three", "two"]

def test_json_mode_non_dict_structure(tmp_path):
    # Case where intermediate key is not a dict
    input_file = tmp_path / "input.json"
    data = {"a": "string_value"}
    # Try to access a.b
    input_file.write_text(json.dumps(data))
    output_file = tmp_path / "output.txt"

    multitool.json_mode(
        [str(input_file)],
        str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        key="a.b",
    )

    # "a" is a string, so it doesn't have key "b". traverse_json checks isinstance(data, dict).
    # so it should yield nothing.
    assert output_file.read_text() == ""

def test_json_mode_numeric_values(tmp_path):
    # Ensure numeric values are converted to string and filtered correctly
    input_file = tmp_path / "input.json"
    data = {"val": 123} # clean_and_filter removes numbers -> "" -> filtered out by length
    input_file.write_text(json.dumps(data))
    output_file = tmp_path / "output.txt"

    multitool.json_mode(
        [str(input_file)],
        str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        key="val",
    )

    assert output_file.read_text() == ""

    # Try with something that remains after filtering
    data2 = {"val": "123abc456"} # -> "abc"
    input_file.write_text(json.dumps(data2))
    multitool.json_mode(
        [str(input_file)],
        str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        key="val",
    )
    assert output_file.read_text().strip() == "abc"
