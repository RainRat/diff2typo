import os
import json
import yaml
import pytest
from multitool import convert_mode

def test_convert_json_to_yaml(tmp_path):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.yaml"

    data = {
        "metadata": {
            "name": "test",
            "tags": ["a", "b", "c"]
        },
        "content": "hello world"
    }

    input_file.write_text(json.dumps(data))

    convert_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        output_format='yaml'
    )

    assert output_file.exists()
    with open(output_file, 'r') as f:
        output_data = yaml.safe_load(f)

    assert output_data == data

def test_convert_with_key_extraction(tmp_path):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"

    data = {
        "items": [
            {"id": 1, "value": "v1"},
            {"id": 2, "value": "v2"}
        ],
        "other": "stuff"
    }

    input_file.write_text(json.dumps(data))

    convert_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        key="items",
        output_format='json'
    )

    assert output_file.exists()
    with open(output_file, 'r') as f:
        output_data = json.load(f)

    # Since there's only one result for 'items' (the list), it should be extracted directly
    assert output_data == data["items"]

def test_convert_multi_doc_yaml_to_jsonl(tmp_path):
    input_file = tmp_path / "input.yaml"
    output_file = tmp_path / "output.json"

    docs = [
        {"a": 1},
        {"b": 2}
    ]

    input_file.write_text(yaml.dump_all(docs))

    # We want to see how it handles multiple docs.
    # Current implementation: if more than one result, it outputs a list of results.
    convert_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        output_format='json'
    )

    with open(output_file, 'r') as f:
        output_data = json.load(f)

    assert output_data == docs
