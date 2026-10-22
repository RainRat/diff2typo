import json
import subprocess
import pytest
import multitool

def test_unflatten_mode_parse_types_primitives(tmp_path):
    input_file = tmp_path / "flat_typed.txt"
    output_file = tmp_path / "unflattened.json"

    content = (
        "config.enabled -> true\n"
        "config.debug -> FALSE\n"
        "config.null_val -> null\n"
        "config.none_val -> None\n"
        "config.count -> 42\n"
        "config.ratio -> 3.14\n"
        "config.name -> hello\n"
    )
    input_file.write_text(content, encoding="utf-8")

    multitool.unflatten_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        clean_items=False,
        parse_types=True,
        output_format="json",
    )

    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert data == {
        "config": {
            "enabled": True,
            "debug": False,
            "null_val": None,
            "none_val": None,
            "count": 42,
            "ratio": 3.14,
            "name": "hello",
        }
    }


def test_unflatten_mode_parse_types_json_structures(tmp_path):
    input_file = tmp_path / "flat_json.txt"
    output_file = tmp_path / "unflattened.json"

    content = (
        'data.tags -> ["a", "b", 3]\n'
        'data.meta -> {"key": "value"}\n'
    )
    input_file.write_text(content, encoding="utf-8")

    multitool.unflatten_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        clean_items=False,
        parse_types=True,
        output_format="json",
    )

    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert data == {
        "data": {
            "tags": ["a", "b", 3],
            "meta": {"key": "value"},
        }
    }


def test_unflatten_mode_default_string_behavior(tmp_path):
    input_file = tmp_path / "flat_str.txt"
    output_file = tmp_path / "unflattened.json"

    content = (
        "config.enabled -> true\n"
        "config.count -> 42\n"
    )
    input_file.write_text(content, encoding="utf-8")

    multitool.unflatten_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        clean_items=False,
        parse_types=False,
        output_format="json",
    )

    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert data == {
        "config": {
            "enabled": "true",
            "count": "42",
        }
    }


def test_unflatten_cli_parse_types_integration(tmp_path):
    input_file = tmp_path / "cli_typed.txt"
    input_file.write_text("app.active -> true\napp.port -> 8080\n")

    result = subprocess.run(
        ["python3", "multitool.py", "unflatten", str(input_file), "--parse-types", "--raw", "-f", "json"],
        capture_output=True,
        text=True,
        check=True
    )
    data = json.loads(result.stdout)
    assert data == {
        "app": {
            "active": True,
            "port": 8080
        }
    }


def test_parse_value_type_invalid_json_fallback():
    assert multitool._parse_value_type("{invalid json}") == "{invalid json}"
    assert multitool._parse_value_type("[1, 2, unquoted]") == "[1, 2, unquoted]"


def test_parse_value_type_nan_and_inf_handling():
    assert multitool._parse_value_type("nan") == "nan"
    assert multitool._parse_value_type("inf") == "inf"
    assert multitool._parse_value_type("-inf") == "-inf"
