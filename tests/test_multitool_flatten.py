import json
import pytest
from multitool import flatten_mode

def test_flatten_json(tmp_path, capsys):
    data = {
        "user": {
            "name": "Alice",
            "tags": ["engineer", "coder"],
            "stats": {"score": 10}
        }
    }
    f = tmp_path / "test.json"
    f.write_text(json.dumps(data))

    flatten_mode(
        input_files=[str(f)],
        output_file="-",
        min_length=1,
        max_length=1000,
        process_output=True,
        key="",
        output_format="line",
        quiet=True,
        clean_items=False
    )

    captured = capsys.readouterr()
    output = captured.out.strip().split("\n")
    assert "user.name -> Alice" in output
    assert "user.tags.0 -> engineer" in output
    assert "user.tags.1 -> coder" in output
    assert "user.stats.score -> 10" in output

def test_flatten_jsonl(tmp_path, capsys):
    lines = [
        json.dumps({"id": 1, "val": "a"}),
        json.dumps({"id": 2, "val": "b"})
    ]
    f = tmp_path / "test.jsonl"
    f.write_text("\n".join(lines))

    # We use extension .json to trigger the JSONL fallback logic in _yield_structured_docs
    # Or just rely on the fallback for unknown extensions.
    # Actually _yield_structured_docs checks .endswith('.json')
    f_json = tmp_path / "test.json"
    f_json.write_text("\n".join(lines))

    flatten_mode(
        input_files=[str(f_json)],
        output_file="-",
        min_length=1,
        max_length=1000,
        process_output=True,
        key="",
        output_format="line",
        quiet=True,
        clean_items=False
    )

    captured = capsys.readouterr()
    output = captured.out.strip().split("\n")
    assert "id -> 1" in output
    assert "id -> 2" in output
    assert "val -> a" in output
    assert "val -> b" in output

def test_flatten_with_key(tmp_path, capsys):
    data = {
        "metadata": {"version": "1.0"},
        "items": [
            {"name": "apple", "color": "red"},
            {"name": "banana", "color": "yellow"}
        ]
    }
    f = tmp_path / "test.json"
    f.write_text(json.dumps(data))

    flatten_mode(
        input_files=[str(f)],
        output_file="-",
        min_length=1,
        max_length=1000,
        process_output=True,
        key="items",
        output_format="line",
        quiet=True,
        clean_items=False
    )

    captured = capsys.readouterr()
    output = captured.out.strip().split("\n")
    assert "0.name -> apple" in output
    assert "1.color -> yellow" in output
    assert "metadata.version -> 1.0" not in output

def test_flatten_yaml(tmp_path, capsys):
    yaml_content = """
- name: first
  data: [1, 2]
---
- name: second
  data: [3]
"""
    f = tmp_path / "test.yaml"
    f.write_text(yaml_content)

    try:
        import yaml
    except ImportError:
        pytest.skip("PyYAML not installed")

    flatten_mode(
        input_files=[str(f)],
        output_file="-",
        min_length=1,
        max_length=1000,
        process_output=True,
        key="",
        output_format="line",
        quiet=True,
        clean_items=False
    )

    captured = capsys.readouterr()
    output = captured.out.strip().split("\n")
    assert "0.name -> first" in output
    assert "0.data.0 -> 1" in output
    assert "0.name -> second" in output
    assert "0.data.0 -> 3" in output

def test_flatten_toml(tmp_path, capsys):
    toml_content = """
[owner]
name = "Tom"
dob = 1979-05-27T07:32:00Z

[database]
server = "192.168.1.1"
ports = [ 8001, 8001, 8002 ]
"""
    f = tmp_path / "test.toml"
    f.write_text(toml_content)

    flatten_mode(
        input_files=[str(f)],
        output_file="-",
        min_length=1,
        max_length=1000,
        process_output=True,
        key="",
        output_format="line",
        quiet=True,
        clean_items=False
    )

    captured = capsys.readouterr()
    output = captured.out.strip().split("\n")
    assert "owner.name -> Tom" in output
    assert "database.server -> 192.168.1.1" in output
    assert "database.ports.0 -> 8001" in output
