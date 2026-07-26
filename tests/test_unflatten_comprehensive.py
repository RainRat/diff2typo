import sys
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool
from multitool import unflatten_mode

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    monkeypatch.setattr(multitool, "tqdm", lambda iterable=None, *_, **__: iterable if iterable is not None else MagicMock())

def test_unflatten_mode_json(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a.b.c -> value\na.b.d -> 123\n")
    output_file = tmp_path / "output.json"

    unflatten_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='json',
        quiet=True,
        clean_items=False
    )

    with open(output_file) as f:
        data = json.load(f)
    assert data == {"a": {"b": {"c": "value", "d": "123"}}}

def test_unflatten_mode_yaml(tmp_path):
    pytest.importorskip("yaml")
    input_file = tmp_path / "input.txt"
    input_file.write_text("a.b -> c\n")
    output_file = tmp_path / "output.yaml"

    unflatten_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='yaml',
        quiet=True,
        clean_items=False
    )

    import yaml
    with open(output_file) as f:
        data = yaml.safe_load(f)
    assert data == {"a": {"b": "c"}}

def test_unflatten_mode_toml(tmp_path):
    pytest.importorskip("toml")
    toml_input = tmp_path / "input.toml"
    toml_input.write_text("a = \"b\"\nkey = \"val\"\n")
    output_file = tmp_path / "output.toml"

    unflatten_mode(
        input_files=[str(toml_input)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='toml',
        quiet=True,
        clean_items=False
    )

    import toml
    with open(output_file) as f:
        data = toml.load(f)
    assert data == {"a": "b", "key": "val"}

def test_unflatten_mode_xml_basic(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("user.name -> Alice\nuser.age -> 25\n")
    output_file = tmp_path / "output.xml"

    unflatten_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='xml',
        quiet=True,
        clean_items=False
    )

    tree = ET.parse(output_file)
    root = tree.getroot()
    assert root.tag == "root"
    user = root.find("user")
    assert user.find("name").text == "Alice"
    assert user.find("age").text == "25"

def test_unflatten_mode_list_reconstruction(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("items.0 -> first\nitems.1 -> second\n")
    output_file = tmp_path / "output.json"

    unflatten_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='json',
        quiet=True,
        clean_items=False
    )

    with open(output_file) as f:
        data = json.load(f)
    assert data == {"items": ["first", "second"]}

def test_unflatten_mode_key_filter(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("meta.id -> 1\ndata.value -> 100\ndata.status -> ok\n")
    output_file = tmp_path / "output.json"

    unflatten_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        key="data",
        output_format='json',
        quiet=True,
        clean_items=False
    )

    with open(output_file) as f:
        data = json.load(f)
    assert data == {"value": "100", "status": "ok"}

def test_unflatten_mode_clean_and_filter(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a -> valid\nb -> 123!@#\nc -> toolong\n")
    output_file = tmp_path / "output.json"

    unflatten_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=2,
        max_length=5,
        process_output=False,
        output_format='json',
        quiet=True,
        clean_items=True
    )

    with open(output_file) as f:
        data = json.load(f)
    assert data == {"a": "valid"}

def test_unflatten_mode_xml_numeric_keys_fixed(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("data.0 -> first\ndata.2 -> second\n")
    output_file = tmp_path / "output.xml"

    unflatten_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='xml',
        quiet=True,
        clean_items=False
    )

    tree = ET.parse(output_file)
    root = tree.getroot()
    data = root.find("data")
    assert data.find("_0").text == "first"
    assert data.find("_2").text == "second"

def test_unflatten_mode_xml_list_items(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("items.0 -> first\nitems.1 -> second\n")
    output_file = tmp_path / "output.xml"

    unflatten_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='xml',
        quiet=True,
        clean_items=False
    )

    tree = ET.parse(output_file)
    root = tree.getroot()
    items = root.find("items")
    item_tags = items.findall("item")
    assert len(item_tags) == 2
    assert item_tags[0].text == "first"
    assert item_tags[1].text == "second"

def test_unflatten_mode_exact_key_skip(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("user -> Alice\nuser.name -> Bob\n")
    output_file = tmp_path / "output.json"

    unflatten_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        key="user",
        output_format='line',
        quiet=True,
        clean_items=False
    )

    with open(output_file) as f:
        data = json.load(f)
    assert data == {"name": "Bob"}

def test_unflatten_mode_empty(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("")
    output_file = tmp_path / "output.json"

    unflatten_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='unknown',
        quiet=True,
        clean_items=False
    )

    with open(output_file) as f:
        data = json.load(f)
    assert data == {}

def test_unflatten_mode_toml_list(tmp_path):
    pytest.importorskip("toml")
    input_file = tmp_path / "input.txt"
    input_file.write_text("0 -> a\n1 -> b\n")
    output_file = tmp_path / "output.toml"

    unflatten_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='toml',
        quiet=True,
        clean_items=False
    )

    with open(output_file) as f:
        data = json.load(f)
    assert data == ["a", "b"]

def test_unflatten_mode_yaml_no_pyyaml(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a -> b\n")
    output_file = tmp_path / "output.yaml"

    with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs:
               (exec("raise ImportError") if name == 'yaml' else __import__(name, *args, **kwargs))):
        unflatten_mode(
            input_files=[str(input_file)],
            output_file=str(output_file),
            min_length=1,
            max_length=100,
            process_output=False,
            output_format='yaml',
            quiet=True,
            clean_items=False
        )

    content = output_file.read_text()
    data = json.loads(content)
    assert data == {"a": "b"}

def test_unflatten_mode_toml_no_toml(tmp_path, monkeypatch):
    import multitool
    monkeypatch.setattr(multitool, "_TOMLLIB_AVAILABLE", False)
    monkeypatch.setattr(multitool, "_TOML_AVAILABLE", False)

    input_file = tmp_path / "input.txt"
    input_file.write_text("a -> b\n")
    output_file = tmp_path / "output.toml"

    unflatten_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='toml',
        quiet=True,
        clean_items=False
    )

    with open(output_file) as f:
        data = json.load(f)
    assert data == {"a": "b"}

def test_unflatten_mode_toml_exception(tmp_path, monkeypatch):
    pytest.importorskip("toml")
    import toml
    def raise_exc(*args, **kwargs):
        raise ValueError("Simulated error")
    monkeypatch.setattr(toml, "dump", raise_exc)

    input_file = tmp_path / "input.txt"
    input_file.write_text("a -> b\n")
    output_file = tmp_path / "output.toml"

    unflatten_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='toml',
        quiet=True,
        clean_items=False
    )

    with open(output_file) as f:
        data = json.load(f)
    assert data == {"a": "b"}

def test_unflatten_mode_toml_alternate_available(tmp_path, monkeypatch):
    import multitool
    monkeypatch.setattr(multitool, "_TOMLLIB_AVAILABLE", False)
    monkeypatch.setattr(multitool, "_TOML_AVAILABLE", True)

    input_file = tmp_path / "input.txt"
    input_file.write_text("a -> b\n")
    output_file = tmp_path / "output.toml"

    unflatten_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='toml',
        quiet=True,
        clean_items=False
    )

    import toml
    with open(output_file) as f:
        data = toml.load(f)
    assert data == {"a": "b"}

def test_unflatten_filters_and_lengths(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a.b -> short\na.c -> verylongvalue\n")
    output_file = tmp_path / "output.json"

    unflatten_mode([str(input_file)], str(output_file), min_length=10, max_length=100, process_output=True)
    with open(output_file) as f:
        data = json.load(f)
    assert data == {"a": {"c": "verylongvalue"}}

    unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=5, process_output=True)
    with open(output_file) as f:
        data = json.load(f)
    assert data == {"a": {"b": "short"}}

def test_unflatten_clean_items_disabled(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a.b -> Value With Spaces 123\n")
    output_file = tmp_path / "output.json"

    unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True, clean_items=False)
    with open(output_file) as f:
        data = json.load(f)
    assert data == {"a": {"b": "Value With Spaces 123"}}

def test_unflatten_key_edge_case(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("root -> somevalue\nroot.sub -> other\nother.data -> 123\n")
    output_file = tmp_path / "output.json"

    unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True, key="root")
    with open(output_file) as f:
        data = json.load(f)
    assert data == {"sub": "other"}

def test_unflatten_format_line_resolves_to_json(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a -> b\n")
    output_file = tmp_path / "output.txt"

    unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True, output_format='line')
    with open(output_file) as f:
        data = json.load(f)
    assert data == {"a": "b"}

def test_unflatten_xml_input(tmp_path):
    input_file = tmp_path / "input.xml"
    input_file.write_text("""
<root>
  <pair>
    <left>user.name</left>
    <right>John</right>
  </pair>
  <pair>
    <typo>user.age</typo>
    <correction>30</correction>
  </pair>
  <pair>
    <typo>user.city</typo>
    <correct>New York</correct>
  </pair>
</root>
""")
    output_file = tmp_path / "output.json"
    unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True, clean_items=False)
    with open(output_file) as f:
        data = json.load(f)
    assert data == {"user": {"name": "John", "age": "30", "city": "New York"}}

def test_unflatten_xml_input_malformed(tmp_path):
    input_file = tmp_path / "input.xml"
    input_file.write_text("<root><pair><left>...</root>")
    output_file = tmp_path / "output.json"
    unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True)
    with open(output_file) as f:
        data = json.load(f)
    assert data == {}

def test_unflatten_cli_subprocess_integration(tmp_path):
    input_file = tmp_path / "test_input.txt"
    input_file.write_text("user.name -> John\nuser.age -> 30\n")

    result = subprocess.run(
        ["python3", "multitool.py", "unflatten", str(input_file), "--output-format", "json", "--raw"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data == {"user": {"name": "John", "age": "30"}}

    input_file_list = tmp_path / "test_input_list.txt"
    input_file_list.write_text("items.0 -> apple\nitems.1 -> banana\n")
    result_list = subprocess.run(
        ["python3", "multitool.py", "unflatten", str(input_file_list), "--output-format", "json", "--raw"],
        capture_output=True,
        text=True
    )
    assert result_list.returncode == 0
    data_list = json.loads(result_list.stdout)
    assert data_list == {"items": ["apple", "banana"]}

    input_file_key = tmp_path / "test_input_key.txt"
    input_file_key.write_text("users.0.name -> Alice\nusers.1.name -> Bob\nother.data -> 123\n")
    result_key = subprocess.run(
        ["python3", "multitool.py", "unflatten", str(input_file_key), "-k", "users", "--output-format", "json", "--raw"],
        capture_output=True,
        text=True
    )
    assert result_key.returncode == 0
    data_key = json.loads(result_key.stdout)
    assert data_key == [{"name": "Alice"}, {"name": "Bob"}]

    input_file_ambig = tmp_path / "test_input_ambig.txt"
    input_file_ambig.write_text("root.0 -> a\nroot.01 -> b\n")
    result_ambig = subprocess.run(
        ["python3", "multitool.py", "unflatten", str(input_file_ambig), "--output-format", "json", "--raw"],
        capture_output=True,
        text=True
    )
    assert result_ambig.returncode == 0
    data_ambig = json.loads(result_ambig.stdout)
    assert data_ambig == {"root": {"0": "a", "01": "b"}}
