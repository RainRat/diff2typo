import pytest
import json
import os
import xml.etree.ElementTree as ET
from multitool import unflatten_mode

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
    # _extract_pairs for TOML only yields top-level simple key-values if 'replacements' is not present
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
        clean_items=True # This will strip non-letters
    )

    with open(output_file) as f:
        data = json.load(f)
    # 'valid' is 5 letters.
    # '123!@#' cleaned is empty string, which is < min_length 2.
    # 'toolong' is 7 letters, > max_length 5.
    assert data == {"a": "valid"}

def test_unflatten_mode_xml_numeric_keys_fixed(tmp_path):
    """
    This test verifies that numeric keys are handled by prefixing with underscore.
    XML tags cannot start with a digit.
    We use a non-continuous sequence so dict_to_lists doesn't turn it into a list.
    """
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
    """
    Tests how lists are handled in XML.
    dict_to_lists converts numeric key dicts to lists.
    The current implementation of build_xml handles lists by using 'item' tags.
    """
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
        output_format='line', # Test line -> json fallback
        quiet=True,
        clean_items=False
    )

    with open(output_file) as f:
        data = json.load(f)
    # 'user -> Alice' is skipped because p == key
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
        output_format='unknown', # Test fallback
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

    # Since it's a list, it should fallback to JSON for TOML output
    with open(output_file) as f:
        data = json.load(f)
    assert data == ["a", "b"]

def test_unflatten_mode_yaml_no_pyyaml(tmp_path, monkeypatch):
    import sys
    from unittest.mock import patch

    input_file = tmp_path / "input.txt"
    input_file.write_text("a -> b\n")
    output_file = tmp_path / "output.yaml"

    # Patching 'yaml' module inside multitool to raise ImportError when imported
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

    # Should fallback to JSON
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

    # Should fallback to JSON
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

    # Should fallback to JSON
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
