
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Ensure we can import multitool
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    monkeypatch.setattr(multitool, "tqdm", lambda iterable=None, *_, **__: iterable if iterable is not None else MagicMock())

def test_unflatten_filters(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a.b -> short\na.c -> verylongvalue\n")
    output_file = tmp_path / "output.json"

    # Test min_length
    multitool.unflatten_mode([str(input_file)], str(output_file), min_length=10, max_length=100, process_output=True)
    with open(output_file) as f:
        data = json.load(f)
    assert data == {"a": {"c": "verylongvalue"}}

    # Test max_length
    multitool.unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=5, process_output=True)
    with open(output_file) as f:
        data = json.load(f)
    assert data == {"a": {"b": "short"}}

def test_unflatten_clean_items_false(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a.b -> Value With Spaces 123\n")
    output_file = tmp_path / "output.json"

    multitool.unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True, clean_items=False)
    with open(output_file) as f:
        data = json.load(f)
    assert data == {"a": {"b": "Value With Spaces 123"}}

def test_unflatten_key_edge_case(tmp_path):
    input_file = tmp_path / "input.txt"
    # root -> matches key exactly (line 1808)
    # root.sub -> matches prefix (line 1806)
    # other.data -> does not match prefix or key (line 1813)
    input_file.write_text("root -> somevalue\nroot.sub -> other\nother.data -> 123\n")
    output_file = tmp_path / "output.json"

    # When key="root", it should skip "root -> somevalue" and "other.data"
    multitool.unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True, key="root")
    with open(output_file) as f:
        data = json.load(f)
    assert data == {"sub": "other"}

def test_unflatten_yaml_output(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a.b -> c\n")
    output_file = tmp_path / "output.yaml"

    multitool.unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True, output_format='yaml')
    content = output_file.read_text()
    import yaml
    data = yaml.safe_load(content)
    assert data == {"a": {"b": "c"}}

def test_unflatten_toml_output(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a.b -> c\n")
    output_file = tmp_path / "output.toml"

    multitool.unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True, output_format='toml')
    content = output_file.read_text()
    import toml
    data = toml.loads(content)
    assert data == {"a": {"b": "c"}}

def test_unflatten_xml_output(tmp_path):
    input_file = tmp_path / "input.txt"
    # Note: since key='data' is used, input must have 'data.' prefix to be included
    input_file.write_text("data.user.name -> John\ndata.user.roles.0 -> admin\ndata.user.roles.1 -> user\n")
    output_file = tmp_path / "output.xml"

    multitool.unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True, output_format='xml', key='data')
    content = output_file.read_text()
    assert "<data>" in content
    assert "<user>" in content
    assert "<name>john</name>" in content
    assert "<item>admin</item>" in content
    assert "<item>user</item>" in content

def test_unflatten_xml_no_key(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a -> b\n")
    output_file = tmp_path / "output.xml"

    multitool.unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True, output_format='xml')
    content = output_file.read_text()
    assert "<root>" in content
    assert "<a>b</a>" in content

def test_unflatten_format_line_resolves_to_json(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a -> b\n")
    output_file = tmp_path / "output.txt"

    multitool.unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True, output_format='line')
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
    multitool.unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True, clean_items=False)
    with open(output_file) as f:
        data = json.load(f)
    assert data == {"user": {"name": "John", "age": "30", "city": "New York"}}

def test_unflatten_xml_input_malformed(tmp_path):
    input_file = tmp_path / "input.xml"
    input_file.write_text("<root><pair><left>...</root>") # Malformed
    output_file = tmp_path / "output.json"
    # Should log error and result in empty dict
    multitool.unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True)
    with open(output_file) as f:
        data = json.load(f)
    assert data == {}

def test_unflatten_toml_available_but_not_tomllib(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a -> b\n")
    output_file = tmp_path / "output.toml"

    # Force _TOMLLIB_AVAILABLE to False but _TOML_AVAILABLE to True (line 1874)
    with patch('multitool._TOMLLIB_AVAILABLE', False), \
         patch('multitool._TOML_AVAILABLE', True), \
         patch('toml.dump') as mock_dump:
        multitool.unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True, output_format='toml')
        assert mock_dump.called

def test_unflatten_toml_dump_error(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a -> b\n")
    output_file = tmp_path / "output.txt"

    with patch('toml.dump', side_effect=Exception("Dump error")):
        multitool.unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True, output_format='toml')

    # Should fallback to JSON (line 1880)
    with open(output_file) as f:
        data = json.load(f)
    assert data == {"a": "b"}

def test_unflatten_empty_root(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("")
    output_file = tmp_path / "output.json"

    multitool.unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True)
    with open(output_file) as f:
        data = json.load(f)
    assert data == {} # Hits line 1837

def test_unflatten_yaml_import_error(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a -> b\n")
    output_file = tmp_path / "output.txt"

    with patch.dict('sys.modules', {'yaml': None}):
        multitool.unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True, output_format='yaml')

    # Should fallback to JSON
    with open(output_file) as f:
        data = json.load(f)
    assert data == {"a": "b"}

def test_unflatten_toml_import_error(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a -> b\n")
    output_file = tmp_path / "output.txt"

    # Force both _TOMLLIB_AVAILABLE and _TOML_AVAILABLE to False to trigger fallback
    with patch('multitool._TOMLLIB_AVAILABLE', False), \
         patch('multitool._TOML_AVAILABLE', False):
        multitool.unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True, output_format='toml')

    # Should fallback to JSON
    with open(output_file) as f:
        data = json.load(f)
    assert data == {"a": "b"}

def test_unflatten_toml_not_dict_fallback(tmp_path):
    # If result_data is not a dict (e.g. it's a list), TOML cannot represent it as root.
    input_file = tmp_path / "input.txt"
    input_file.write_text("0 -> a\n1 -> b\n")
    output_file = tmp_path / "output.txt"

    multitool.unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True, output_format='toml')

    # Should fallback to JSON because root is a list
    with open(output_file) as f:
        data = json.load(f)
    assert data == ["a", "b"]

def test_unflatten_unknown_format_fallback(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a -> b\n")
    output_file = tmp_path / "output.txt"

    multitool.unflatten_mode([str(input_file)], str(output_file), min_length=1, max_length=100, process_output=True, output_format='unknown')

    # Should fallback to JSON
    with open(output_file) as f:
        data = json.load(f)
    assert data == {"a": "b"}
