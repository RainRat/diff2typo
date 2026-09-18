import io
import json
import xml.etree.ElementTree as ET
from unittest.mock import patch
import pytest

import multitool
from multitool import _write_structured_data


class NonSeekableStream(io.TextIOBase):
    """A stream wrapper that explicitly raises io.UnsupportedOperation on seek/truncate."""
    def __init__(self, buffer):
        self.buffer = buffer

    def write(self, s):
        return self.buffer.write(s)

    def seek(self, cookie, whence=0):
        raise io.UnsupportedOperation("seek not supported")

    def truncate(self, size=None):
        raise io.UnsupportedOperation("truncate not supported")


def test_write_structured_data_json(tmp_path):
    output_file = tmp_path / "data.json"
    data = {"key": "value", "list": [1, 2, 3]}
    _write_structured_data(data, str(output_file), output_format="json")

    with open(output_file, "r") as f:
        loaded = json.load(f)
    assert loaded == data


def test_write_structured_data_yaml(tmp_path):
    pytest.importorskip("yaml")
    import yaml

    output_file = tmp_path / "data.yaml"
    data = {"a": "b", "nested": {"c": True}}
    _write_structured_data(data, str(output_file), output_format="yaml")

    with open(output_file, "r") as f:
        loaded = yaml.safe_load(f)
    assert loaded == data


def test_write_structured_data_yaml_import_error(tmp_path):
    output_file = tmp_path / "data.yaml"
    data = {"a": "b"}

    with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs:
               (exec("raise ImportError") if name == "yaml" else __import__(name, *args, **kwargs))):
        _write_structured_data(data, str(output_file), output_format="yaml")

    with open(output_file, "r") as f:
        loaded = json.load(f)
    assert loaded == data


def test_write_structured_data_toml_valid(tmp_path):
    pytest.importorskip("toml")
    import toml

    output_file = tmp_path / "data.toml"
    data = {"title": "Config", "settings": {"enabled": True}}
    _write_structured_data(data, str(output_file), output_format="toml")

    with open(output_file, "r") as f:
        loaded = toml.load(f)
    assert loaded == data


def test_write_structured_data_toml_dump_failure_fallback_non_seekable():
    pytest.importorskip("toml")
    buffer = io.StringIO()
    stream = NonSeekableStream(buffer)

    # Dictionary with object that toml cannot dump
    class CustomObj:
        pass

    data = {"valid_key": "valid_val", "bad_obj": CustomObj()}

    _write_structured_data(data, stream, output_format="toml")

    output_text = buffer.getvalue()
    # Should fall back to json dump without raising UnsupportedOperation or producing partial TOML
    assert "valid_key" in output_text
    assert '"valid_val"' in output_text


def test_write_structured_data_toml_unavailable(monkeypatch):
    monkeypatch.setattr(multitool, "_TOML_AVAILABLE", False)
    buffer = io.StringIO()
    data = {"a": "b"}

    _write_structured_data(data, buffer, output_format="toml")

    output_text = buffer.getvalue()
    loaded = json.loads(output_text)
    assert loaded == {"a": "b"}


def test_write_structured_data_xml(tmp_path):
    output_file = tmp_path / "data.xml"
    data = {"user": {"name": "Alice", "tags": ["admin", "user"]}}
    _write_structured_data(data, str(output_file), output_format="xml", root_tag="config")

    tree = ET.parse(output_file)
    root = tree.getroot()
    assert root.tag == "config"
    user = root.find("user")
    assert user.find("name").text == "Alice"
    item_tags = user.find("tags").findall("item")
    assert len(item_tags) == 2
    assert item_tags[0].text == "admin"
    assert item_tags[1].text == "user"


def test_write_structured_data_fallback_format():
    buffer = io.StringIO()
    data = [1, 2, 3]
    _write_structured_data(data, buffer, output_format="unsupported_format")

    output_text = buffer.getvalue()
    loaded = json.loads(output_text)
    assert loaded == [1, 2, 3]
