import json
import pytest
from unittest.mock import patch
import multitool
from multitool import _yield_structured_docs, main, to_case
import sys

def test_yield_structured_docs_empty_file(tmp_path):
    f = tmp_path / "empty.json"
    f.write_text("   ")
    results = list(_yield_structured_docs(str(f)))
    assert results == []

def test_yield_structured_docs_invalid_jsonl(tmp_path):
    content = '{"a": 1}\n{invalid_json}\n{"b": 2}'
    f = tmp_path / "test.json"
    f.write_text(content)
    results = list(_yield_structured_docs(str(f)))
    assert results == [{"a": 1}, {"b": 2}]

def test_yield_structured_docs_yaml_exception(tmp_path):
    f = tmp_path / "test.yaml"
    f.write_text("key: value")
    with patch("yaml.safe_load_all", side_effect=Exception("YAML error")):
        results = list(_yield_structured_docs(str(f)))
        assert results == []

def test_yield_structured_docs_toml_exception(tmp_path):
    f = tmp_path / "test.toml"
    f.write_text("key = value")
    if multitool._TOMLLIB_AVAILABLE:
        patch_path = "tomllib.loads"
    elif multitool._TOML_AVAILABLE:
        patch_path = "toml.loads"
    else:
        pytest.skip("No TOML library available")

    with patch(patch_path, side_effect=Exception("TOML error")):
        results = list(_yield_structured_docs(str(f)))
        assert results == []

def test_yield_structured_docs_fallback_all_fail(tmp_path):
    f = tmp_path / "test.unknown"
    f.write_text("not json or yaml")
    with patch("json.loads", side_effect=json.JSONDecodeError("msg", "doc", 0)), \
         patch("yaml.safe_load_all", side_effect=Exception("YAML error")):
        results = list(_yield_structured_docs(str(f)))
        assert results == []

def test_yield_structured_docs_fallback_yaml(tmp_path):
    f = tmp_path / "test.unknown"
    f.write_text("key: value")
    results = list(_yield_structured_docs(str(f)))
    assert results == [{"key": "value"}]

def test_to_case_no_words():
    assert to_case("", "snake") == ""
    assert to_case("!!!", "camel") == "!!!"

def test_main_help_subcommand():
    test_args = ["multitool.py", "help", "scan"]
    with patch.object(sys, 'argv', test_args), \
         patch("multitool.show_mode_help") as mock_help:
        assert main() is None
        mock_help.assert_called_once()
