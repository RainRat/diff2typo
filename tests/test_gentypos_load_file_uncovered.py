import sys
import json
import pytest
import gentypos


def test_load_file_latin1_fallback(tmp_path):
    filepath = tmp_path / "latin1_sample.txt"
    filepath.write_bytes("hello\nworld\ncaf\xe9".encode("latin-1"))

    result = gentypos.load_file(str(filepath))
    assert "hello" in result
    assert "world" in result


def test_load_file_empty_content(tmp_path):
    filepath = tmp_path / "empty.txt"
    filepath.write_text("   \n\t  \n")

    result = gentypos.load_file(str(filepath))
    assert result == set()


def test_load_file_json_exception_fallback(tmp_path):
    filepath = tmp_path / "invalid.json"
    filepath.write_text("{invalid json content word1 word2")

    result = gentypos.load_file(str(filepath))
    assert "word1" in result
    assert "word2" in result


def test_load_file_yaml_exception_fallback(tmp_path):
    filepath = tmp_path / "invalid.yaml"
    filepath.write_text("key: : invalid yaml : : word1 word2")

    result = gentypos.load_file(str(filepath))
    assert "word1" in result or "key:" in result or len(result) > 0


def test_load_file_toml_fallback_and_exception(tmp_path, monkeypatch):
    filepath = tmp_path / "sample.toml"
    filepath.write_text("[section]\nkey = \"value1\"\n")

    monkeypatch.setattr(gentypos, "_TOMLLIB_AVAILABLE", False)
    monkeypatch.setattr(gentypos, "_TOML_AVAILABLE", True)

    class DummyToml:
        @staticmethod
        def loads(content):
            return {"section": {"key": "value1"}}

    monkeypatch.setitem(sys.modules, "toml", DummyToml)

    result = gentypos.load_file(str(filepath))
    assert "value1" in result

    invalid_toml = tmp_path / "invalid.toml"
    invalid_toml.write_text("this is invalid toml = = = ")

    class FailingToml:
        @staticmethod
        def loads(content):
            raise ValueError("Invalid TOML syntax")

    monkeypatch.setitem(sys.modules, "toml", FailingToml)

    result_invalid = gentypos.load_file(str(invalid_toml))
    assert "invalid" in result_invalid or "toml" in result_invalid


def test_load_file_csv_exception_fallback(tmp_path, monkeypatch):
    filepath = tmp_path / "broken.csv"
    filepath.write_text("col1,col2\nval1,val2\n")

    def failing_reader(*args, **kwargs):
        raise ValueError("CSV reading failure")

    monkeypatch.setattr(gentypos.csv, "reader", failing_reader)

    result = gentypos.load_file(str(filepath))
    assert "val1" in result or "col1" in result
