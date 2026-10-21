import pytest
import gentypos

def test_load_file_latin1_encoding(tmp_path):
    file_path = tmp_path / "latin1_test.txt"
    file_path.write_bytes("caf\xe9\n".encode("latin-1"))

    words = gentypos.load_file(str(file_path))
    assert isinstance(words, set)

def test_load_file_empty_content(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")

    words = gentypos.load_file(str(file_path))
    assert words == set()

def test_load_file_invalid_json_fallback(tmp_path):
    file_path = tmp_path / "invalid.json"
    file_path.write_text("{invalid json}")

    words = gentypos.load_file(str(file_path))
    assert "invalid" in words
    assert "json" in words

def test_load_file_invalid_yaml_fallback(tmp_path):
    file_path = tmp_path / "invalid.yaml"
    file_path.write_text(":\n  invalid: yaml: [")

    words = gentypos.load_file(str(file_path))
    assert "invalid" in words

def test_load_file_invalid_toml_fallback(tmp_path):
    file_path = tmp_path / "invalid.toml"
    file_path.write_text("invalid = toml = [")

    words = gentypos.load_file(str(file_path))
    assert "invalid" in words

def test_load_file_invalid_csv_fallback(tmp_path):
    file_path = tmp_path / "invalid.csv"
    file_path.write_text("a\x00b,c")

    words = gentypos.load_file(str(file_path))
    assert isinstance(words, set)

def test_load_file_toml_fallback_module(tmp_path, monkeypatch):
    monkeypatch.setattr(gentypos, "_TOMLLIB_AVAILABLE", False)
    monkeypatch.setattr(gentypos, "_TOML_AVAILABLE", True)

    file_path = tmp_path / "config.toml"
    file_path.write_text('key = "value"\n')

    words = gentypos.load_file(str(file_path))
    assert "key" in words
    assert "value" in words
