import pytest
from unittest.mock import patch
import gentypos


def test_load_file_latin1_encoding_fallback(tmp_path):
    file_path = tmp_path / "latin1_sample.txt"
    file_path.write_bytes("café résumé".encode("latin-1"))

    words = gentypos.load_file(str(file_path))
    assert isinstance(words, set)


def test_load_file_empty_content_early_return(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   \n\t  ", encoding="utf-8")

    words = gentypos.load_file(str(file_path))
    assert words == set()


def test_load_file_invalid_json_fallback(tmp_path):
    file_path = tmp_path / "invalid.json"
    file_path.write_text("{invalid json content", encoding="utf-8")

    words = gentypos.load_file(str(file_path))
    assert "invalid" in words or "content" in words or "json" in words


def test_load_file_invalid_yaml_fallback(tmp_path):
    file_path = tmp_path / "invalid.yaml"
    file_path.write_text(":\n:\n: invalid yaml", encoding="utf-8")

    words = gentypos.load_file(str(file_path))
    assert "invalid" in words or "yaml" in words


def test_load_file_toml_fallback_and_invalid_toml(tmp_path):
    file_path = tmp_path / "sample.toml"
    file_path.write_text('key = "value"\n', encoding="utf-8")

    with patch("gentypos._TOMLLIB_AVAILABLE", False), patch("gentypos._TOML_AVAILABLE", True):
        words = gentypos.load_file(str(file_path))
        assert "key" in words and "value" in words

    invalid_toml_path = tmp_path / "invalid.toml"
    invalid_toml_path.write_text("[invalid toml section header", encoding="utf-8")

    words_invalid = gentypos.load_file(str(invalid_toml_path))
    assert "invalid" in words_invalid or "section" in words_invalid or "toml" in words_invalid


def test_load_file_invalid_csv_exception_fallback(tmp_path):
    file_path = tmp_path / "invalid.csv"
    file_path.write_bytes(b"val1,\x00val2\n")

    words = gentypos.load_file(str(file_path))
    assert "val1" in words
