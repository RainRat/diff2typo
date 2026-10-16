import tempfile
import os
import sys
import pytest
from gentypos import load_file


def test_load_file_unicode_decode_error_latin1_fallback(tmp_path):
    binary_file = tmp_path / "latin1.txt"
    # Write non-UTF-8 bytes containing ASCII words (e.g. "hello \xe9 world\n")
    binary_file.write_bytes(b"hello \xe9 world\n")
    words = load_file(str(binary_file))
    assert "hello" in words
    assert "world" in words


def test_load_file_empty_file_returns_empty_set(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   \n\t\n", encoding="utf-8")
    words = load_file(str(empty_file))
    assert words == set()


def test_load_file_invalid_json_fallback(tmp_path):
    json_file = tmp_path / "invalid.json"
    json_file.write_text("{invalid json content:", encoding="utf-8")
    words = load_file(str(json_file))
    assert isinstance(words, set)


def test_load_file_invalid_yaml_fallback(tmp_path):
    yaml_file = tmp_path / "invalid.yaml"
    yaml_file.write_text(":\n  - invalid: [yaml", encoding="utf-8")
    words = load_file(str(yaml_file))
    assert isinstance(words, set)


def test_load_file_invalid_toml_fallback(tmp_path):
    toml_file = tmp_path / "invalid.toml"
    toml_file.write_text("[table\ninvalid = toml =", encoding="utf-8")
    words = load_file(str(toml_file))
    assert isinstance(words, set)


def test_load_file_toml_fallback_to_toml_package(tmp_path, monkeypatch):
    toml_file = tmp_path / "valid.toml"
    toml_file.write_text('word = "testtoml"\n', encoding="utf-8")
    monkeypatch.setattr("gentypos._TOMLLIB_AVAILABLE", False)
    monkeypatch.setattr("gentypos._TOML_AVAILABLE", True)
    words = load_file(str(toml_file))
    assert "testtoml" in words


def test_load_file_html_header_filtering(tmp_path):
    html_file = tmp_path / "table.html"
    html_content = """
    <table>
      <tr><th>typo</th><th>correction</th><th>word</th><th>correct</th><th>left</th><th>right</th><th>validword</th></tr>
    </table>
    """
    html_file.write_text(html_content, encoding="utf-8")
    words = load_file(str(html_file))
    assert "validword" in words
    assert "typo" not in words
    assert "correction" not in words
    assert "word" not in words
    assert "correct" not in words
    assert "left" not in words
    assert "right" not in words


def test_load_file_csv_exception_fallback(tmp_path, monkeypatch):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2\nval1,val2\n", encoding="utf-8")

    def mock_csv_reader(*args, **kwargs):
        raise ValueError("Simulated CSV error")

    monkeypatch.setattr("csv.reader", mock_csv_reader)
    words = load_file(str(csv_file))
    assert isinstance(words, set)


def test_load_file_csv_header_keywords_skipping(tmp_path):
    csv_file = tmp_path / "header.csv"
    csv_content = "typo,correction,word 1,word 2,left,right,before,after,key,val,value\napple,banana\n"
    csv_file.write_text(csv_content, encoding="utf-8")
    words = load_file(str(csv_file))
    assert "apple" in words
    assert "banana" in words
    assert "typo" not in words
    assert "correction" not in words
    assert "value" not in words


def test_load_file_markdown_table_header_filtering(tmp_path):
    md_file = tmp_path / "table.md"
    md_content = """
| typo | correction | word | correct | sample |
| :--- | :--- | :--- | :--- | :--- |
| err | fix | item | right | testval |
"""
    md_file.write_text(md_content, encoding="utf-8")
    words = load_file(str(md_file))
    assert "testval" in words
    assert "err" in words
    assert "typo" not in words
    assert "correction" not in words
    assert "word" not in words
    assert "correct" not in words
