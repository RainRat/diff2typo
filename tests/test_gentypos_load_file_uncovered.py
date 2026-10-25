import os
import sys
import pytest
import gentypos


def test_load_file_latin1_encoding_fallback(tmp_path):
    file_path = tmp_path / "latin1.txt"
    # Write bytes with non-UTF-8 Latin-1 character (e.g., 0xe9 for 'é')
    file_path.write_bytes(b"caf\xe9\nhello\n")

    words = gentypos.load_file(str(file_path))
    assert "hello" in words


def test_load_file_empty_content(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   \n\t\n", encoding="utf-8")

    words = gentypos.load_file(str(file_path))
    assert words == set()


def test_load_file_json_exception_fallback(tmp_path):
    file_path = tmp_path / "bad.json"
    file_path.write_text("{ invalid jsoncontent }", encoding="utf-8")

    # Should catch JSONDecodeError and fall back to plain text line parsing
    words = gentypos.load_file(str(file_path))
    assert isinstance(words, set)


def test_load_file_yaml_exception_fallback(tmp_path):
    file_path = tmp_path / "bad.yaml"
    file_path.write_text(":\n  - : : bad yaml", encoding="utf-8")

    words = gentypos.load_file(str(file_path))
    assert isinstance(words, set)


def test_load_file_toml_fallback(tmp_path, monkeypatch):
    file_path = tmp_path / "config.toml"
    file_path.write_text("key = 'value'\nwords = ['apple', 'banana']\n", encoding="utf-8")

    # Monkeypatch _TOMLLIB_AVAILABLE to False and _TOML_AVAILABLE to True
    monkeypatch.setattr(gentypos, "_TOMLLIB_AVAILABLE", False)
    monkeypatch.setattr(gentypos, "_TOML_AVAILABLE", True)

    words = gentypos.load_file(str(file_path))
    assert "apple" in words
    assert "banana" in words


def test_load_file_html_table_header_filtering(tmp_path):
    file_path = tmp_path / "report.html"
    html_content = """
    <table>
        <tr><th>Typo</th><th>Correction</th><th>Word</th><th>ValidWord</th></tr>
        <tr><td>teh</td><td>the</td><td>example</td><td>testing</td></tr>
    </table>
    """
    file_path.write_text(html_content, encoding="utf-8")

    words = gentypos.load_file(str(file_path))
    # Header words ('typo', 'correction', 'word') should be filtered out
    assert "typo" not in words
    assert "correction" not in words
    assert "word" not in words
    assert "validword" in words
    assert "teh" in words
    assert "the" in words


def test_load_file_csv_exception_fallback(tmp_path, monkeypatch):
    file_path = tmp_path / "data.csv"
    file_path.write_text("word1,word2\nfoo,bar\n", encoding="utf-8")

    # Mock csv.reader to raise an exception during iteration
    def bad_reader(*args, **kwargs):
        raise RuntimeError("CSV error")

    monkeypatch.setattr(gentypos.csv, "reader", bad_reader)

    # Should catch exception and fall back to plain text line parsing
    words = gentypos.load_file(str(file_path))
    assert isinstance(words, set)
