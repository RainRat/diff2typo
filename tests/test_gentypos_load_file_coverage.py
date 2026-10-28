import sys
import pytest
import gentypos

def test_load_file_latin1_encoding(tmp_path):
    file_path = tmp_path / "latin1_file.txt"
    # Write bytes that are invalid UTF-8 (0xe9 in Latin-1 is 'é')
    file_path.write_bytes(b"caf\xe9\nhello\n")

    words = gentypos.load_file(str(file_path))
    assert "hello" in words

def test_load_file_empty_content(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   \n\t  \n")

    words = gentypos.load_file(str(file_path))
    assert words == set()

def test_load_file_invalid_json_fallback(tmp_path):
    file_path = tmp_path / "invalid.json"
    file_path.write_text("[invalid json content...")

    words = gentypos.load_file(str(file_path))
    assert "content" in words or "invalid" in words

def test_load_file_yaml_exception_fallback(tmp_path, monkeypatch):
    file_path = tmp_path / "invalid.yaml"
    file_path.write_text("key: : invalid yaml")

    monkeypatch.setattr(gentypos, "_YAML_AVAILABLE", True)
    words = gentypos.load_file(str(file_path))
    assert isinstance(words, set)

def test_load_file_toml_tomllib_and_toml_module_fallbacks(tmp_path, monkeypatch):
    file_path = tmp_path / "sample.toml"
    file_path.write_text("word = 'hello'\n")

    # Test when tomllib is NOT available but _TOML_AVAILABLE is True
    monkeypatch.setattr(gentypos, "_TOMLLIB_AVAILABLE", False)
    monkeypatch.setattr(gentypos, "_TOML_AVAILABLE", True)

    class DummyToml:
        @staticmethod
        def loads(content):
            return {"word": "hello"}

    monkeypatch.setitem(sys.modules, "toml", DummyToml)

    words = gentypos.load_file(str(file_path))
    assert "hello" in words

def test_load_file_toml_exception_fallback(tmp_path, monkeypatch):
    file_path = tmp_path / "invalid.toml"
    file_path.write_text("invalid toml [[")

    monkeypatch.setattr(gentypos, "_TOMLLIB_AVAILABLE", True)
    words = gentypos.load_file(str(file_path))
    assert isinstance(words, set)

def test_load_file_html_headers_filtered_and_exception(tmp_path):
    file_path = tmp_path / "report.html"
    file_path.write_text("<table><tr><td>Typo</td><td>Correction</td><td>word</td><td>correct</td><td>apple</td></tr></table>")

    words = gentypos.load_file(str(file_path))
    assert "apple" in words
    assert "typo" not in words
    assert "correction" not in words

def test_load_file_csv_exception_fallback(tmp_path, monkeypatch):
    file_path = tmp_path / "corrupt.csv"
    file_path.write_text("a,b,c\n")

    def mock_reader(*args, **kwargs):
        raise ValueError("Simulated CSV parsing error")

    monkeypatch.setattr("csv.reader", mock_reader)

    # Should catch Exception and proceed to markdown/plain text fallback
    words = gentypos.load_file(str(file_path))
    assert "a" in words or "b" in words
