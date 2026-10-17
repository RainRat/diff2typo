import io
import sys
import gentypos


def test_load_file_stdin_plain_text(monkeypatch):
    stdin_content = "hello\nworld\npython\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_content))

    words = gentypos.load_file("-")
    assert words == {"hello", "world", "python"}


def test_load_file_stdin_json_structured(monkeypatch):
    json_content = '["apple", "banana", "cherry"]'
    monkeypatch.setattr(sys, "stdin", io.StringIO(json_content))

    words = gentypos.load_file("-")
    assert words == {"apple", "banana", "cherry"}
