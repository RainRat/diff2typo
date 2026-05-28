import os
import pytest
import logging
from multitool import replace_mode

def test_replace_literal(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world\nhello universe\n")

    out = tmp_path / "out.txt"
    replace_mode([str(f)], "hello", "hi", str(out), quiet=True)

    assert out.read_text() == "hi world\nhi universe\n"

def test_replace_regex(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("abc 123 def 456")

    out = tmp_path / "out.txt"
    replace_mode([str(f)], r"\d+", "NUM", str(out), quiet=True, use_regex=True)

    assert out.read_text() == "abc NUM def NUM\n"

def test_replace_regex_backreference(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("v1.2.3")

    out = tmp_path / "out.txt"
    replace_mode([str(f)], r"v(\d+)\.(\d+)\.(\d+)", r"version \1", str(out), quiet=True, use_regex=True)

    assert out.read_text() == "version 1\n"

def test_replace_in_place(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("change me")

    replace_mode([str(f)], "change", "changed", "-", quiet=True, in_place="")

    # _write_file_in_place adds a newline if missing
    assert f.read_text() == "changed me\n"

def test_replace_in_place_backup(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("change me")

    replace_mode([str(f)], "change", "changed", "-", quiet=True, in_place=".bak")

    assert f.read_text() == "changed me\n"
    assert (tmp_path / "test.txt.bak").read_text() == "change me"

def test_replace_dry_run(tmp_path, caplog):
    f = tmp_path / "test.txt"
    f.write_text("change me")

    with caplog.at_level(logging.WARNING):
        replace_mode([str(f)], "change", "changed", "-", quiet=True, in_place="", dry_run=True)

    assert f.read_text() == "change me"
    assert "Would make 1 replacement(s)" in caplog.text

def test_replace_limit(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("line1\nline2\nline3\n")

    out = tmp_path / "out.txt"
    replace_mode([str(f)], "line", "item", str(out), quiet=True, limit=2)

    assert out.read_text() == "item1\nitem2\n"
