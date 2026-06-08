import sys
import os
import pytest
import logging
from unittest.mock import patch
from io import StringIO
import multitool

def test_replace_main_positional_three_args(tmp_path, capsys):
    # Tests: replace OLD NEW FILE
    f = tmp_path / "test.txt"
    f.write_text("hello world")

    out = tmp_path / "out.txt"

    test_args = ["multitool.py", "replace", "hello", "hi", str(f), "--output", str(out), "--quiet"]

    with patch("sys.argv", test_args):
        multitool.main()

    assert out.read_text() == "hi world\n"

def test_replace_main_positional_two_args_stdin(tmp_path, capsys, monkeypatch):
    # Tests: replace OLD NEW (reads from stdin)
    # We can't easily capture the output of replace_mode when it writes to stdout in main()
    # unless we check how it's called.
    # Actually replace_mode writes to args.output which defaults to '-' (stdout)

    monkeypatch.setattr("sys.stdin", StringIO("hello world"))

    test_args = ["multitool.py", "replace", "hello", "hi", "--quiet"]

    with patch("sys.argv", test_args):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            multitool.main()
            assert fake_out.getvalue() == "hi world\n"

def test_replace_main_missing_args(tmp_path, caplog):
    # Tests: replace (missing OLD and NEW)
    test_args = ["multitool.py", "replace"]

    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            multitool.main()
        assert excinfo.value.code == 1

    assert "Replace mode requires both OLD and NEW text" in caplog.text

def test_replace_main_missing_one_arg(tmp_path, caplog):
    # Tests: replace OLD (missing NEW)
    test_args = ["multitool.py", "replace", "only_one"]

    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            multitool.main()
        assert excinfo.value.code == 1

    assert "Replace mode requires both OLD and NEW text" in caplog.text

def test_replace_main_with_flags_no_fallback(tmp_path):
    # Tests that if --old and --new are provided, positional args are treated as input files
    f1 = tmp_path / "f1.txt"
    f1.write_text("aaa")

    out = tmp_path / "out.txt"

    test_args = ["multitool.py", "replace", "--old", "a", "--new", "b", str(f1), "--output", str(out), "--quiet"]

    with patch("sys.argv", test_args):
        multitool.main()

    assert out.read_text() == "bbb\n"

def test_replace_main_multiple_files(tmp_path):
    # Tests: replace OLD NEW FILE1 FILE2
    f1 = tmp_path / "f1.txt"
    f1.write_text("aaa")
    f2 = tmp_path / "f2.txt"
    f2.write_text("aba")

    # We'll use in-place to verify both files
    test_args = ["multitool.py", "replace", "a", "c", str(f1), str(f2), "--in-place", "--quiet"]

    with patch("sys.argv", test_args):
        multitool.main()

    assert f1.read_text() == "ccc\n"
    assert f2.read_text() == "cbc\n"
