import io
import sys
import pytest
import diff2typo
import gentypos
import typostats


def test_diff2typo_format_typos_toml():
    typos = ["teh -> the", "wrd -> word"]
    result = diff2typo.format_typos(typos, "toml")
    assert result == ['teh = "the"', 'wrd = "word"']


def test_diff2typo_cli_format_toml(monkeypatch, capsys):
    diff_text = """diff --git a/file.txt b/file.txt
--- a/file.txt
+++ b/file.txt
-teh
+the
"""
    monkeypatch.setattr(diff2typo, "_read_diff_sources", lambda files: diff_text)
    monkeypatch.setattr(diff2typo, "read_words_mapping", lambda f, required=False: {})
    monkeypatch.setattr(diff2typo, "read_allowed_words", lambda f: set())
    monkeypatch.setattr(diff2typo, "filter_known_typos", lambda c, typos_tool_path: c)

    test_args = ["diff2typo.py", "-f", "toml", "-"]
    monkeypatch.setattr(sys, "argv", test_args)

    diff2typo.main()
    captured = capsys.readouterr()
    assert 'teh = "the"' in captured.out


def test_gentypos_format_typos_toml():
    mapping = {"teh": "the"}
    result = gentypos.format_typos(mapping, "toml")
    assert result == ['teh = "the"']


def test_gentypos_cli_format_toml(monkeypatch, capsys):
    test_args = ["gentypos.py", "hello", "-f", "toml", "-N"]
    monkeypatch.setattr(sys, "argv", test_args)

    gentypos.main()
    captured = capsys.readouterr()
    assert "=" in captured.out


def test_typostats_generate_report_toml(capsys):
    counts = {("the", "teh"): 3}
    typostats.generate_report(counts, output_format="toml")
    captured = capsys.readouterr()
    assert 'teh = "the"' in captured.out


def test_typostats_cli_format_toml(monkeypatch, capsys):
    monkeypatch.setattr(typostats, "_read_file_lines_robust", lambda filepath: ["teh -> the"])
    test_args = ["typostats.py", "sample.txt", "-f", "toml"]
    monkeypatch.setattr(sys, "argv", test_args)

    typostats.main()
    captured = capsys.readouterr()
    assert 'eh = "he"' in captured.out
