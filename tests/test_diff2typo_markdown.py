import os
import sys
import tempfile
import pytest

from diff2typo import format_typos, main

def test_format_typos_markdown_pairs():
    typos = ["teh -> the", "fiel -> file"]
    result = format_typos(typos, "markdown")
    expected = [
        "| Typo | Correction |",
        "| :--- | :--- |",
        "| `teh` | `the` |",
        "| `fiel` | `file` |"
    ]
    assert result == expected

def test_format_typos_markdown_singles():
    typos = ["teh", "fiel"]
    result = format_typos(typos, "md")
    expected = [
        "- `teh`",
        "- `fiel`"
    ]
    assert result == expected

def test_format_typos_markdown_mixed():
    typos = ["teh -> the", "fiel"]
    result = format_typos(typos, "markdown")
    expected = [
        "| Typo | Correction |",
        "| :--- | :--- |",
        "| `teh` | `the` |",
        "",
        "- `fiel`"
    ]
    assert result == expected

def test_main_markdown_auto_detect_extension(monkeypatch, tmp_path):
    diff_content = (
        "diff --git a/test.txt b/test.txt\n"
        "--- a/test.txt\n"
        "+++ b/test.txt\n"
        "@@ -1 +1 @@\n"
        "-teh\n"
        "+the\n"
    )
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text(diff_content, encoding="utf-8")

    output_file = tmp_path / "output.md"

    test_args = ["diff2typo.py", str(diff_file), "-o", str(output_file), "-q"]
    monkeypatch.setattr(sys, "argv", test_args)

    main()

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "| Typo | Correction |" in content
    assert "| `teh` | `the` |" in content

def test_main_markdown_both_mode(monkeypatch, tmp_path):
    diff_content = (
        "diff --git a/test.txt b/test.txt\n"
        "--- a/test.txt\n"
        "+++ b/test.txt\n"
        "@@ -1 +1 @@\n"
        "-teh\n"
        "+the\n"
    )
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text(diff_content, encoding="utf-8")

    dict_file = tmp_path / "words.csv"
    dict_file.write_text("teh,the\n", encoding="utf-8")

    output_file = tmp_path / "output.txt"

    test_args = [
        "diff2typo.py",
        str(diff_file),
        "-o", str(output_file),
        "-f", "markdown",
        "-M", "both",
        "-d", str(dict_file),
        "-q"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    main()

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "### Typos" in content or "### Corrections" in content
    assert "| Typo | Correction |" in content
