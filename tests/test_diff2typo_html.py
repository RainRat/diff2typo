import os
import pytest
from diff2typo import format_typos, main


def test_format_typos_html_pairs():
    typos = ["teh -> the", "recieve -> receive"]
    res = format_typos(typos, "html")
    html_str = "\n".join(res)
    assert "<!DOCTYPE html>" in html_str
    assert "<table>" in html_str
    assert "<code>teh</code>" in html_str
    assert "<code>the</code>" in html_str
    assert "<code>recieve</code>" in html_str
    assert "<code>receive</code>" in html_str


def test_format_typos_html_single_items():
    typos = ["teh", "wrold"]
    res = format_typos(typos, "htm")
    html_str = "\n".join(res)
    assert "<!DOCTYPE html>" in html_str
    assert "<ul>" in html_str
    assert "<li><code>teh</code></li>" in html_str
    assert "<li><code>wrold</code></li>" in html_str


def test_format_typos_html_escaping():
    typos = ["a<b -> c&d"]
    res = format_typos(typos, "html")
    html_str = "\n".join(res)
    assert "&lt;" in html_str
    assert "&amp;" in html_str
    assert "a<b" not in html_str


def test_main_html_format_auto_detect(tmp_path, monkeypatch):
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text(
        "diff --git a/file.txt b/file.txt\n"
        "--- a/file.txt\n"
        "+++ b/file.txt\n"
        "@@ -1 +1 @@\n"
        "-teh text\n"
        "+the text\n"
    )
    out_html = tmp_path / "report.html"

    monkeypatch.setattr(
        "sys.argv",
        [
            "diff2typo.py",
            str(diff_file),
            "--output",
            str(out_html),
            "--dictionary",
            "nonexistent.csv",
            "--allowed",
            "nonexistent.csv",
        ],
    )

    main()

    assert out_html.exists()
    content = out_html.read_text()
    assert "<!DOCTYPE html>" in content
    assert "<code>teh</code>" in content
    assert "<code>the</code>" in content


def test_main_html_both_mode(tmp_path, monkeypatch):
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text(
        "diff --git a/file.txt b/file.txt\n"
        "--- a/file.txt\n"
        "+++ b/file.txt\n"
        "@@ -1 +1 @@\n"
        "-teh text\n"
        "+the text\n"
    )
    out_html = tmp_path / "report_both.html"

    monkeypatch.setattr(
        "sys.argv",
        [
            "diff2typo.py",
            str(diff_file),
            "--mode",
            "both",
            "-f",
            "html",
            "--output",
            str(out_html),
            "--dictionary",
            "nonexistent.csv",
            "--allowed",
            "nonexistent.csv",
        ],
    )

    main()

    assert out_html.exists()
    content = out_html.read_text()
    assert "<!DOCTYPE html>" in content
    assert "<h2>Typos</h2>" in content
    assert "<code>teh</code>" in content
