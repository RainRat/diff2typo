import pytest
import sys
from gentypos import _load_substitutions_file, main

def test_load_substitutions_file_html_basic(tmp_path):
    html_content = """<!DOCTYPE html>
<html>
<body>
<table>
<thead><tr><th>Typo</th><th>Correction</th></tr></thead>
<tbody>
<tr><td><code>teh</code></td><td><code>the</code></td></tr>
<tr><td><code>f</code></td><td><code>ph</code></td></tr>
</tbody>
</table>
</body>
</html>
"""
    html_file = tmp_path / "substitutions.html"
    html_file.write_text(html_content, encoding="utf-8")

    subs = _load_substitutions_file(str(html_file))
    assert "the" in subs
    assert "teh" in subs["the"]
    assert "ph" in subs
    assert "f" in subs["ph"]


def test_load_substitutions_file_html_reversed_header(tmp_path):
    html_content = """<!DOCTYPE html>
<html>
<body>
<table>
<thead><tr><th>Correction</th><th>Typo</th></tr></thead>
<tbody>
<tr><td><code>the</code></td><td><code>teh</code></td></tr>
<tr><td><code>ph</code></td><td><code>f</code></td></tr>
</tbody>
</table>
</body>
</html>
"""
    html_file = tmp_path / "substitutions_rev.htm"
    html_file.write_text(html_content, encoding="utf-8")

    subs = _load_substitutions_file(str(html_file))
    assert "the" in subs
    assert "teh" in subs["the"]
    assert "ph" in subs
    assert "f" in subs["ph"]


def test_load_substitutions_file_html_unescape_entities(tmp_path):
    html_content = """<table>
<thead><tr><th>Typo</th><th>Correction</th></tr></thead>
<tbody>
<tr><td><code>a&amp;b</code></td><td><code>a_b</code></td></tr>
</tbody>
</table>
"""
    html_file = tmp_path / "substitutions.html"
    html_file.write_text(html_content, encoding="utf-8")

    subs = _load_substitutions_file(str(html_file))
    assert "a_b" in subs
    assert "a&b" in subs["a_b"]


def test_load_substitutions_file_html_latin1_encoding(tmp_path):
    html_bytes = """<table>
<thead><tr><th>Typo</th><th>Correction</th></tr></thead>
<tbody>
<tr><td><code>tëh</code></td><td><code>the</code></td></tr>
</tbody>
</table>
""".encode("latin-1")

    html_file = tmp_path / "substitutions_latin1.html"
    html_file.write_bytes(html_bytes)

    subs = _load_substitutions_file(str(html_file))
    assert "the" in subs
    assert "tëh" in subs["the"]


def test_gentypos_cli_with_html_substitutions(tmp_path, monkeypatch, capsys):
    html_content = """<!DOCTYPE html>
<html>
<body>
<table>
<thead><tr><th>Typo</th><th>Correction</th></tr></thead>
<tbody>
<tr><td><code>f</code></td><td><code>ph</code></td></tr>
</tbody>
</table>
</body>
</html>
"""
    html_file = tmp_path / "subs.html"
    html_file.write_text(html_content, encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gentypos.py",
            "phrase",
            "--substitutions",
            str(html_file),
            "-k",
            "--no-filter",
        ],
    )

    main()
    captured = capsys.readouterr()
    assert "frase" in captured.out or "phrase" in captured.out
