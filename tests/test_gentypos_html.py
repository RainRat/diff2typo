import os
import subprocess
import sys
import tempfile
import gentypos


def test_format_typos_html():
    typo_dict = {"helo": "hello", "<script>": "test & code"}
    html_lines = gentypos.format_typos(typo_dict, "html")
    content = "\n".join(html_lines)

    assert "<!DOCTYPE html>" in content
    assert "<h1>Generated Typos Report</h1>" in content
    assert "<code>helo</code>" in content
    assert "<code>hello</code>" in content
    assert "&lt;script&gt;" in content
    assert "test &amp; code" in content
    assert "<script>" not in content


def test_format_typos_htm_alias():
    typo_dict = {"wrod": "word"}
    htm_lines = gentypos.format_typos(typo_dict, "htm")
    content = "\n".join(htm_lines)

    assert "<!DOCTYPE html>" in content
    assert "<code>wrod</code>" in content
    assert "<code>word</code>" in content


def test_extension_auto_detection_html(tmp_path):
    empty_config = tmp_path / "dummy_config.yaml"
    empty_config.write_text("{}", encoding="utf-8")
    out_file = tmp_path / "output.html"
    cmd = [
        sys.executable,
        "gentypos.py",
        "hello",
        "-c",
        str(empty_config),
        "--no-filter",
        "-o",
        str(out_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    content = out_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "<table>" in content


def test_extension_auto_detection_htm(tmp_path):
    empty_config = tmp_path / "dummy_config.yaml"
    empty_config.write_text("{}", encoding="utf-8")
    out_file = tmp_path / "output.htm"
    cmd = [
        sys.executable,
        "gentypos.py",
        "hello",
        "-c",
        str(empty_config),
        "--no-filter",
        "-o",
        str(out_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    content = out_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "<table>" in content


def test_explicit_format_html_flag(tmp_path):
    out_file = tmp_path / "report.txt"
    cmd = [
        sys.executable,
        "gentypos.py",
        "world",
        "--no-filter",
        "-f",
        "html",
        "-o",
        str(out_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    content = out_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "<h1>Generated Typos Report</h1>" in content
