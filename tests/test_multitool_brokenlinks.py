import pytest
from multitool import main
import sys
import os
from unittest.mock import patch
import json

def test_brokenlinks_basic(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
# Heading 1
[Valid Link](#heading-1)
[Broken Link](#non-existent)
[External Link](https://google.com)
""")

    with patch('sys.stdout.isatty', return_value=False):
        sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "arrow"]
        main()

    captured = capsys.readouterr()
    assert "Broken Link" in captured.out
    assert "#non-existent" in captured.out
    assert "Anchor not found" in captured.out
    assert "Heading 1" not in captured.out

def test_brokenlinks_duplicates(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
# Heading
# Heading
[First](#heading)
[Second](#heading-1)
[Third](#heading-2)
""")

    with patch('sys.stdout.isatty', return_value=False):
        sys.argv = ["multitool.py", "brokenlinks", str(md_file)]
        main()

    captured = capsys.readouterr()
    assert "Third" in captured.out
    assert "#heading-2" in captured.out
    assert "First" not in captured.out
    assert "Second" not in captured.out

def test_brokenlinks_files(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    other_file = tmp_path / "other.md"
    other_file.write_text("# Target\n")

    md_file.write_text(f"""
[Valid File](other.md)
[Broken File](missing.md)
[Valid Anchor](other.md#target)
[Broken Anchor](other.md#wrong)
""")

    with patch('sys.stdout.isatty', return_value=False):
        sys.argv = ["multitool.py", "brokenlinks", str(md_file), str(other_file)]
        main()

    captured = capsys.readouterr()
    assert "missing.md" in captured.out
    assert "File not found" in captured.out
    assert "other.md#wrong" in captured.out
    assert "Anchor not found" in captured.out
    assert "Valid File" not in captured.out
    assert "Valid Anchor" not in captured.out

def test_brokenlinks_references(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
[Valid][ref]
[Broken][missing]
[Short][]

[ref]: #target
[Short]: #target
# Target
""")

    with patch('sys.stdout.isatty', return_value=False):
        sys.argv = ["multitool.py", "brokenlinks", str(md_file)]
        main()

    captured = capsys.readouterr()
    assert "missing" in captured.out
    assert "Undefined reference label" in captured.out
    assert "Valid" not in captured.out
    assert "Short" not in captured.out

def test_brokenlinks_limit(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[B1](#1)\n[B2](#2)\n[B3](#3)")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--limit", "1", "--output-format", "json"]
    main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert len(data) == 1
    assert any("B1" in k for k in data.keys())

def test_brokenlinks_on_the_fly_scan(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    target_file = tmp_path / "target.md"
    target_file.write_text("# Hello\n")

    md_file.write_text("[Broken](target.md#wrong)\n[Link](target.md#hello)")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file)]
    main()

    captured = capsys.readouterr()
    assert "target.md#wrong" in captured.out
    assert "Anchor not found in target.md: #wrong" in captured.out
    assert "target.md#hello" not in captured.out

def test_brokenlinks_empty_link(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Empty]()\n[Empty2](?query)\n[Skip](?query#anchor)")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file)]
    main()

    captured = capsys.readouterr()
    assert "Empty link" in captured.out
    assert "Skip" not in captured.out

def test_brokenlinks_non_markdown(tmp_path, caplog):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("[Link](#anchor)")

    sys.argv = ["multitool.py", "brokenlinks", str(txt_file)]
    with caplog.at_level("INFO"):
        main()

    assert "Found 0 broken links" in caplog.text

def test_brokenlinks_stdin(caplog):
    with patch('multitool._read_file_lines_robust', return_value=["[Link](#anchor)"]):
        sys.argv = ["multitool.py", "brokenlinks", "-"]
        with caplog.at_level("INFO"):
            main()

    assert "Found 0 broken links" in caplog.text
