import pytest
from multitool import main
import sys
import os
from unittest.mock import patch

def test_brokenlinks_basic(tmp_path, capsys):
    # Create a markdown file with one good link and one broken internal anchor
    md_file = tmp_path / "test.md"
    md_file.write_text("# Hello\n\n[Good](#hello)\n[Bad](#world)")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "line"]
    main()

    captured = capsys.readouterr()
    assert "Anchor not found: #world" in captured.out
    assert "test.md:4 [Bad]" in captured.out
    assert "Good" not in captured.out

def test_brokenlinks_reference_style(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Ref][label]\n\n[label]: #good\n# Good\n[Broken][missing]")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "line"]
    main()

    captured = capsys.readouterr()
    assert "Undefined reference label: missing" in captured.out
    assert "test.md:5 [Broken]" in captured.out

def test_brokenlinks_duplicate_anchors(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    # Second 'Hello' should be 'hello-1'
    md_file.write_text("# Hello\n# Hello\n[First](#hello)\n[Second](#hello-1)\n[Third](#hello-2)")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "line"]
    main()

    captured = capsys.readouterr()
    assert "Anchor not found: #hello-2" in captured.out
    assert "First" not in captured.out
    assert "Second" not in captured.out

def test_brokenlinks_cross_file(tmp_path, capsys):
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    file1 = dir1 / "file1.md"
    file2 = dir1 / "file2.md"

    file1.write_text("# File 1\n[Link to file 2](file2.md)\n[Link to file 2 anchor](file2.md#target)\n[Link to non-existent](missing.md)")
    file2.write_text("# Target")

    # Run on file1 only. It should find file2 because it's a local reference.
    sys.argv = ["multitool.py", "brokenlinks", str(file1), "--output-format", "line"]
    main()

    captured = capsys.readouterr()
    assert "File not found: missing.md" in captured.out
    assert "File 1" not in captured.out
    assert "file2.md" not in captured.out
    assert "target" not in captured.out

def test_brokenlinks_cross_file_broken_anchor(tmp_path, capsys):
    file1 = tmp_path / "file1.md"
    file2 = tmp_path / "file2.md"

    file1.write_text("[Link](file2.md#wrong)")
    file2.write_text("# Right")

    # Pass BOTH files so file2 is in the initial anchor_map
    sys.argv = ["multitool.py", "brokenlinks", str(file1), str(file2), "--output-format", "line"]
    main()

    captured = capsys.readouterr()
    assert "Anchor not found in file2.md: #wrong" in captured.out

def test_brokenlinks_reference_style_shortcut(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    # Testing [text][] style
    md_file.write_text("[Label][]\n\n[label]: #target\n# Target")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "line"]
    main()

    captured = capsys.readouterr()
    assert "Anchor not found" not in captured.out
    assert "label" not in captured.out.lower()

def test_brokenlinks_on_the_fly_scanning(tmp_path, capsys):
    # Test scanning a file that was NOT in the initial input_files list
    file1 = tmp_path / "file1.md"
    file2 = tmp_path / "file2.md"

    file1.write_text("[Link](file2.md#target)")
    file2.write_text("# Target")

    # Only file1 is passed to multitool
    sys.argv = ["multitool.py", "brokenlinks", str(file1), "--output-format", "line"]
    main()

    captured = capsys.readouterr()
    assert "Anchor not found" not in captured.out

def test_brokenlinks_arrow_format(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Broken](#missing)")

    # Force color off for predictable output
    with patch('multitool._should_enable_color', return_value=False):
        sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "arrow"]
        main()

    captured = capsys.readouterr()
    assert "Location" in captured.out
    assert "Text" in captured.out
    assert "URL" in captured.out
    assert "Reason" in captured.out
    assert "│" in captured.out
    assert "test.md:1" in captured.out
    assert "BROKEN LINKS ANALYSIS" in captured.out

def test_brokenlinks_limit(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[B1](#m1)\n[B2](#m2)\n[B3](#m3)")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "line", "--limit", "2"]
    main()

    captured = capsys.readouterr()
    # Should only show 2 broken links
    assert captured.out.count("Anchor not found") == 2

def test_brokenlinks_non_markdown_ignored(tmp_path, capsys):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("[Broken](#missing)")

    # Run with --quiet so we don't get logging noise, but we should check if it actually does anything
    sys.argv = ["multitool.py", "brokenlinks", str(txt_file), "--output-format", "line"]
    main()

    captured = capsys.readouterr()
    # The output should be empty because no markdown files were processed
    assert captured.out == ""

def test_brokenlinks_empty_link(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Empty]()")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "line"]
    main()

    captured = capsys.readouterr()
    assert "Empty link" in captured.out

def test_brokenlinks_external_ignored(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[External](https://google.com/404)")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "line"]
    main()

    captured = capsys.readouterr()
    assert "https://google.com/404" not in captured.out
