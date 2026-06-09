import os
import pytest
from multitool import main
import sys
from io import StringIO

def test_links_extraction(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
# Test
This is a [link](https://example.com).
Here is an image: ![alt text](image.png)
Another [link with spaces](https://google.com).
    """)

    # Test default (extract text)
    sys.argv = ["multitool.py", "links", str(md_file)]
    main()
    captured = capsys.readouterr()
    assert "link" in captured.out
    assert "alttext" in captured.out
    assert "linkwithspaces" in captured.out
    assert "https://example.com" not in captured.out

def test_links_right_side(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[link](https://example.com)")

    sys.argv = ["multitool.py", "links", str(md_file), "--right"]
    main()
    captured = capsys.readouterr()
    assert "https://example.com" in captured.out
    assert "link" not in captured.out

def test_links_pairs(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[link](https://example.com)")

    # Test pairs with arrow format
    sys.argv = ["multitool.py", "links", str(md_file), "--pairs", "--output-format", "line"]
    main()
    captured = capsys.readouterr()
    assert "link -> https://example.com" in captured.out

def test_links_raw(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[My Link](https://example.com)")

    # Test without --raw (default cleaning)
    sys.argv = ["multitool.py", "links", str(md_file)]
    main()
    captured = capsys.readouterr()
    assert "mylink" in captured.out # lowercase and filtered to letters

    # Test with --raw
    sys.argv = ["multitool.py", "links", str(md_file), "--raw"]
    main()
    captured = capsys.readouterr()
    assert "My Link" in captured.out
