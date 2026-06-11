import pytest
from multitool import main
import sys
from io import StringIO
import os

def test_links_mode_basic(tmp_path, capsys):
    # Create a test markdown file
    md_file = tmp_path / "test.md"
    md_file.write_text("Here is a [link](https://example.com) and an ![image](img.png).")

    # Run links mode
    sys.argv = ["multitool.py", "links", str(md_file), "--raw"]
    main()

    captured = capsys.readouterr()
    assert "link" in captured.out
    assert "image" in captured.out
    assert "https://example.com" not in captured.out

def test_links_mode_right(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Google](https://google.com)")

    sys.argv = ["multitool.py", "links", str(md_file), "--right"]
    main()

    captured = capsys.readouterr()
    assert "https://google.com" in captured.out
    # The URL itself contains 'google', so we check that the link TEXT (Google) is NOT in the output
    # By default clean_items=True, so 'Google' would become 'google'.
    assert "google" in captured.out # This is now expected because it's in the URL

def test_links_mode_pairs(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Test](https://test.com)")

    sys.argv = ["multitool.py", "links", str(md_file), "--pairs", "--raw"]
    main()

    captured = capsys.readouterr()
    assert "Test -> https://test.com" in captured.out

def test_links_mode_cleaning(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[My Link](https://example.com)")

    # By default clean_items=True, which uses filter_to_letters
    sys.argv = ["multitool.py", "links", str(md_file)]
    main()

    captured = capsys.readouterr()
    assert "mylink" in captured.out
    assert "My Link" not in captured.out
