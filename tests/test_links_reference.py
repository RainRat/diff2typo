import pytest
from multitool import main
import sys
import os

def test_links_mode_reference(tmp_path, capsys):
    # Create a test markdown file with reference-style links
    md_file = tmp_path / "refs.md"
    md_file.write_text("Check [this][1] and [that][].\n\n[1]: https://example.com\n[that]: https://that.com")

    # Run links mode
    sys.argv = ["multitool.py", "links", str(md_file), "--raw"]
    main()

    captured = capsys.readouterr()
    assert "this" in captured.out
    assert "that" in captured.out

def test_links_mode_reference_right(tmp_path, capsys):
    md_file = tmp_path / "refs.md"
    md_file.write_text("Check [this][1].\n\n[1]: https://example.com")

    sys.argv = ["multitool.py", "links", str(md_file), "--right"]
    main()

    captured = capsys.readouterr()
    assert "https://example.com" in captured.out

def test_links_mode_reference_pairs(tmp_path, capsys):
    md_file = tmp_path / "refs.md"
    md_file.write_text("[Test][1]\n\n[1]: https://test.com")

    sys.argv = ["multitool.py", "links", str(md_file), "--pairs", "--raw"]
    main()

    captured = capsys.readouterr()
    assert "Test -> https://test.com" in captured.out

def test_links_mode_broken_reference(tmp_path, capsys):
    md_file = tmp_path / "refs.md"
    md_file.write_text("[Broken][2]")

    sys.argv = ["multitool.py", "links", str(md_file), "--right"]
    main()

    captured = capsys.readouterr()
    assert "broken-ref:2" in captured.out
