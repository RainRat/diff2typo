import pytest
from multitool import main
import sys
import os

def test_frontmatter_yaml(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("---\ntitle: YAML\n---\nBody content")

    sys.argv = ["multitool.py", "frontmatter", str(md_file), "--raw"]
    main()

    captured = capsys.readouterr()
    assert "title: YAML" in captured.out
    assert "Body content" not in captured.out

def test_frontmatter_toml(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("+++\ntitle = \"TOML\"\n+++\nBody content")

    sys.argv = ["multitool.py", "frontmatter", str(md_file), "--raw"]
    main()

    captured = capsys.readouterr()
    assert "title = \"TOML\"" in captured.out
    assert "Body content" not in captured.out

def test_frontmatter_body(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("---\ntitle: YAML\n---\nBody content")

    sys.argv = ["multitool.py", "frontmatter", str(md_file), "--body", "--raw"]
    main()

    captured = capsys.readouterr()
    assert "title: YAML" not in captured.out
    assert "Body content" in captured.out

def test_frontmatter_no_fm(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("No frontmatter here")

    # Frontmatter mode should return nothing if no delimiter found
    sys.argv = ["multitool.py", "frontmatter", str(md_file), "--raw"]
    main()

    captured = capsys.readouterr()
    assert "No frontmatter here" not in captured.out

    # Body mode should return everything if no delimiter found
    sys.argv = ["multitool.py", "frontmatter", str(md_file), "--body", "--raw"]
    main()

    captured = capsys.readouterr()
    assert "No frontmatter here" in captured.out

def test_frontmatter_unclosed(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("---\ntitle: Unclosed\nBody content")

    # Frontmatter mode should return nothing if unclosed
    sys.argv = ["multitool.py", "frontmatter", str(md_file), "--raw"]
    main()

    captured = capsys.readouterr()
    assert "title: Unclosed" not in captured.out

    # Body mode should return everything if unclosed
    sys.argv = ["multitool.py", "frontmatter", str(md_file), "--body", "--raw"]
    main()

    captured = capsys.readouterr()
    assert "title: Unclosed" in captured.out
