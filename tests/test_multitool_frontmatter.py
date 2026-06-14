import pytest
from multitool import main
import sys
import os

def test_frontmatter_yaml_raw(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    content = """---
title: Hello World
tags: [test, doc]
---
# Content
"""
    md_file.write_text(content)

    sys.argv = ["multitool.py", "frontmatter", str(md_file), "--raw"]
    main()

    captured = capsys.readouterr()
    assert "title: Hello World" in captured.out
    assert "tags: [test, doc]" in captured.out
    assert "# Content" not in captured.out

def test_frontmatter_toml_raw(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    content = """+++
title = "Hello World"
tags = ["test", "doc"]
+++
# Content
"""
    md_file.write_text(content)

    sys.argv = ["multitool.py", "frontmatter", str(md_file), "--raw"]
    main()

    captured = capsys.readouterr()
    assert 'title = "Hello World"' in captured.out
    assert 'tags = ["test", "doc"]' in captured.out
    assert "# Content" not in captured.out

def test_frontmatter_yaml_key(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    content = """---
title: Hello World
author: Jules
---
"""
    md_file.write_text(content)

    sys.argv = ["multitool.py", "frontmatter", str(md_file), "--key", "title", "--raw"]
    main()

    captured = capsys.readouterr()
    assert "Hello World" in captured.out
    assert "Jules" not in captured.out

def test_frontmatter_toml_key(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    content = """+++
title = "Hello World"
author = "Jules"
+++
"""
    md_file.write_text(content)

    sys.argv = ["multitool.py", "frontmatter", str(md_file), "--key", "author", "--raw"]
    main()

    captured = capsys.readouterr()
    assert "Jules" in captured.out
    assert "Hello World" not in captured.out

def test_frontmatter_no_fm(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Just Content")

    sys.argv = ["multitool.py", "frontmatter", str(md_file)]
    main()

    captured = capsys.readouterr()
    assert captured.out.strip() == ""

def test_frontmatter_nested_key(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    content = """---
metadata:
  inner:
    value: FoundIt
---
"""
    md_file.write_text(content)

    sys.argv = ["multitool.py", "frontmatter", str(md_file), "--key", "metadata.inner.value", "--raw"]
    main()

    captured = capsys.readouterr()
    assert "FoundIt" in captured.out

def test_frontmatter_invalid_yaml(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    content = """---
title: : invalid
---
"""
    md_file.write_text(content)

    sys.argv = ["multitool.py", "frontmatter", str(md_file), "--key", "title"]
    # This might log an error but should not crash the whole process
    main()

    # We check it doesn't crash.
    captured = capsys.readouterr()
    # It should not have the value
    assert "invalid" not in captured.out
