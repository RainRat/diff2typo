import pytest
from multitool import main, _extract_markdown_codeblocks
import sys
from io import StringIO
import os

def test_extract_markdown_codeblocks_basic(tmp_path):
    md_file = tmp_path / "test.md"
    content = """
# Heading
```python
print("hello")
```
Some text
~~~bash
ls -l
~~~
"""
    md_file.write_text(content)

    blocks = list(_extract_markdown_codeblocks(str(md_file), quiet=True))
    assert len(blocks) == 2
    assert blocks[0] == ("python", 'print("hello")\n')
    assert blocks[1] == ("bash", "ls -l\n")

def test_extract_markdown_codeblocks_unclosed(tmp_path):
    md_file = tmp_path / "unclosed.md"
    content = "```javascript\nconsole.log('hi');"
    md_file.write_text(content)

    blocks = list(_extract_markdown_codeblocks(str(md_file), quiet=True))
    assert len(blocks) == 1
    assert blocks[0] == ("javascript", "console.log('hi');")

def test_extract_markdown_codeblocks_nested_fences(tmp_path):
    md_file = tmp_path / "nested.md"
    content = """
````markdown
```python
print("nested")
```
````
"""
    md_file.write_text(content)

    blocks = list(_extract_markdown_codeblocks(str(md_file), quiet=True))
    assert len(blocks) == 1
    assert blocks[0][0] == "markdown"
    assert '```python' in blocks[0][1]

def test_codeblocks_mode_cli_basic(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("```python\nx = 1\n```")

    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--raw"]
    main()

    captured = capsys.readouterr()
    assert "x = 1" in captured.out

def test_codeblocks_mode_cli_language_filter(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("```python\nprint(1)\n```\n```bash\necho 1\n```")

    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--language", "python", "--raw"]
    main()

    captured = capsys.readouterr()
    assert "print(1)" in captured.out
    assert "echo 1" not in captured.out

def test_codeblocks_mode_cli_pairs(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("```python\nx = 1\n```")

    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--pairs", "--raw"]
    main()

    captured = capsys.readouterr()
    assert "python -> x = 1" in captured.out

def test_codeblocks_mode_cli_cleaning(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("```\nHello World!\n```")

    # By default clean_items=True, which uses filter_to_letters
    sys.argv = ["multitool.py", "codeblocks", str(md_file)]
    main()

    captured = capsys.readouterr()
    assert "helloworld" in captured.out
    assert "Hello World!" not in captured.out

def test_codeblocks_mode_cli_min_length(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("```\na\n```\n```\nabcd\n```")

    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--min-length", "3", "--raw"]
    main()

    captured = capsys.readouterr()
    results = captured.out.strip().split('\n')
    assert "abcd" in results
    assert "a" not in results
