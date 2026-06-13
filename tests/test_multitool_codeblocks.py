import pytest
import multitool
from pathlib import Path
import json

def test_extract_markdown_codeblocks_basic(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
Some text
```python
print("hello")
```
More text
~~~bash
ls -l
~~~
""")
    results = list(multitool._extract_markdown_codeblocks(str(md_file), quiet=True))
    assert len(results) == 2
    assert results[0] == ("python", 'print("hello")\n')
    assert results[1] == ("bash", "ls -l\n")

def test_extract_markdown_codeblocks_indented(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
    ```python
    indented
    ```
""")
    results = list(multitool._extract_markdown_codeblocks(str(md_file), quiet=True))
    assert len(results) == 1
    assert results[0] == ("python", "    indented\n")

def test_codeblocks_mode_basic(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("```python\ncode\n```")
    output_file = tmp_path / "out.txt"

    multitool.codeblocks_mode(
        [str(md_file)], str(output_file), 1, 100, False, clean_items=False
    )

    assert output_file.read_text().strip() == "code"

def test_codeblocks_mode_language_filter(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("```python\npycode\n```\n```bash\nbashcode\n```")
    output_file = tmp_path / "out.txt"

    multitool.codeblocks_mode(
        [str(md_file)], str(output_file), 1, 100, False, language="python", clean_items=False
    )

    results = output_file.read_text().splitlines()
    assert "pycode" in results
    assert "bashcode" not in results

def test_codeblocks_mode_length_filter(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("```\nshort\n```\n```\nverylongcontent\n```")
    output_file = tmp_path / "out.txt"

    multitool.codeblocks_mode(
        [str(md_file)], str(output_file), 1, 10, False, clean_items=False
    )

    results = output_file.read_text().splitlines()
    assert "short" in results
    assert "verylongcontent" not in results

def test_codeblocks_mode_pairs(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("```python\ncode\n```")
    output_file = tmp_path / "out.txt"

    multitool.codeblocks_mode(
        [str(md_file)], str(output_file), 1, 100, False, pairs=True, clean_items=False
    )

    assert "python -> code" in output_file.read_text()

def test_codeblocks_mode_json_output(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("```python\ncode\n```")
    output_file = tmp_path / "out.json"

    multitool.codeblocks_mode(
        [str(md_file)], str(output_file), 1, 100, False, output_format="json", clean_items=False
    )

    data = json.loads(output_file.read_text())
    # Content extracted includes the trailing newline from inside the fenced block
    assert "code\n" in data

def test_codeblocks_mode_clean_items(tmp_path):
    # This test verifies the behavior after the fix.
    md_file = tmp_path / "test.md"
    md_file.write_text("```\nHello World!\n```")
    output_file = tmp_path / "out.txt"

    # When clean_items=True, it uses filter_to_letters for length check.
    # filter_to_letters("Hello World!\n") -> "helloworld" (length 10)
    multitool.codeblocks_mode(
        [str(md_file)], str(output_file), 1, 100, False, clean_items=True
    )

    content = output_file.read_text().strip()
    # After fix, saves cleaned content
    assert content == "helloworld"

def test_codeblocks_mode_process_output(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("```\ncode\n```\n```\ncode\n```")
    output_file = tmp_path / "out.txt"

    multitool.codeblocks_mode(
        [str(md_file)], str(output_file), 1, 100, True, clean_items=False
    )

    results = output_file.read_text().splitlines()
    # "code\n" is extracted twice. Sorted set gives ["code\n"].
    # write_output writes "code\n" + newline -> "code\n\n"
    # splitlines() on "code\n\n" gives ["code", ""]
    assert results == ["code", ""]

def test_extract_markdown_codeblocks_unclosed(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("```python\nunclosed")
    results = list(multitool._extract_markdown_codeblocks(str(md_file), quiet=True))
    assert len(results) == 1
    assert results[0] == ("python", "unclosed")
