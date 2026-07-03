import pytest
from multitool import codeblocks_mode, _extract_markdown_codeblocks

def test_extract_markdown_codeblocks_basic(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
Some text
```python
print("hello")
```
More text
~~~bash
echo "hi"
~~~
""")

    results = list(_extract_markdown_codeblocks(str(md_file)))
    assert len(results) == 2
    assert results[0] == ("python", 'print("hello")\n')
    assert results[1] == ("bash", 'echo "hi"\n')

def test_extract_markdown_codeblocks_unclosed(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("```python\nprint('unclosed')")

    results = list(_extract_markdown_codeblocks(str(md_file)))
    assert len(results) == 1
    assert results[0] == ("python", "print('unclosed')")

def test_extract_markdown_codeblocks_fence_lengths(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
````python
```
inner
```
````
""")
    results = list(_extract_markdown_codeblocks(str(md_file)))
    assert len(results) == 1
    assert results[0][0] == "python"
    assert "```\ninner\n```" in results[0][1]

def test_codeblocks_mode_basic(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("```python\nprint(1)\n```\n```bash\nls\n```")
    output_file = tmp_path / "output.txt"

    codeblocks_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        clean_items=False
    )

    content = output_file.read_text()
    assert "print(1)" in content
    assert "ls" in content

def test_codeblocks_mode_language_filter(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("```python\nprint(1)\n```\n```bash\nls\n```")
    output_file = tmp_path / "output.txt"

    codeblocks_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        language="python",
        clean_items=False
    )

    content = output_file.read_text()
    assert "print(1)" in content
    assert "ls" not in content

def test_codeblocks_mode_pairs(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("```python\nprint(1)\n```")
    output_file = tmp_path / "output.txt"

    codeblocks_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        pairs=True,
        clean_items=False
    )

    content = output_file.read_text()
    assert "python -> print(1)" in content

def test_codeblocks_mode_clean_and_length(tmp_path):
    md_file = tmp_path / "test.md"
    # Content "abc 123" cleaned becomes "abc" (length 3)
    md_file.write_text("```\nabc 123\n```")
    output_file = tmp_path / "output.txt"

    # min_length=3, clean_items=True.
    # Current bug: it checks length of "abc" (3) but saves "abc 123\n"
    codeblocks_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=5,
        process_output=False,
        clean_items=True
    )

    content = output_file.read_text().strip()
    # After fix, content will be "abc" (cleaned version)
    assert content == "abc"

def test_codeblocks_mode_process_output_and_length_skip(tmp_path):
    md_file = tmp_path / "test.md"
    # Duplicate blocks and one too short block
    md_file.write_text("```\nlongblock\n```\n```\nlongblock\n```\n```\nsh\n```")
    output_file = tmp_path / "output.txt"

    codeblocks_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=5,
        max_length=100,
        process_output=True, # Should trigger sorted(set(results))
        clean_items=False
    )

    content = output_file.read_text().splitlines()
    # "longblock\n" has length 10. "sh\n" has length 3.
    # min_length=5, so "sh\n" is skipped (triggers line 2258).
    # process_output=True, so "longblock\n" is de-duplicated (triggers line 2266).
    # Note: write_output ensures each item has a newline.
    # "longblock\n" already has one, so it is preserved as "longblock\n"
    assert content == ["longblock"]
