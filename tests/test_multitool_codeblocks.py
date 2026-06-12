import pytest
from multitool import main, codeblocks_mode
import sys
import os

def test_codeblocks_basic(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
Some text.
```python
print("hello")
```
More text.
~~~bash
ls -l
~~~
""")

    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--raw"]
    main()

    captured = capsys.readouterr()
    assert 'print("hello")' in captured.out
    assert 'ls -l' in captured.out

def test_codeblocks_language_filter(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
```python
import sys
```
```bash
echo "hi"
```
""")

    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--language", "python", "--raw"]
    main()

    captured = capsys.readouterr()
    assert 'import sys' in captured.out
    assert 'echo "hi"' not in captured.out

def test_codeblocks_pairs(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
```python
x = 1
```
""")

    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--pairs", "--raw"]
    main()

    captured = capsys.readouterr()
    assert "python -> x = 1" in captured.out

def test_codeblocks_cleaning(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
```python
Print("Hello World!")
```
""")

    # Use codeblocks_mode directly to avoid argparse and logging complexities
    output_file = tmp_path / "output.txt"
    codeblocks_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        clean_items=True
    )

    content = output_file.read_text()
    # filter_to_letters("Print(\"Hello World!\")\n") -> "printhelloworld"
    assert "printhelloworld" in content
    assert "Print" not in content

def test_codeblocks_length_filter(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
```python
short
```
```python
very long code block indeed
```
""")

    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--min-length", "10", "--raw"]
    main()

    captured = capsys.readouterr()
    assert "very long code block indeed" in captured.out
    assert "short" not in captured.out

def test_codeblocks_process_output(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
```python
b
```
```python
a
```
```python
a
```
""")

    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--process", "--raw"]
    main()

    captured = capsys.readouterr()
    out = captured.out.strip().split('\n')
    # Filter out potential empty lines or stats
    items = [line for line in out if line in ["a", "b"]]
    assert items == ["a", "b"]

def test_codeblocks_unclosed(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
```python
unclosed block
""")

    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--raw"]
    main()

    captured = capsys.readouterr()
    assert "unclosed block" in captured.out

def test_codeblocks_indented(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
    ```python
    indented code
    ```
""")

    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--raw"]
    main()

    captured = capsys.readouterr()
    assert "indented code" in captured.out

def test_codeblocks_empty(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
```python
```
""")

    # Just verify it doesn't crash and we get some output (even if empty)
    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--raw"]
    main()

    captured = capsys.readouterr()
    # If it's empty, we might just get a newline or nothing.
    assert captured.out == "\n" or captured.out == ""

def test_codeblocks_nested_fences(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
````markdown
```python
print("nested")
```
````
""")

    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--raw"]
    main()

    captured = capsys.readouterr()
    assert '```python' in captured.out
    assert 'print("nested")' in captured.out
    assert '```' in captured.out
