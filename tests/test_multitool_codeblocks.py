import pytest
from multitool import main
import sys
import os
import json

def test_codeblocks_mode_basic(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("Some text\n```python\nprint('hello')\n```\nMore text\n~~~bash\necho hi\n~~~")

    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--raw"]
    main()

    captured = capsys.readouterr()
    assert "print('hello')" in captured.out
    assert "echo hi" in captured.out

def test_codeblocks_mode_language_filter(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("```python\nprint('hello')\n```\n```bash\necho hi\n```")

    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--language", "python", "--raw"]
    main()

    captured = capsys.readouterr()
    assert "print('hello')" in captured.out
    assert "echo hi" not in captured.out

def test_codeblocks_mode_pairs(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("```python\nprint('hello')\n```")

    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--pairs", "--raw"]
    main()

    captured = capsys.readouterr()
    assert "python -> print('hello')" in captured.out

def test_codeblocks_mode_cleaning(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("```python\nprint('Hello World!')\n```")

    # By default clean_items=True, which uses filter_to_letters (only lowercase a-z)
    sys.argv = ["multitool.py", "codeblocks", str(md_file)]
    main()

    captured = capsys.readouterr()
    assert "printhelloworld" in captured.out
    assert "Hello World" not in captured.out

def test_codeblocks_mode_unclosed(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("```python\nprint('unclosed')")

    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--raw"]
    main()

    captured = capsys.readouterr()
    assert "print('unclosed')" in captured.out

def test_codeblocks_mode_json_output(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("```python\nx = 1\n```")

    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--output-format", "json", "--raw"]
    main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data == ["x = 1\n"]

def test_codeblocks_mode_min_max_length(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("```\nshort\n```\n```\nthis is a much longer code block\n```")

    # Only include blocks between 10 and 100 characters (cleaned)
    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--min-length", "10", "--max-length", "100"]
    main()

    captured = capsys.readouterr()
    assert "thisisamuchlongercodeblock" in captured.out
    assert "short" not in captured.out

def test_codeblocks_mode_process_output(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("```\nblock\n```\n```\nblock\n```")

    # --process-output triggers process_output=True
    sys.argv = ["multitool.py", "codeblocks", str(md_file), "--process-output", "--raw"]
    main()

    captured = capsys.readouterr()
    # Should only appear once
    assert captured.out.strip() == "block"
