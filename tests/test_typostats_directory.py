import os
import sys
from unittest.mock import patch
import pytest
from typostats import main, _STDIN_CACHE

def test_typostats_recursive_directory_scanning(tmp_path, monkeypatch):
    # Setup directories
    root = tmp_path / "typostats_test_root"
    root.mkdir()

    subdir = root / "subdir"
    subdir.mkdir()

    ignored_dir = root / "node_modules"
    ignored_dir.mkdir()

    # Supported files in active folders
    file1 = root / "file1.txt"
    file1.write_text("teh -> the\n")

    file2 = subdir / "file2.csv"
    file2.write_text("recived,received\n")

    # JSON file
    file3 = subdir / "file3.json"
    file3.write_text('{"replacements": [{"typo": "seperate", "correct": "separate"}]}')

    # Files inside ignored folder (should be ignored)
    ignored_file = ignored_dir / "ignored.txt"
    ignored_file.write_text("ignoredtypo -> correction\n")

    # Unsupported extension in active folder (should be ignored)
    unsupported_file = root / "script.py"
    unsupported_file.write_text("pytypo -> pycorrect\n")

    # Output file path
    output_file = tmp_path / "report.json"

    # Clear cache
    monkeypatch.setattr("typostats._STDIN_CACHE", None)

    # Run typostats with recursive scan on the root folder
    with patch("sys.argv", ["typostats.py", str(root), "--format", "json", "--output", str(output_file), "--all"]):
        main()

    assert output_file.exists()
    content = output_file.read_text()

    # The typos we expect:
    # 1. teh -> the (e -> h transpose, or e -> h)
    # 2. recived -> received (i -> ei insertion/1-to-2 or deletion/2-to-1)
    # 3. seperate -> separate (e -> a replacement)
    # The ignored files should NOT be processed.
    assert "ignoredtypo" not in content
    assert "pytypo" not in content

    # Let's verify standard replacements are processed
    # separate / seperate -> correct: a, typo: e
    assert '"typo": "e"' in content
    assert '"correct": "a"' in content
