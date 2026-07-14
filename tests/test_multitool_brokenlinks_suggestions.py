import os
import sys
import json
import pytest
from multitool import main

def test_brokenlinks_suggestions(tmp_path, capsys):
    # Setup files
    file1 = tmp_path / "main.md"
    file1.write_text("# Hello World\n\n[Internal Link](#hello-word)\n[Cross File Link](other.md#target-sectio)\n[Missing File](miss.md)")

    file2 = tmp_path / "other.md"
    file2.write_text("# Target Section\n")

    # Run brokenlinks mode
    sys.argv = ["multitool.py", "brokenlinks", str(file1), str(file2), "--output-format", "json"]

    main()

    captured = capsys.readouterr()
    results = json.loads(captured.out)

    internal_found = False
    for loc, val in results.items():
        if "#hello-word" in val:
            assert "(Did you mean: #hello-world?)" in val
            internal_found = True
    assert internal_found

    cross_found = False
    for loc, val in results.items():
        if "other.md#target-sectio" in val:
            assert "(Did you mean: #target-section?)" in val
            cross_found = True
    assert cross_found

    # Check for missing file suggestion
    file_near = tmp_path / "missy.md"
    file_near.write_text("# Nearby")

    # Run again to pick up the new file
    sys.argv = ["multitool.py", "brokenlinks", str(file1), str(file2), str(file_near), "--output-format", "json"]
    main()

    captured = capsys.readouterr()
    results = json.loads(captured.out)
    file_found = False
    for loc, val in results.items():
        if "miss.md" in val and "File not found" in val:
            assert "(Did you mean: missy.md?)" in val
            file_found = True
    assert file_found

def test_brokenlinks_on_the_fly_scan_suggestions(tmp_path, capsys):
    # Setup files - file2 is NOT in the input list but is referenced
    file1 = tmp_path / "main.md"
    file1.write_text("[Cross File](other.md#target-sectio)")

    file2 = tmp_path / "other.md"
    file2.write_text("# Target Section\n")

    # Run brokenlinks mode - only file1 is input
    sys.argv = ["multitool.py", "brokenlinks", str(file1), "--output-format", "json"]

    main()

    captured = capsys.readouterr()
    results = json.loads(captured.out)

    cross_found = False
    for loc, val in results.items():
        if "other.md#target-sectio" in val:
            assert "(Did you mean: #target-section?)" in val
            cross_found = True
    assert cross_found
