import os
import sys
import json
import pytest
from multitool import main

def test_brokenlinks_anchor_suggestion(tmp_path, capsys):
    # Create a markdown file with a heading and a broken link to a similar heading
    md_file = tmp_path / "test.md"
    md_file.write_text("# My Great Heading\n\n[Link](#my-grate-heading)", encoding='utf-8')

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "json"]

    try:
        main()
    except SystemExit as e:
        assert e.code == 0

    out, err = capsys.readouterr()
    results = json.loads(out)

    assert len(results) == 1
    # For JSON format, it's a dict {location: "url (reason)"}
    reason = list(results.values())[0]
    assert "Did you mean: #my-great-heading?" in reason

def test_brokenlinks_file_suggestion(tmp_path, capsys):
    # Create a target file and another file with a broken link to a similar filename
    target_file = tmp_path / "target_file.md"
    target_file.write_text("# Target", encoding='utf-8')

    source_file = tmp_path / "source.md"
    source_file.write_text("[Link](target_file.md)\n[Broken Link](targat_file.md)", encoding='utf-8')

    sys.argv = ["multitool.py", "brokenlinks", str(source_file), str(target_file), "--output-format", "json"]

    try:
        main()
    except SystemExit as e:
        assert e.code == 0

    out, err = capsys.readouterr()
    results = json.loads(out)

    # Only the broken link should be reported
    assert len(results) == 1
    reason = list(results.values())[0]
    assert "Did you mean: target_file.md?" in reason

def test_brokenlinks_cross_file_anchor_suggestion(tmp_path, capsys):
    # Create two files, one with a heading and another with a broken link to that heading
    target_file = tmp_path / "target.md"
    target_file.write_text("# Specific Heading", encoding='utf-8')

    source_file = tmp_path / "source.md"
    source_file.write_text("[Link](target.md#specfic-heading)", encoding='utf-8')

    sys.argv = ["multitool.py", "brokenlinks", str(source_file), str(target_file), "--output-format", "json"]

    try:
        main()
    except SystemExit as e:
        assert e.code == 0

    out, err = capsys.readouterr()
    results = json.loads(out)

    assert len(results) == 1
    reason = list(results.values())[0]
    assert "Did you mean: #specific-heading?" in reason
