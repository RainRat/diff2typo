import pytest
import sys
import os
import logging
from multitool import main

def test_brokenlinks_duplicate_slugs(tmp_path, caplog, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Header\n# Header\n# Header\n\n[Link 1](#header)\n[Link 2](#header-1)\n[Link 3](#header-2)\n[Broken](#header-3)")

    with caplog.at_level(logging.INFO):
        sys.argv = ["multitool.py", "brokenlinks", str(md_file)]
        main()

    captured = capsys.readouterr()
    assert "Anchor not found: #header-3" in captured.out
    assert "Found 1 broken links" in caplog.text

def test_brokenlinks_reference_styles(tmp_path, caplog, capsys):
    md_file = tmp_path / "test.md"
    content = [
        "[inline](https://google.com)",
        "[ref][label]",
        "[shortcut][]",
        "[broken-ref][missing]",
        "",
        "[label]: #target",
        "[shortcut]: #target2",
        "# Target",
        "## Target2"
    ]
    md_file.write_text("\n".join(content))

    with caplog.at_level(logging.INFO):
        sys.argv = ["multitool.py", "brokenlinks", str(md_file)]
        main()

    captured = capsys.readouterr()
    assert "Undefined reference label: missing" in captured.out
    assert "Found 1 broken links" in caplog.text

def test_brokenlinks_on_the_fly_scan(tmp_path, caplog, capsys):
    subdir = tmp_path / "sub"
    subdir.mkdir()
    target_file = subdir / "target.md"
    target_file.write_text("# Target Header")

    source_file = tmp_path / "source.md"
    source_file.write_text("[Good](sub/target.md#target-header)\n[Bad](sub/target.md#missing)")

    with caplog.at_level(logging.INFO):
        sys.argv = ["multitool.py", "brokenlinks", str(source_file)]
        main()

    captured = capsys.readouterr()
    assert "Anchor not found in sub/target.md: #missing" in captured.out
    assert "Found 1 broken links" in caplog.text

def test_brokenlinks_relative_paths(tmp_path, caplog):
    subdir = tmp_path / "sub"
    subdir.mkdir()
    source_file = subdir / "source.md"
    source_file.write_text("[Up](../target.md)")

    target_file = tmp_path / "target.md"
    target_file.touch()

    with caplog.at_level(logging.INFO):
        sys.argv = ["multitool.py", "brokenlinks", str(source_file)]
        main()

    assert "Found 0 broken links" in caplog.text

import json

def test_brokenlinks_arrow_format(tmp_path, capsys, monkeypatch):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Broken](#missing)")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "arrow"]
    monkeypatch.setenv("NO_COLOR", "1")
    main()

    captured = capsys.readouterr()
    assert "Location" in captured.out
    assert "Reason" in captured.out
    assert "Anchor not found: #missing" in captured.out
    assert "BROKEN LINKS ANALYSIS" in captured.out

def test_brokenlinks_json_format(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Broken](#missing)")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "json"]
    main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    # The key is Location [Text] and the value is URL (Reason)
    key = next(k for k in data.keys() if "[Broken]" in k)
    assert "Anchor not found: #missing" in data[key]

def test_brokenlinks_empty_link(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[]()")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file)]
    main()

    captured = capsys.readouterr()
    assert "Empty link" in captured.out

def test_brokenlinks_external_and_special(tmp_path, caplog):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Mail](mailto:test@example.com)\n[HTTP](http://example.com)\n[FTP](ftp://example.com)")

    with caplog.at_level(logging.INFO):
        sys.argv = ["multitool.py", "brokenlinks", str(md_file)]
        main()

    assert "Found 0 broken links" in caplog.text

def test_brokenlinks_ignore_non_markdown(tmp_path, caplog):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("[Broken](#missing)")

    with caplog.at_level(logging.INFO):
        sys.argv = ["multitool.py", "brokenlinks", str(txt_file)]
        main()

    assert "Found 0 broken links" in caplog.text

def test_brokenlinks_query_params(tmp_path, caplog):
    target_file = tmp_path / "target.md"
    target_file.touch()

    md_file = tmp_path / "test.md"
    md_file.write_text("[Query](target.md?v=1)\n[AnchorQuery](#anchor?v=1)\n# Anchor")

    with caplog.at_level(logging.INFO):
        sys.argv = ["multitool.py", "brokenlinks", str(md_file)]
        main()

    assert "Found 0 broken links" in caplog.text

def test_brokenlinks_limit(tmp_path, caplog, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[B1](#m1)\n[B2](#m2)")

    with caplog.at_level(logging.INFO):
        sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--limit", "1"]
        main()

    captured = capsys.readouterr()
    assert "Found 1 broken links" in caplog.text
    assert "m1" in captured.out
    assert "m2" not in captured.out

def test_brokenlinks_missing_anchor_in_file_not_in_input(tmp_path, caplog, capsys):
    target_file = tmp_path / "target.md"
    target_file.write_text("# Target")

    md_file = tmp_path / "test.md"
    md_file.write_text("[Bad](target.md#missing)\n[Good](target.md#target)")

    with caplog.at_level(logging.INFO):
        # target.md exists but not in input_files
        sys.argv = ["multitool.py", "brokenlinks", str(md_file)]
        main()

    captured = capsys.readouterr()
    assert "Anchor not found in target.md: #missing" in captured.out
    assert "Found 1 broken links" in caplog.text

def test_brokenlinks_unhandled_edge_case(tmp_path, caplog, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Edge](?query#anchor)")

    with caplog.at_level(logging.INFO):
        sys.argv = ["multitool.py", "brokenlinks", str(md_file)]
        main()

    assert "Found 0 broken links" in caplog.text

def test_brokenlinks_missing_file(tmp_path, caplog, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Missing](nonexistent.md)")

    with caplog.at_level(logging.INFO):
        sys.argv = ["multitool.py", "brokenlinks", str(md_file)]
        main()

    captured = capsys.readouterr()
    assert "File not found: nonexistent.md" in captured.out
    assert "Found 1 broken links" in caplog.text

def test_brokenlinks_anchor_in_non_markdown(tmp_path, caplog, capsys):
    target_file = tmp_path / "data.txt"
    target_file.touch()

    md_file = tmp_path / "test.md"
    md_file.write_text("[Anchor in text](data.txt#anchor)")

    with caplog.at_level(logging.INFO):
        sys.argv = ["multitool.py", "brokenlinks", str(md_file)]
        main()

    assert "Found 0 broken links" in caplog.text
