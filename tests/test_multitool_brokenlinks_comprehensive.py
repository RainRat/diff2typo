import pytest
from multitool import main
import sys
import os
import json

def test_brokenlinks_internal_anchors(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
# Header
Content

# Header
More content

[Link 1](#header)
[Link 2](#header-1)
[Broken](#non-existent)
""")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "json"]
    main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert any("non-existent" in val for val in data.values())
    assert not any("header" in val and "non-existent" not in val for val in data.values())

def test_brokenlinks_relative_paths(tmp_path, capsys):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    file1 = doc_dir / "file1.md"
    file1.write_text("[Link to file2](file2.md)\n[Broken link](missing.md)")
    file2 = doc_dir / "file2.md"
    file2.write_text("# Target")

    sys.argv = ["multitool.py", "brokenlinks", str(file1), "--output-format", "json"]
    main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert any("missing.md" in val for val in data.values())
    assert not any("file2.md" in val for val in data.values())

def test_brokenlinks_cross_file_anchors(tmp_path, capsys):
    file1 = tmp_path / "file1.md"
    file1.write_text("[Valid](file2.md#target)\n[Invalid](file2.md#missing)")
    file2 = tmp_path / "file2.md"
    file2.write_text("# Target")

    sys.argv = ["multitool.py", "brokenlinks", str(file1), str(file2), "--output-format", "json"]
    main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert any("file2.md#missing" in val for val in data.values())
    assert not any("file2.md#target" in val for val in data.values())

def test_brokenlinks_reference_style(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("""
[Text][label]
[Shortcut][]

[label]: #valid
[Shortcut]: #missing
# Valid
""")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "json"]
    main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert any("#missing" in val for val in data.values())
    assert not any("#valid" in val for val in data.values())

def test_brokenlinks_on_the_fly_scan(tmp_path, capsys):
    file1 = tmp_path / "file1.md"
    file1.write_text("[Cross](file2.md#target)")

    file2 = tmp_path / "file2.md"
    file2.write_text("# Target")

    sys.argv = ["multitool.py", "brokenlinks", str(file1), "--output-format", "json"]
    main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data == {}

def test_brokenlinks_arrow_format(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Broken](#missing)")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "arrow"]
    main()

    captured = capsys.readouterr()
    assert "BROKEN LINKS ANALYSIS" in captured.out
    assert "Location" in captured.out
    assert "#missing" in captured.out

def test_brokenlinks_external_ignored(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Google](https://google.com/404)")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "json"]
    main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data == {}

def test_brokenlinks_empty_link(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Empty]()")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "json"]
    main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert any("Empty link" in val for val in data.values())

def test_brokenlinks_undefined_reference(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Missing Ref][missing]")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "json"]
    main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert any("Undefined reference label: missing" in val for val in data.values())

def test_brokenlinks_skips_non_markdown_files(tmp_path, capsys):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("[Broken](#missing)")

    sys.argv = ["multitool.py", "brokenlinks", str(txt_file), "--output-format", "json"]
    main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data == {}

def test_brokenlinks_respects_limit(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[B1](#m1)\n[B2](#m2)")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "json", "--limit", "1"]
    main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert len(data) == 1

def test_brokenlinks_on_the_fly_scan_identifies_missing_anchor(tmp_path, capsys):
    file1 = tmp_path / "file1.md"
    file1.write_text("[Cross](file2.md#missing)")

    file2 = tmp_path / "file2.md"
    file2.write_text("# Target")

    sys.argv = ["multitool.py", "brokenlinks", str(file1), "--output-format", "json"]
    main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert any("Anchor not found in file2.md: #missing" in val for val in data.values())

def test_brokenlinks_ignores_query_params_in_internal_anchor(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Target\n[Link](#target?param=1)")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "json"]
    main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data == {}

def test_brokenlinks_ignores_query_params_in_file_path(tmp_path, capsys):
    file1 = tmp_path / "file1.md"
    file1.write_text("[Link](file2.md?param=1)")
    file2 = tmp_path / "file2.md"
    file2.write_text("# Target")

    sys.argv = ["multitool.py", "brokenlinks", str(file1), "--output-format", "json"]
    main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data == {}

def test_brokenlinks_handles_malformed_anchor_only_links(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Target\n[Link](?query#target)")

    sys.argv = ["multitool.py", "brokenlinks", str(md_file), "--output-format", "json"]
    main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data == {}
