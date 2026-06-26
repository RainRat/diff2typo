from unittest.mock import patch
import pytest
from multitool import main

def test_brokenlinks_basic(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("Valid [link](https://google.com), Broken [file](missing.md).")

    with patch('sys.argv', ["multitool.py", "brokenlinks", str(md_file), "--output-format", "line"]):
        main()

    captured = capsys.readouterr()
    assert "missing.md" in captured.out
    assert "File not found: missing.md" in captured.out
    assert "https://google.com" not in captured.out

def test_brokenlinks_references(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Valid][ref]\n[Broken][missing]\n\n[ref]: ok.md")
    (tmp_path / "ok.md").write_text("ok")

    with patch('sys.argv', ["multitool.py", "brokenlinks", str(md_file), "--output-format", "line"]):
        main()

    captured = capsys.readouterr()
    assert "Undefined reference label: missing" in captured.out
    assert "ok.md" not in captured.out

def test_brokenlinks_ref_shortcut(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Ref][]\n\n[ref]: ok.md")
    (tmp_path / "ok.md").write_text("ok")

    with patch('sys.argv', ["multitool.py", "brokenlinks", str(md_file), "--output-format", "line"]):
        main()

    captured = capsys.readouterr()
    assert "ok.md" not in captured.out
    assert "Undefined reference label" not in captured.out

def test_brokenlinks_anchors(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Heading\n# Heading\n[Link 1](#heading)\n[Link 2](#heading-1)\n[Broken](#missing)")

    with patch('sys.argv', ["multitool.py", "brokenlinks", str(md_file), "--output-format", "line"]):
        main()

    captured = capsys.readouterr()
    assert "Anchor not found: #missing" in captured.out
    assert "#heading" not in captured.out
    assert "#heading-1" not in captured.out

def test_brokenlinks_cross_file(tmp_path, capsys):
    main_md = tmp_path / "main.md"
    other_md = tmp_path / "other.md"

    main_md.write_text("[Valid](other.md#target)\n[Broken Anchor](other.md#missing)\n[Broken File](none.md#anchor)")
    other_md.write_text("# Target")

    with patch('sys.argv', ["multitool.py", "brokenlinks", str(main_md), str(other_md), "--output-format", "line"]):
        main()

    captured = capsys.readouterr()
    assert f"Anchor not found in other.md: #missing" in captured.out
    assert "File not found: none.md" in captured.out
    assert "other.md#target" not in captured.out

def test_brokenlinks_arrow_output(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Broken](missing.md)")

    with patch('sys.argv', ["multitool.py", "brokenlinks", str(md_file), "--output-format", "arrow"]):
        with patch('multitool._should_enable_color', return_value=False):
            main()

    captured = capsys.readouterr()
    assert "Location" in captured.out
    assert "Reason" in captured.out
    assert "missing.md" in captured.out
    assert "BROKEN LINKS ANALYSIS" in captured.out

def test_brokenlinks_on_the_fly_scan(tmp_path, capsys):
    main_md = tmp_path / "main.md"
    other_md = tmp_path / "other.md"

    main_md.write_text("[Broken](other.md#missing)\n[Valid](other.md#target)")
    other_md.write_text("# Target")

    with patch('sys.argv', ["multitool.py", "brokenlinks", str(main_md), "--output-format", "line"]):
        main()

    captured = capsys.readouterr()
    assert "Anchor not found in other.md: #missing" in captured.out
    assert "Anchor not found in other.md: #target" not in captured.out

def test_brokenlinks_ignored_inputs(tmp_path, capsys):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("[Broken](missing.md)")

    with patch('sys.argv', ["multitool.py", "brokenlinks", str(txt_file), "--output-format", "arrow"]):
        with patch('multitool._should_enable_color', return_value=False):
            main()

    captured = capsys.readouterr()
    assert "BROKEN LINKS ANALYSIS" in captured.out
    assert "Total links analyzed:" in captured.out
    assert "0" in captured.out

def test_brokenlinks_empty_link(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[]()\n[Anchor Only](?query#anchor)")

    with patch('sys.argv', ["multitool.py", "brokenlinks", str(md_file), "--output-format", "line"]):
        main()

    captured = capsys.readouterr()
    assert "Empty link" in captured.out
    assert "?query" not in captured.out

def test_brokenlinks_limit(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("[B1](m1.md)\n[B2](m2.md)")

    with patch('sys.argv', ["multitool.py", "brokenlinks", str(md_file), "--limit", "1", "--output-format", "line"]):
        main()

    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if "File not found" in l]
    assert len(lines) == 1
