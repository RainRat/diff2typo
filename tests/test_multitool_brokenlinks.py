import os
import sys
import pytest
import json
from multitool import (
    _get_markdown_anchor_map,
    _extract_markdown_links_detailed,
    brokenlinks_mode
)

def test_get_markdown_anchor_map_basic(tmp_path):
    f1 = tmp_path / "test1.md"
    f1.write_text("# Heading 1\n## Heading 2\n# Heading 1\n### Subheading")

    f2 = tmp_path / "test2.md"
    f2.write_text("# Other File")

    # Test skipping non-markdown and '-' (line 1741-1743)
    anchor_map = _get_markdown_anchor_map([str(f1), str(f2), "-", "test.txt"])

    assert str(f1) in anchor_map
    assert str(f2) in anchor_map
    assert "-" not in anchor_map
    assert "test.txt" not in anchor_map

    assert "heading-1" in anchor_map[str(f1)]
    assert "heading-2" in anchor_map[str(f1)]
    assert "heading-1-1" in anchor_map[str(f1)]
    assert "subheading" in anchor_map[str(f1)]
    assert "other-file" in anchor_map[str(f2)]

def test_extract_markdown_links_detailed_all_types(tmp_path):
    f = tmp_path / "links.md"
    content = [
        "[inline](url1)",
        "![image](img.png)",
        "[ref link][label]",
        "[shortcut][]",
        "[label]: url2",
        "[shortcut]: url3",
        "[Label Only][]" # Should use text as label
    ]
    f.write_text("\n".join(content))

    links = list(_extract_markdown_links_detailed(str(f)))

    assert ("inline", "url1", 1) in links
    assert ("image", "img.png", 2) in links
    assert ("ref link", "url2", 3) in links
    assert ("shortcut", "url3", 4) in links
    # [Label Only][] -> text="Label Only", label="label only"
    assert ("Label Only", "broken-ref:label only", 7) in links

def test_extract_markdown_links_detailed_resolved_shortcut(tmp_path):
    f = tmp_path / "shortcut_resolved.md"
    f.write_text("[Label][]\n\n[label]: url")
    links = list(_extract_markdown_links_detailed(str(f)))
    assert ("Label", "url", 1) in links

def test_extract_markdown_links_detailed_broken_ref(tmp_path):
    f = tmp_path / "broken_ref.md"
    f.write_text("[undefined][missing]")

    links = list(_extract_markdown_links_detailed(str(f)))
    assert ("undefined", "broken-ref:missing", 1) in links

def test_brokenlinks_mode_comprehensive(tmp_path, capsys):
    main_md = tmp_path / "main.md"
    main_md.write_text("""
# Main
[Valid Anchor](#main)
[Invalid Anchor](#missing)
[External](https://google.com)
[Valid File](other.md)
[Invalid File](nope.md)
[Valid Cross Anchor](other.md#section)
[Invalid Cross Anchor](other.md#wrong)
[Valid On-the-fly](standalone.md#here)
[Invalid On-the-fly](standalone2.md#missing)
[Empty Link]()
[Missing Local Anchor](other.md#)
""")

    other_md = tmp_path / "other.md"
    other_md.write_text("# Section")

    standalone_md = tmp_path / "standalone.md"
    standalone_md.write_text("# Here")

    standalone2_md = tmp_path / "standalone2.md"
    standalone2_md.write_text("# SomethingElse")

    # Run brokenlinks_mode
    # standalone.md and standalone2.md are NOT in the input_files list to test on-the-fly scanning
    # also add a non-md file to test skipping at line 2587
    brokenlinks_mode(
        input_files=[str(main_md), str(other_md), "dummy.txt"],
        output_file="-",
        output_format="json",
        limit=20
    )

    captured = capsys.readouterr()
    output_str = captured.out
    assert "#missing" in output_str
    assert "Anchor not found: #missing" in output_str
    assert "nope.md" in output_str
    assert "File not found: nope.md" in output_str
    assert "other.md#wrong" in output_str
    assert "Anchor not found in other.md: #wrong" in output_str
    assert "standalone2.md#missing" in output_str
    assert "Anchor not found in standalone2.md: #missing" in output_str
    assert "Empty link" in output_str

    # Valid ones should NOT be in the output
    assert "#main" not in output_str
    assert "https://google.com" not in output_str
    assert "standalone.md#here" not in output_str

def test_brokenlinks_mode_arrow_format(tmp_path, capsys):
    f = tmp_path / "test.md"
    f.write_text("[Broken](#missing)")

    brokenlinks_mode(
        input_files=[str(f)],
        output_file="-",
        output_format="arrow"
    )

    captured = capsys.readouterr()
    assert "BROKEN LINKS ANALYSIS" in captured.out
    assert "Location" in captured.out
    assert "Text" in captured.out
    assert "URL" in captured.out
    assert "Reason" in captured.out
    assert "Anchor not found: #missing" in captured.out

def test_brokenlinks_mode_shortcut_and_mapping(tmp_path, capsys):
    # Both files included to test internal mapping
    f = tmp_path / "shortcut.md"
    f.write_text("# My Heading\n[Shortcut][]\n\n[Shortcut]: #my-heading")

    brokenlinks_mode(
        input_files=[str(f)],
        output_file="-",
        output_format="json"
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == "{}"

def test_brokenlinks_mode_missing_ref_label(tmp_path, capsys):
    f = tmp_path / "test.md"
    f.write_text("[Missing][label]")

    brokenlinks_mode(
        input_files=[str(f)],
        output_file="-",
        output_format="json"
    )
    captured = capsys.readouterr()
    assert "Undefined reference label: label" in captured.out

def test_brokenlinks_mode_query_params(tmp_path, capsys):
    f = tmp_path / "test.md"
    # Testing both internal anchor and file with anchor having query params
    f.write_text("# Target\n[Link](#target?param=1)\n[Link2](test.md?v=2#target)")

    brokenlinks_mode(
        input_files=[str(f)],
        output_file="-",
        output_format="json"
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == "{}"

def test_brokenlinks_mode_file_with_query_and_anchor(tmp_path, capsys):
    main = tmp_path / "main.md"
    other = tmp_path / "other.md"
    other.write_text("# Target")
    main.write_text("[Link](other.md?v=1#target)")

    brokenlinks_mode(
        input_files=[str(main), str(other)],
        output_file="-",
        output_format="json"
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == "{}"

def test_brokenlinks_mode_local_anchor_edge_cases(tmp_path, capsys):
    f = tmp_path / "test.md"
    # url="" (empty link) and url="#anchor" where anchor is handled by first part of if/else
    f.write_text("[Empty]()\n[Only Anchor](#target)")

    brokenlinks_mode(
        input_files=[str(f)],
        output_file="-",
        output_format="json"
    )
    captured = capsys.readouterr()
    assert "Empty link" in captured.out
    assert "Anchor not found: #target" in captured.out

def test_brokenlinks_mode_unreachable_branch(tmp_path, capsys):
    """Try to reach line 2620, though it may be logic-unreachable."""
    f = tmp_path / "test.md"
    f.write_text("# Heading")

    from unittest.mock import patch

    class MockStr(str):
        def startswith(self, prefix, *args, **kwargs):
            if prefix == "#": return False
            return super().startswith(prefix, *args, **kwargs)

    mock_url = MockStr("#anchor")

    with patch("multitool._extract_markdown_links_detailed", return_value=[("text", mock_url, 1)]):
        brokenlinks_mode(
            input_files=[str(f)],
            output_file="-",
            output_format="json"
        )
    captured = capsys.readouterr()
    assert captured.out.strip() == "{}"
