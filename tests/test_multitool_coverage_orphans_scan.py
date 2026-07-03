import os
import sys
import logging
from io import StringIO
import pytest
from unittest.mock import patch

# Ensure the repository root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import multitool

def strip_ansi(text):
    import re
    return re.sub(r'\x1b\[[0-9;]*m', '', text)

def test_orphans_mode_arrow_output(tmp_path):
    # Setup files: file1.md references file2.md. file3.md and unused.png are orphans.
    file1 = tmp_path / "file1.md"
    file2 = tmp_path / "file2.md"
    file3 = tmp_path / "file3.md"
    image = tmp_path / "image.png"
    unused = tmp_path / "unused.png"

    file1.write_text("[link](file2.md) ![alt](image.png)")
    file2.write_text("Referenced")
    file3.write_text("Orphan")
    image.write_text("img")
    unused.write_text("unused img")

    output_file = tmp_path / "output.txt"

    with patch.dict(os.environ, {"FORCE_COLOR": "1"}):
        multitool.orphans_mode(
            input_files=[str(file1), str(file2), str(file3), str(image), str(unused)],
            output_file=str(output_file),
            output_format='arrow'
        )

    content = strip_ansi(output_file.read_text())
    assert "Item" in content
    assert "Reason" in content
    assert str(file3) in content
    assert str(unused) in content
    assert "Unreferenced file" in content
    assert "ORPHANS ANALYSIS" in content

def test_orphans_mode_stdin_handling(tmp_path, caplog):
    # Test stdin '-' in input_files to cover line 2760
    input_file = tmp_path / "file1.md"
    input_file.write_text("content")

    with caplog.at_level(logging.INFO):
        multitool.orphans_mode(
            input_files=["-", str(input_file)],
            output_file=str(tmp_path / "out.txt"),
            output_format='line'
        )
    # Stdin is skipped in orphans mode, so it shouldn't crash or cause issues.

def test_orphans_mode_reference_links(tmp_path):
    # Test ref: and broken-ref: coverage (lines 2788, 2790)
    md_file = tmp_path / "links.md"
    # [used]: defined and used via ref
    # [unused]: defined but not used
    # [broken]: used but not defined
    md_file.write_text("[text][used]\n[broken-text][broken]\n\n[used]: ok.md\n[unused]: nop.md")

    target_ok = tmp_path / "ok.md"
    target_ok.write_text("ok")

    output_file = tmp_path / "output.json"

    multitool.orphans_mode(
        input_files=[str(md_file), str(target_ok)],
        output_file=str(output_file),
        output_format='json'
    )

    import json
    result = json.loads(output_file.read_text())

    # [used] is used, so ok.md should NOT be an orphan (referenced via ref:used)
    assert str(target_ok) not in result

    # [unused] is a defined label but not used
    orphan_label = f"{str(md_file)} (label: unused)"
    assert orphan_label in result

    # [broken] is used but not defined.
    # Orphans mode handles it (it's added to used_labels),
    # but since it's not defined, it won't be in defined_labels and won't show as unused definition.

def test_scan_mode_arrow_output(tmp_path):
    # Test scan mode with arrow output to cover lines 6824-6837
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("typo1 -> correction1\ntypo2")

    input_file = tmp_path / "input.txt"
    input_file.write_text("This line has typo1.\nThis line has typo2.")

    output_file = tmp_path / "output.txt"

    with patch.dict(os.environ, {"FORCE_COLOR": "1"}):
        multitool.scan_mode(
            input_files=[str(input_file)],
            mapping_file=str(mapping_file),
            output_file=str(output_file),
            output_format='arrow',
            min_length=1,
            max_length=100,
            process_output=False
        )

    content = strip_ansi(output_file.read_text())
    assert "SCAN ANALYSIS SUMMARY" in content
    assert "Matches found" in content
    assert "Matched files count" in content
    assert "typo1" in content
    assert "typo2" in content
