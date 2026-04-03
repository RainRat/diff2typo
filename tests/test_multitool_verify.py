import sys
import os
import re
import logging
from pathlib import Path
import pytest

# Add parent directory to sys.path to import multitool
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

def strip_ansi(text):
    """Remove ANSI escape sequences from text."""
    return re.sub(r'\x1b\[[0-9;]*m', '', text)

def test_verify_mode_basic(tmp_path, caplog):
    """Test verify mode identifies present and absent typos."""
    caplog.set_level(logging.INFO)
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\nmisspeled,misspelled\nabsent,present")

    content_file = tmp_path / "content.txt"
    content_file.write_text("This is teh content with a misspeled word.")

    output_file = tmp_path / "report.txt"

    multitool.verify_mode(
        input_files=[str(content_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        ad_hoc=None,
        smart=False,
        prune=False,
        clean_items=True
    )

    report = output_file.read_text()
    # Output should contain all mappings with status
    assert "teh -> FOUND (1)" in strip_ansi(report)
    assert "misspeled -> FOUND (1)" in strip_ansi(report)
    assert "absent -> NOT FOUND" in strip_ansi(report)

    # Check log for summary - using regex to handle multiple spaces
    assert re.search(r"Active mappings found:\s+2", caplog.text)
    assert re.search(r"Stale mappings \(not found\):\s+1", caplog.text)

def test_verify_mode_prune(tmp_path):
    """Test verify mode with --prune flag."""
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\nabsent,present")

    content_file = tmp_path / "content.txt"
    content_file.write_text("I have teh typo.")

    output_file = tmp_path / "pruned.csv"

    multitool.verify_mode(
        input_files=[str(content_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        ad_hoc=None,
        smart=False,
        prune=True,
        clean_items=True
    )

    # Pruned file should only contain found typos
    pruned_content = output_file.read_text()
    assert "teh -> the" in strip_ansi(pruned_content)
    assert "absent" not in pruned_content

def test_verify_mode_smart(tmp_path):
    """Test verify mode with --smart flag for subword matching."""
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the")

    content_file = tmp_path / "content.txt"
    # Use proper CamelCase where 'Teh' is a subword
    content_file.write_text("ThisIsTehCamelCaseWord")

    output_file = tmp_path / "report.txt"

    # Without smart matching
    multitool.verify_mode(
        input_files=[str(content_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        smart=False,
        prune=False
    )
    assert "teh -> NOT FOUND" in strip_ansi(output_file.read_text())

    # With smart matching
    multitool.verify_mode(
        input_files=[str(content_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        smart=True,
        prune=False
    )
    assert "teh -> FOUND (1)" in strip_ansi(output_file.read_text())

def test_verify_mode_ad_hoc(tmp_path):
    """Test verify mode with ad-hoc pairs."""
    content_file = tmp_path / "content.txt"
    content_file.write_text("find me")

    output_file = tmp_path / "report.txt"

    multitool.verify_mode(
        input_files=[str(content_file)],
        mapping_file=None,
        ad_hoc=["find:found", "missing:gone"],
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        prune=False
    )

    report = strip_ansi(output_file.read_text())
    assert "find -> FOUND (1)" in report
    assert "missing -> NOT FOUND" in report

def test_verify_mode_empty_mapping(tmp_path):
    """Test verify mode exits when no mapping is found."""
    with pytest.raises(SystemExit) as excinfo:
        multitool.verify_mode(
            input_files=["-"],
            mapping_file=None,
            ad_hoc=None,
            output_file="-",
            min_length=1,
            max_length=1000,
            process_output=False
        )
    assert excinfo.value.code == 1
