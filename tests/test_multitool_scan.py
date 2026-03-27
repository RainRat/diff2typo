import sys
import os
import pytest
import re
from unittest.mock import patch

# Add the repository root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multitool import scan_mode

def strip_ansi(text):
    """Remove ANSI escape sequences from a string."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def test_scan_basic(tmp_path):
    # Mapping file (just words)
    mapping_file = tmp_path / "typos.txt"
    mapping_file.write_text("teh\nrecieve", encoding='utf-8')

    input_file = tmp_path / "input.txt"
    input_file.write_text("This is teh first line.\nI did not recieve it.", encoding='utf-8')

    output_file = tmp_path / "output.txt"

    scan_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    results = [strip_ansi(r) for r in results]

    assert len(results) == 2
    assert "input.txt:1: This is teh first line." in results[0]
    assert "input.txt:2: I did not recieve it." in results[1]

def test_scan_mapping_csv(tmp_path):
    # Mapping file (CSV)
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\nrecieve,receive", encoding='utf-8')

    input_file = tmp_path / "input.txt"
    input_file.write_text("Check teh logic.", encoding='utf-8')

    output_file = tmp_path / "output.txt"

    scan_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    results = [strip_ansi(r) for r in results]

    assert len(results) == 1
    assert "input.txt:1: Check teh logic." in results[0]

def test_scan_smart_mode(tmp_path):
    mapping_file = tmp_path / "typos.txt"
    mapping_file.write_text("teh", encoding='utf-8')

    input_file = tmp_path / "input.txt"
    input_file.write_text("tehVariable = 1\nother_teh_thing = 2", encoding='utf-8')

    output_file = tmp_path / "output.txt"

    scan_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        smart=True,
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    results = [strip_ansi(r) for r in results]

    assert len(results) == 2
    assert "tehVariable" in results[0]
    assert "other_teh_thing" in results[1]

def test_scan_multiple_files(tmp_path):
    mapping_file = tmp_path / "typos.txt"
    mapping_file.write_text("badword", encoding='utf-8')

    f1 = tmp_path / "f1.txt"
    f1.write_text("contains badword", encoding='utf-8')
    f2 = tmp_path / "f2.txt"
    f2.write_text("no match here", encoding='utf-8')
    f3 = tmp_path / "f3.txt"
    f3.write_text("another badword here", encoding='utf-8')

    output_file = tmp_path / "output.txt"

    scan_mode(
        input_files=[str(f1), str(f2), str(f3)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    results = [strip_ansi(r) for r in results]

    assert len(results) == 2
    assert "f1.txt:1: contains badword" in results[0]
    assert "f3.txt:1: another badword here" in results[1]

def test_scan_raw_mode(tmp_path):
    mapping_file = tmp_path / "typos.txt"
    mapping_file.write_text("TeH", encoding='utf-8') # Case sensitive match requested

    input_file = tmp_path / "input.txt"
    input_file.write_text("teh lowercase\nTeH uppercase", encoding='utf-8')

    output_file = tmp_path / "output.txt"

    # With clean_items=False (raw mode), it should only match "TeH"
    scan_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        clean_items=False,
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    results = [strip_ansi(r) for r in results]

    assert len(results) == 1
    assert "TeH uppercase" in results[0]
    assert "teh lowercase" not in "".join(results)

def test_scan_limit(tmp_path):
    mapping_file = tmp_path / "typos.txt"
    mapping_file.write_text("a", encoding='utf-8')

    input_file = tmp_path / "input.txt"
    input_file.write_text("a\na\na\na", encoding='utf-8')

    output_file = tmp_path / "output.txt"

    scan_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        limit=2,
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    assert len(results) == 2
