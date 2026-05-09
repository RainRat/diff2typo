from unittest.mock import MagicMock

import sys
import pytest
from pathlib import Path
import logging

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    monkeypatch.setattr(multitool, "tqdm", lambda iterable=None, *_, **__: iterable if iterable is not None else MagicMock())

def test_extract_arrow_items_coverage(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("left -> right\nno arrow here\n  spaced -> items  ")

    # Left side
    results = list(multitool._extract_arrow_items(str(f), right_side=False))
    assert results == ["left", "spaced"]

    # Right side
    results = list(multitool._extract_arrow_items(str(f), right_side=True))
    assert results == ["right", "items"]

def test_extract_table_items_coverage(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text('key1 = "value1"\nno table here\nkey2 = "quoted " value"')

    # Left side
    results = list(multitool._extract_table_items(str(f), right_side=False))
    assert results == ["key1", "key2"]

    # Right side
    results = list(multitool._extract_table_items(str(f), right_side=True))
    assert results == ["value1", "quoted \" value"]

def test_extract_backtick_items_coverage(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("some `code` here\nerror: in `bad_file.py`\n`plain` and `more`\n` ` empty\nno backticks")

    # Heuristics: prioritizes items near 'error:', 'warning:', 'note:'
    results = list(multitool._extract_backtick_items(str(f)))
    # bad_file.py should be prioritized if present, but the function yields ALL
    # from prioritize if ANY prioritized found in line.
    # Line 1: code
    # Line 2: bad_file.py (prioritized)
    # Line 3: plain, more
    # Line 4: empty string skipped
    # Line 5: len(parts) < 3, triggers continue
    assert "code" in results
    assert "bad_file.py" in results
    assert "plain" in results
    assert "more" in results

def test_arrow_mode_wrapper_coverage(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("a -> b")
    out = tmp_path / "out.txt"
    # This covers the wrapper and the nested extractor function
    multitool.arrow_mode([str(f)], str(out), 1, 100, False)
    assert out.read_text().strip() == "a"

def test_extract_csv_items_coverage(tmp_path):
    f = tmp_path / "test.csv"
    f.write_text("a,b,c\n1,2,3")

    # First column
    results = list(multitool._extract_csv_items(str(f), first_column=True))
    assert results == ["a", "1"]

    # Remaining columns
    results = list(multitool._extract_csv_items(str(f), first_column=False))
    assert results == ["b", "c", "2", "3"]

def test_check_mode_coverage(tmp_path, caplog):
    f = tmp_path / "typos.csv"
    # typo1 is also a correction for typo2
    f.write_text("typoa,corra\ntypob,typoa\n\n") # \n\n triggers 'if not row: continue'

    out = tmp_path / "out.txt"
    with caplog.at_level(logging.INFO):
        multitool.check_mode([str(f)], str(out), 1, 100, True)

    assert "typoa" in out.read_text()
    assert "Found 1 overlapping words" in caplog.text

def test_line_mode_coverage(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("linea\nlineb")
    out = tmp_path / "out.txt"
    multitool.line_mode([str(f)], str(out), 1, 100, False)
    assert out.read_text().strip() == "linea\nlineb"

def test_combine_mode_coverage(tmp_path):
    f1 = tmp_path / "f1.txt"
    f1.write_text("apple\nbanana")
    f2 = tmp_path / "f2.txt"
    f2.write_text("banana\ncherry")
    out = tmp_path / "out.txt"

    multitool.combine_mode([str(f1), str(f2)], str(out), 1, 100, False)
    # combine_mode sorts and dedups
    assert out.read_text().strip() == "apple\nbanana\ncherry"

def test_set_operation_mode_union(tmp_path):
    f1 = tmp_path / "f1.txt"
    f1.write_text("a\nb")
    f2 = tmp_path / "f2.txt"
    f2.write_text("b\nc")
    out = tmp_path / "out.txt"

    # process_output=True covers line 4259
    multitool.set_operation_mode([str(f1)], str(f2), str(out), 1, 100, True, operation='union')
    assert "a" in out.read_text()
    assert "b" in out.read_text()
    assert "c" in out.read_text()

def test_set_operation_mode_difference(tmp_path):
    f1 = tmp_path / "f1.txt"
    f1.write_text("a\nb")
    f2 = tmp_path / "f2.txt"
    f2.write_text("b\nc")
    out = tmp_path / "out.txt"

    multitool.set_operation_mode([str(f1)], str(f2), str(out), 1, 100, False, operation='difference')
    assert out.read_text().strip() == "a"
