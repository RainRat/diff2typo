
import os
import re
import sys
import logging
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

import multitool

def test_count_mode_arrow_semantic_colors_more_tags(tmp_path):
    # Covers multitool.py:2336-2339
    f = tmp_path / "test.txt"
    f.write_text("the\n")

    with patch("multitool._extract_pairs", return_value=[("the", "thee"), ("the", "the2"), ("the", "tha"), ("the", "xxx")]), \
         patch("multitool.classify_typo", side_effect=["[Ins]", "[1:2]", "[R]", "[M]"]), \
         patch("multitool._should_enable_color", return_value=True):

        # We need to capture stdout
        out = StringIO()
        with patch("sys.stdout", out):
            multitool.count_mode(
                input_files=[str(f)],
                output_file="-",
                min_length=1,
                max_length=100,
                process_output=False,
                pairs=True,
                output_format='arrow',
                quiet=True
            )
        output = out.getvalue()
        assert "[Ins]" in output
        assert "[1:2]" in output
        assert "[R]" in output
        assert "[M]" in output

def test_case_mode_empty_lines_and_dedup(tmp_path):
    # Covers multitool.py:3124 and 3134
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple\n\napple\n") # Empty line and duplicate
    output_file = tmp_path / "output.txt"

    multitool.case_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True, # Triggers line 3134
        to='upper',
        pairs=False,
        output_format='line',
        quiet=True,
        clean_items=False
    )
    # Deduplicated and empty line skipped
    assert output_file.read_text() == "APPLE\n"

def test_format_search_line_no_parts_v2():
    # Covers multitool.py:3595
    # To have no parts, show_filename and line_numbers must be false
    res = multitool._format_search_line("filename", 0, "line", False, False, False, True)
    assert res == "line"

def test_resolve_full_mapping_gaps(tmp_path):
    # Covers multitool.py:4431, 4437-4438, 4450
    map_file = tmp_path / "map.txt"
    map_file.write_text("# comment\nkey: val\n") # 4431, 4437-4438

    mapping = multitool._resolve_full_mapping(
        str(map_file),
        ad_hoc_pairs=["adhoc"], # 4450
        clean_items=False
    )
    assert mapping["key"] == "val"
    assert mapping["adhoc"] == ""
    assert "# comment" not in mapping

def test_flatten_mode_list_navigation(tmp_path, capsys):
    # Covers multitool.py:4493-4497
    import json
    data = {
        "items": [
            {"name": "a"},
            {"name": "b"}
        ]
    }
    f = tmp_path / "test.json"
    f.write_text(json.dumps(data))

    multitool.flatten_mode(
        input_files=[str(f)],
        output_file="-",
        min_length=1,
        max_length=1000,
        process_output=True,
        key="items.name", # Should navigate into the list
        output_format="line",
        quiet=True,
        clean_items=False
    )

    captured = capsys.readouterr()
    output = captured.out.strip().split("\n")
    # flatten_mode produces key -> value pairs
    # In list mode with key "items.name", it yields from the list.
    # get_sub_data yields from each item in the list if the current path part matches.
    # In our case, parts=["items", "name"].
    # 1. get_sub_data(doc, ["items", "name"])
    # 2. data is dict, "items" in data -> yield from get_sub_data(doc["items"], ["name"])
    # 3. data is list, iterate -> get_sub_data({"name": "a"}, ["name"]) and get_sub_data({"name": "b"}, ["name"])
    # 4. data is dict, "name" in data -> yield from get_sub_data("a", []) -> yield "a"
    # 5. So flatten_mode gets values "a" and "b".
    # 6. _flatten_data("a") yields ("", "a")
    # Output format "line" with pairs (which flatten_mode uses) will show " -> value" if path is empty.
    # Note: ' -> b' has a leading space because of how _write_paired_output formats the arrow.
    # We strip the output lines above, so let's check for the presence of the values in the list.
    assert any("a" in line for line in output)
    assert any("b" in line for line in output)

def test_replace_mode_re_error(caplog):
    # Covers multitool.py:4678-4680
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as cm:
            multitool.replace_mode(["f.txt"], "[", "new", "-", use_regex=True)
    assert cm.value.code == 1
    assert "Invalid regular expression" in caplog.text

def test_replace_mode_stdin_inplace_warning(monkeypatch, caplog):
    # Covers multitool.py:4687
    monkeypatch.setattr(sys, "stdin", StringIO("data"))
    multitool._STDIN_CACHE = None
    with caplog.at_level(logging.WARNING):
        multitool.replace_mode(["-"], "old", "new", "-", in_place="", quiet=True)
    assert "In-place modification requested for standard input" in caplog.text

def test_replace_mode_diff_report(tmp_path):
    # Covers multitool.py:4711
    f = tmp_path / "test.txt"
    f.write_text("old text")
    out = tmp_path / "diff.txt"

    multitool.replace_mode([str(f)], "old", "new", str(out), diff=True, quiet=True)
    assert "---" in out.read_text()
    assert "-old text" in out.read_text()
    assert "+new text" in out.read_text()

def test_rename_mode_missing_path(caplog):
    # Covers multitool.py:5042
    with caplog.at_level(logging.INFO):
        # Should just continue silently if path doesn't exist
        multitool.rename_mode(["nonexistent.txt"], None, "-", 1, 100, False, quiet=True)
    # No "Renamed" log should be present
    assert "Renamed" not in caplog.text

def test_rename_mode_exception(tmp_path, caplog):
    # Covers multitool.py:5060-5062
    f = tmp_path / "test.txt"
    f.write_text("content")

    # We need a mapping that will trigger a rename
    # ad_hoc=["test:new"]

    with patch("os.rename", side_effect=OSError("Permission denied")):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit) as cm:
                multitool.rename_mode([str(f)], None, "-", 1, 100, False, in_place=True, quiet=True, ad_hoc=["test:new"])
    assert cm.value.code == 1
    assert "Failed to rename" in caplog.text

def test_rename_mode_dry_run(tmp_path, caplog):
    # Covers multitool.py:5068
    f = tmp_path / "test.txt"
    f.write_text("content")

    with caplog.at_level(logging.WARNING):
        multitool.rename_mode([str(f)], None, "-", 1, 100, False, in_place=True, dry_run=True, quiet=True, ad_hoc=["test:new"])
    assert "[Dry Run] Would rename" in caplog.text
    assert f.exists()
    assert not (tmp_path / "new.txt").exists()
