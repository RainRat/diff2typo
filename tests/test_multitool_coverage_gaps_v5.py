import json
import pytest
import sys
import io
import os
import logging
from unittest.mock import patch
import multitool

def test_flatten_mode_list_navigation(tmp_path, capsys):
    data = {
        "items": [
            {"name": "apple"},
            {"name": "banana"}
        ]
    }
    f = tmp_path / "test.json"
    f.write_text(json.dumps(data))

    # key="items.name" should navigate into the list 'items' and then pick 'name' from each dict
    multitool.flatten_mode(
        input_files=[str(f)],
        output_file="-",
        min_length=1,
        max_length=1000,
        process_output=True,
        key="items.name",
        output_format="line",
        quiet=True,
        clean_items=False
    )

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.splitlines() if line.strip()]
    assert "-> apple" in lines
    assert "-> banana" in lines

def test_count_mode_semantic_coloring_extended(tmp_path, capsys):
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple phone house abcde aple fone teh w")

    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text(
        "apple -> aple\n"    # [Ins]
        "phone -> fone\n"    # [1:2]
        "house -> horse\n"   # [R]
        "abcde -> fghij\n"   # [M]
        "aple -> apple\n"    # [Del]
        "fone -> phone\n"    # [2:1]
        "teh -> the\n"      # [T]
        "w -> s\n"          # [K]
    )

    with patch("multitool._should_enable_color", return_value=True), \
         patch("multitool.MAGENTA", "\033[1;35m"), \
         patch("multitool.RED", "\033[1;31m"), \
         patch("multitool.GREEN", "\033[1;32m"), \
         patch("multitool.YELLOW", "\033[1;33m"), \
         patch("multitool.CYAN", "\033[1;36m"), \
         patch("multitool.BLUE", "\033[1;34m"), \
         patch("multitool.BOLD", "\033[1m"), \
         patch("multitool.RESET", "\033[0m"):
        multitool.count_mode(
            input_files=[str(input_file)],
            output_file="-",
            min_length=1,
            max_length=1000,
            process_output=False,
            mapping_file=str(mapping_file),
            output_format="arrow",
            quiet=False
        )

    captured = capsys.readouterr()
    out = captured.out

    tags = ["[Ins]", "[1:2]", "[R]", "[M]", "[Del]", "[2:1]", "[T]", "[K]"]
    for tag in tags:
        assert tag in out, f"Tag {tag} not found in output. Output was:\n{out}"

    # Verify colors
    assert "\033[1;35m" in out # Magenta [T]
    assert "\033[1;31m" in out # Red [Del], [2:1]
    assert "\033[1;32m" in out # Green [Ins], [1:2]
    assert "\033[1;33m" in out # Yellow [R], [M]
    assert "\033[1;36m" in out # Cyan [K]

def test_replace_mode_regex_error(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")

    # Invalid regex "[" should trigger re.error and sys.exit(1)
    with pytest.raises(SystemExit) as excinfo:
        multitool.replace_mode(
            input_files=[str(f)],
            old_text="[",
            new_text="X",
            output_file="-",
            use_regex=True,
            quiet=True
        )
    assert excinfo.value.code == 1

def test_replace_mode_stdin_inplace(tmp_path, capsys, caplog):
    # Clear cache before test
    multitool._STDIN_CACHE = None

    # Mock stdin
    stdin_content = "hello world\n"

    with patch("sys.stdin", io.StringIO(stdin_content)):
        multitool.replace_mode(
            input_files=["-"],
            old_text="hello",
            new_text="hi",
            output_file="-",
            in_place="",
            diff=True,
            quiet=True
        )

    assert "In-place modification requested for standard input; ignoring." in caplog.text
    captured = capsys.readouterr()
    # Check for diff content in stdout
    assert "--- a/-" in captured.out
    assert "+++ b/-" in captured.out
    assert "-hello world" in captured.out
    assert "+hi world" in captured.out

def test_case_mode_deduplication(tmp_path, capsys):
    f = tmp_path / "test.txt"
    # Duplicate entries that will become the same after case conversion
    f.write_text("Apple\napple\nAPPLE")

    multitool.case_mode(
        input_files=[str(f)],
        to="lower",
        output_file="-",
        process_output=True, # Triggers sorted(set(results))
        min_length=1,
        max_length=100,
        quiet=True
    )

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    assert lines == ["apple"]
