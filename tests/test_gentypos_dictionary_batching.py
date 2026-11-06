import os
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos


def test_dictionary_multiple_files_cli(tmp_path, capsys):
    dict1 = tmp_path / "dict1.txt"
    dict2 = tmp_path / "dict2.txt"

    # "appl" is deletion typo of "apple"; "spple" is adjacent replacement of 'a' with 's'
    dict1.write_text("appl\n", encoding="utf-8")
    dict2.write_text("spple\n", encoding="utf-8")

    test_args = [
        "gentypos.py",
        "apple",
        "-d", str(dict1), str(dict2),
        "-D", "-k",
        "--min-length", "3",
        "-f", "arrow",
    ]

    with patch.object(sys, "argv", test_args):
        gentypos.main()

    captured = capsys.readouterr()
    output_lines = captured.out.strip().splitlines()

    # "appl" and "spple" should both be filtered out because they exist in dict1 and dict2
    assert not any(line.startswith("appl ->") for line in output_lines)
    assert not any(line.startswith("spple ->") for line in output_lines)
    # Other typos (like "wpple" or "qpple" or "aple") should still be generated
    assert any("-> apple" in line for line in output_lines)


def test_dictionary_directory_scan(tmp_path, capsys):
    dict_dir = tmp_path / "dictionaries"
    dict_dir.mkdir()

    d1 = dict_dir / "dict_a.txt"
    d2 = dict_dir / "dict_b.csv"

    d1.write_text("appl\n", encoding="utf-8")
    d2.write_text("spple\n", encoding="utf-8")

    test_args = [
        "gentypos.py",
        "apple",
        "-d", str(dict_dir),
        "-D", "-k",
        "--min-length", "3",
        "-f", "arrow",
    ]

    with patch.object(sys, "argv", test_args):
        gentypos.main()

    captured = capsys.readouterr()
    output_lines = captured.out.strip().splitlines()

    assert not any(line.startswith("appl ->") for line in output_lines)
    assert not any(line.startswith("spple ->") for line in output_lines)
    assert any("-> apple" in line for line in output_lines)


def test_dictionary_glob_pattern(tmp_path, capsys):
    dict_dir = tmp_path / "glob_dicts"
    dict_dir.mkdir()

    d1 = dict_dir / "part1.txt"
    d2 = dict_dir / "part2.txt"
    ignored = dict_dir / "ignored.other"

    d1.write_text("appl\n", encoding="utf-8")
    d2.write_text("spple\n", encoding="utf-8")
    ignored.write_text("aple\n", encoding="utf-8")

    glob_pattern = str(dict_dir / "*.txt")

    test_args = [
        "gentypos.py",
        "apple",
        "-d", glob_pattern,
        "-D", "-k",
        "--min-length", "3",
        "-f", "arrow",
    ]

    with patch.object(sys, "argv", test_args):
        gentypos.main()

    captured = capsys.readouterr()
    output_lines = captured.out.strip().splitlines()

    # appl and spple were in *.txt and should be filtered
    assert not any(line.startswith("appl ->") for line in output_lines)
    assert not any(line.startswith("spple ->") for line in output_lines)
    # aple was in ignored.other, so it shouldn't be loaded from the glob
    assert any(line.startswith("aple ->") for line in output_lines)


def test_dictionary_yaml_list_config(tmp_path, capsys):
    d1 = tmp_path / "dict1.txt"
    d2 = tmp_path / "dict2.txt"

    d1.write_text("appl\n", encoding="utf-8")
    d2.write_text("spple\n", encoding="utf-8")

    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        f"""
input_file: null
dictionary_file:
  - "{str(d1)}"
  - "{str(d2)}"
output_file: "-"
output_format: "arrow"
typo_types:
  deletion: true
  transposition: false
  replacement: true
  duplication: false
word_length:
  min_length: 3
  max_length: null
""",
        encoding="utf-8",
    )

    test_args = [
        "gentypos.py",
        "apple",
        "-c", str(config_file),
    ]

    with patch.object(sys, "argv", test_args):
        gentypos.main()

    captured = capsys.readouterr()
    output_lines = captured.out.strip().splitlines()

    assert not any(line.startswith("appl ->") for line in output_lines)
    assert not any(line.startswith("spple ->") for line in output_lines)
    assert any("-> apple" in line for line in output_lines)


def test_dictionary_cli_missing_fallback(tmp_path, caplog):
    import logging

    non_existent = tmp_path / "non_existent_dict.txt"

    test_args = [
        "gentypos.py",
        "hello",
        "-d", str(non_existent),
    ]

    with patch.object(sys, "argv", test_args):
        with caplog.at_level(logging.INFO):
            gentypos.main()

    assert any("not found; skipping large dictionary filtering" in record.message for record in caplog.records)
