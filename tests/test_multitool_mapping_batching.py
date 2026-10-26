"""
Tests for batching mapping sources (-s/--mapping) in multitool.py across multiple modes.
"""

import sys
from unittest.mock import patch
import pytest
import multitool


def test_resolve_full_mapping_multiple_files(tmp_path):
    map1 = tmp_path / "map1.csv"
    map2 = tmp_path / "map2.json"

    map1.write_text("teh,the\nwrod,word\n", encoding="utf-8")
    map2.write_text('{"recieve": "receive", "freind": "friend"}', encoding="utf-8")

    mapping = multitool._resolve_full_mapping([str(map1), str(map2)], ad_hoc_pairs=None, clean_items=True)
    assert mapping["teh"] == "the"
    assert mapping["wrod"] == "word"
    assert mapping["recieve"] == "receive"
    assert mapping["freind"] == "friend"


def test_resolve_full_mapping_directory(tmp_path):
    map_dir = tmp_path / "mappings"
    map_dir.mkdir()

    m1 = map_dir / "a.txt"
    m2 = map_dir / "b.csv"

    m1.write_text("foo -> bar\n", encoding="utf-8")
    m2.write_text("baz,qux\n", encoding="utf-8")

    mapping = multitool._resolve_full_mapping(str(map_dir), ad_hoc_pairs=None, clean_items=True)
    assert mapping["foo"] == "bar"
    assert mapping["baz"] == "qux"


def test_resolve_full_mapping_glob_pattern(tmp_path):
    m1 = tmp_path / "batch_1.csv"
    m2 = tmp_path / "batch_2.csv"

    m1.write_text("a,b\n", encoding="utf-8")
    m2.write_text("c,d\n", encoding="utf-8")

    glob_pattern = str(tmp_path / "batch_*.csv")
    mapping = multitool._resolve_full_mapping(glob_pattern, ad_hoc_pairs=None, clean_items=True)
    assert mapping["a"] == "b"
    assert mapping["c"] == "d"


def test_map_mode_batch_mapping(tmp_path, capsys):
    m1 = tmp_path / "m1.csv"
    m2 = tmp_path / "m2.csv"
    inp = tmp_path / "input.txt"

    m1.write_text("teh,the\n", encoding="utf-8")
    m2.write_text("wrod,word\n", encoding="utf-8")
    inp.write_text("teh\nwrod\nother\n", encoding="utf-8")

    test_args = [
        "multitool.py",
        "map",
        str(inp),
        "-s", str(m1), str(m2),
        "--raw",
    ]

    with patch.object(sys, "argv", test_args):
        multitool.main()

    captured = capsys.readouterr()
    assert "the" in captured.out
    assert "word" in captured.out
    assert "other" in captured.out


def test_scrub_mode_batch_mapping(tmp_path):
    m1 = tmp_path / "m1.csv"
    m2 = tmp_path / "m2.csv"
    inp = tmp_path / "input.txt"

    m1.write_text("teh,the\n", encoding="utf-8")
    m2.write_text("wrod,word\n", encoding="utf-8")
    inp.write_text("Fix teh wrod in this sentence.\n", encoding="utf-8")

    out_file = tmp_path / "out.txt"

    test_args = [
        "multitool.py",
        "scrub",
        str(inp),
        "-s", str(m1), str(m2),
        "-o", str(out_file),
        "-R",
    ]

    with patch.object(sys, "argv", test_args):
        multitool.main()

    assert out_file.read_text(encoding="utf-8").strip() == "Fix the word in this sentence."


def test_scan_mode_batch_mapping(tmp_path, capsys):
    m1 = tmp_path / "m1.txt"
    m2 = tmp_path / "m2.txt"
    inp = tmp_path / "input.txt"

    m1.write_text("teh\n", encoding="utf-8")
    m2.write_text("wrod\n", encoding="utf-8")
    inp.write_text("1: Here is teh word.\n2: Here is a wrod.\n", encoding="utf-8")

    test_args = [
        "multitool.py",
        "scan",
        str(inp),
        "-s", str(m1), str(m2),
        "-n",
        "--raw",
    ]

    with patch.object(sys, "argv", test_args):
        multitool.main()

    captured = capsys.readouterr()
    assert "teh" in captured.out or "teh" in captured.err
    assert "wrod" in captured.out or "wrod" in captured.err


def test_verify_mode_batch_mapping(tmp_path, capsys):
    m1 = tmp_path / "m1.csv"
    m2 = tmp_path / "m2.csv"
    inp = tmp_path / "input.txt"

    m1.write_text("teh,the\n", encoding="utf-8")
    m2.write_text("wrod,word\n", encoding="utf-8")
    inp.write_text("There is teh first typo.\n", encoding="utf-8")

    test_args = [
        "multitool.py",
        "verify",
        str(inp),
        "-s", str(m1), str(m2),
        "--prune",
        "--raw",
    ]

    with patch.object(sys, "argv", test_args):
        multitool.main()

    captured = capsys.readouterr()
    assert "teh" in captured.out
    assert "wrod" not in captured.out
