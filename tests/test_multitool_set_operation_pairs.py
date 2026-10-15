import sys
import json
import pytest
import multitool

def test_set_operation_pairs_intersection(tmp_path):
    f1 = tmp_path / "file1.txt"
    f2 = tmp_path / "file2.txt"
    out = tmp_path / "out.txt"

    f1.write_text("teh -> the\nwrod -> word\nfoo -> bar\n", encoding="utf-8")
    f2.write_text("wrod -> word\nfoo -> bar\nbaz -> qux\n", encoding="utf-8")

    multitool.set_operation_mode(
        [str(f1)], str(f2), str(out), 1, 100, False, "intersection", pairs=True, output_format="csv"
    )

    content = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 2
    assert "wrod,word" in content
    assert "foo,bar" in content


def test_set_operation_pairs_union(tmp_path):
    f1 = tmp_path / "file1.txt"
    f2 = tmp_path / "file2.txt"
    out = tmp_path / "out.txt"

    f1.write_text("teh -> the\nwrod -> word\n", encoding="utf-8")
    f2.write_text("wrod -> word\nbaz -> qux\n", encoding="utf-8")

    multitool.set_operation_mode(
        [str(f1)], str(f2), str(out), 1, 100, False, "union", pairs=True, output_format="csv"
    )

    content = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 3
    assert "teh,the" in content
    assert "wrod,word" in content
    assert "baz,qux" in content


def test_set_operation_pairs_difference(tmp_path):
    f1 = tmp_path / "file1.txt"
    f2 = tmp_path / "file2.txt"
    out = tmp_path / "out.txt"

    f1.write_text("teh -> the\nwrod -> word\nfoo -> bar\n", encoding="utf-8")
    f2.write_text("wrod -> word\n", encoding="utf-8")

    multitool.set_operation_mode(
        [str(f1)], str(f2), str(out), 1, 100, False, "difference", pairs=True, output_format="csv"
    )

    content = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 2
    assert "teh,the" in content
    assert "foo,bar" in content
    assert "wrod,word" not in content


def test_set_operation_pairs_symmetric_difference(tmp_path):
    f1 = tmp_path / "file1.txt"
    f2 = tmp_path / "file2.txt"
    out = tmp_path / "out.txt"

    f1.write_text("teh -> the\nwrod -> word\n", encoding="utf-8")
    f2.write_text("wrod -> word\nbaz -> qux\n", encoding="utf-8")

    multitool.set_operation_mode(
        [str(f1)], str(f2), str(out), 1, 100, False, "symmetric_difference", pairs=True, output_format="csv"
    )

    content = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 2
    assert "teh,the" in content
    assert "baz,qux" in content
    assert "wrod,word" not in content


def test_set_operation_pairs_cli_integration(monkeypatch, tmp_path):
    f1 = tmp_path / "file1.csv"
    f2 = tmp_path / "file2.csv"
    out = tmp_path / "out.json"

    f1.write_text("teh,the\nwrod,word\n", encoding="utf-8")
    f2.write_text("wrod,word\n", encoding="utf-8")

    args = [
        "multitool.py", "set_operation", str(f1), str(f2),
        "--operation", "difference",
        "-p",
        "--output-format", "json",
        "--output", str(out),
        "-q"
    ]
    monkeypatch.setattr(sys, "argv", args)
    multitool.main()

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data == {"teh": "the"}


def test_set_operation_pairs_process_output_sorting(tmp_path):
    f1 = tmp_path / "file1.txt"
    f2 = tmp_path / "file2.txt"
    out = tmp_path / "out.txt"

    f1.write_text("zbr -> zebra\napple -> fruit\n", encoding="utf-8")
    f2.write_text("banana -> fruit\n", encoding="utf-8")

    multitool.set_operation_mode(
        [str(f1)], str(f2), str(out), 1, 100, True, "union", pairs=True, output_format="csv"
    )

    content = out.read_text(encoding="utf-8").strip().splitlines()
    assert content == ["apple,fruit", "banana,fruit", "zbr,zebra"]
