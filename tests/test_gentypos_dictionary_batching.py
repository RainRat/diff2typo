import os
import sys
import unittest.mock
import pytest
import gentypos


def test_cli_multiple_dictionary_files(tmp_path, monkeypatch, caplog):
    dict1 = tmp_path / "dict1.txt"
    dict1.write_text("wrod\n", encoding="utf-8")

    dict2 = tmp_path / "dict2.txt"
    dict2.write_text("wprd\n", encoding="utf-8")

    out_file = tmp_path / "output.txt"

    test_args = [
        "gentypos.py",
        "word",
        "-d",
        str(dict1),
        str(dict2),
        "-o",
        str(out_file),
        "-f",
        "list",
        "-k",
        "-t",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with caplog.at_level("INFO"):
        gentypos.main()

    # 'wrod' (transposition of word) and 'wprd' (keyboard replacement of word)
    # were both included in the dictionary files, so they should be filtered out.
    content = out_file.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    assert "wrod" not in lines
    assert "wprd" not in lines


def test_cli_directory_and_glob_dictionary(tmp_path, monkeypatch):
    dict_dir = tmp_path / "dicts"
    dict_dir.mkdir()

    d1 = dict_dir / "dict1.txt"
    d1.write_text("wrod\n", encoding="utf-8")

    d2 = dict_dir / "dict2.csv"
    d2.write_text("typo,correction\nwprd,word\n", encoding="utf-8")

    out_file = tmp_path / "output.txt"

    glob_pattern = str(dict_dir / "*.txt")

    test_args = [
        "gentypos.py",
        "word",
        "-d",
        str(dict_dir),
        "-o",
        str(out_file),
        "-f",
        "list",
        "-k",
        "-t",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    gentypos.main()

    content = out_file.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    assert "wrod" not in lines
    assert "wprd" not in lines


def test_yaml_config_list_dictionary(tmp_path, monkeypatch):
    dict1 = tmp_path / "dict1.txt"
    dict1.write_text("wrod\n", encoding="utf-8")

    dict2 = tmp_path / "dict2.txt"
    dict2.write_text("wprd\n", encoding="utf-8")

    words_file = tmp_path / "words.txt"
    words_file.write_text("word\n", encoding="utf-8")

    out_file = tmp_path / "output.txt"

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        f"""
input_file: "{words_file}"
dictionary_file:
  - "{dict1}"
  - "{dict2}"
output_file: "{out_file}"
output_format: "list"
typo_types:
  deletion: false
  transposition: true
  replacement: true
  duplication: false
""",
        encoding="utf-8",
    )

    test_args = ["gentypos.py", "-c", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    gentypos.main()

    content = out_file.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    assert "wrod" not in lines
    assert "wprd" not in lines


def test_dry_run_multi_dictionary(tmp_path, monkeypatch, caplog):
    dict1 = tmp_path / "dict1.txt"
    dict1.write_text("wrod\n", encoding="utf-8")

    dict2 = tmp_path / "dict2.txt"
    dict2.write_text("wprd\n", encoding="utf-8")

    test_args = [
        "gentypos.py",
        "word",
        "-d",
        str(dict1),
        str(dict2),
        "-n",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        with caplog.at_level("INFO"):
            gentypos.main()

    assert exc_info.value.code == 0
    assert "Large Dictionary:" in caplog.text
    assert str(dict1) in caplog.text
    assert str(dict2) in caplog.text
