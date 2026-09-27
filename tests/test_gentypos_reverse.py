import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import gentypos


def test_run_typo_generation_reverse_sorting():
    settings_normal = SimpleNamespace(
        min_length=0,
        max_length=None,
        typo_types={'transposition': True, 'deletion': False, 'replacement': False, 'duplication': False},
        transposition_distance=1,
        repeat_modifications=1,
        enable_adjacent_substitutions=True,
        enable_custom_substitutions=True,
        reverse=False,
    )
    settings_reverse = SimpleNamespace(
        min_length=0,
        max_length=None,
        typo_types={'transposition': True, 'deletion': False, 'replacement': False, 'duplication': False},
        transposition_distance=1,
        repeat_modifications=1,
        enable_adjacent_substitutions=True,
        enable_custom_substitutions=True,
        reverse=True,
    )

    adjacent_keys = {}
    custom_subs = {}
    word_list = ["hello"]

    normal_result = gentypos._run_typo_generation(
        word_list,
        set(),
        settings_normal,
        adjacent_keys,
        custom_subs,
        quiet=True,
    )
    reverse_result = gentypos._run_typo_generation(
        word_list,
        set(),
        settings_reverse,
        adjacent_keys,
        custom_subs,
        quiet=True,
    )

    normal_keys = list(normal_result.keys())
    reverse_keys = list(reverse_result.keys())

    assert len(normal_keys) > 1
    assert reverse_keys == list(reversed(normal_keys))


def test_main_cli_reverse_flag(tmp_path):
    output_file = tmp_path / "out.txt"

    test_args = [
        "gentypos.py",
        "hello",
        "-m",
        "0",
        "--no-filter",
        "-t",
        "-R",
        "-f",
        "arrow",
        "--output",
        str(output_file),
    ]

    with patch.object(sys, "argv", test_args):
        gentypos.main()

    content = output_file.read_text(encoding="utf-8").strip().splitlines()
    typos = [line.split(" -> ")[0].strip() for line in content if " -> " in line]

    assert len(typos) > 1
    assert typos == sorted(typos, reverse=True)


def test_main_yaml_reverse_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    output_file = tmp_path / "out.txt"
    words_file = tmp_path / "words.txt"
    words_file.write_text("hello\n", encoding="utf-8")

    config_content = f"""
input_file: "{words_file}"
output_file: "{output_file}"
output_format: "arrow"
reverse: true
dictionary_file: null
word_length:
  min_length: 0
  max_length: 100
typo_types:
  transposition: true
  deletion: false
  replacement: false
  duplication: false
"""
    config_file.write_text(config_content, encoding="utf-8")

    test_args = [
        "gentypos.py",
        "--config",
        str(config_file),
    ]

    with patch.object(sys, "argv", test_args):
        gentypos.main()

    content = output_file.read_text(encoding="utf-8").strip().splitlines()
    typos = [line.split(" -> ")[0].strip() for line in content if " -> " in line]

    assert len(typos) > 1
    assert typos == sorted(typos, reverse=True)


def test_main_dry_run_reverse(tmp_path):
    test_args = [
        "gentypos.py",
        "hello",
        "--no-filter",
        "-t",
        "-R",
        "-n",
    ]

    with patch.object(sys, "argv", test_args):
        with unittest.TestCase().assertRaises(SystemExit) as cm:
            gentypos.main()
        assert cm.exception.code == 0
