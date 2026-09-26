import os
import sys
from unittest.mock import patch
import pytest

import multitool


def test_unzip_mode_two_files_default(tmp_path):
    pairs_file = tmp_path / "pairs.txt"
    pairs_file.write_text("teh -> the\nwrod -> word\n")

    out_left = tmp_path / "left.txt"
    out_right = tmp_path / "right.txt"

    multitool.unzip_mode(
        input_files=[str(pairs_file)],
        output_file=str(out_left),
        file2=str(out_right),
        min_length=1,
        max_length=100,
        process_output=False,
        right_side=False,
    )

    assert out_left.read_text().splitlines() == ["teh", "wrod"]
    assert out_right.read_text().splitlines() == ["the", "word"]


def test_unzip_mode_two_files_right_side(tmp_path):
    pairs_file = tmp_path / "pairs.txt"
    pairs_file.write_text("teh -> the\nwrod -> word\n")

    out_primary = tmp_path / "primary.txt"
    out_secondary = tmp_path / "secondary.txt"

    multitool.unzip_mode(
        input_files=[str(pairs_file)],
        output_file=str(out_primary),
        file2=str(out_secondary),
        min_length=1,
        max_length=100,
        process_output=False,
        right_side=True,
    )

    # When right_side is True, primary output receives right items, secondary receives left items
    assert out_primary.read_text().splitlines() == ["the", "word"]
    assert out_secondary.read_text().splitlines() == ["teh", "wrod"]


def test_unzip_mode_two_files_process_output(tmp_path):
    pairs_file = tmp_path / "pairs.txt"
    pairs_file.write_text("wrod -> word\nteh -> the\nteh -> the\n")

    out_left = tmp_path / "left.txt"
    out_right = tmp_path / "right.txt"

    multitool.unzip_mode(
        input_files=[str(pairs_file)],
        output_file=str(out_left),
        file2=str(out_right),
        min_length=1,
        max_length=100,
        process_output=True,
        right_side=False,
    )

    assert out_left.read_text().splitlines() == ["teh", "wrod"]
    assert out_right.read_text().splitlines() == ["the", "word"]


def test_unzip_cli_flags(tmp_path):
    pairs_file = tmp_path / "pairs.csv"
    pairs_file.write_text("teh,the\nwrod,word\n")

    out_left = tmp_path / "left.txt"
    out_right = tmp_path / "right.txt"

    test_args = [
        "multitool.py",
        "unzip",
        str(pairs_file),
        "-o",
        str(out_left),
        "--file2",
        str(out_right),
    ]

    with patch.object(sys, "argv", test_args):
        multitool.main()

    assert out_left.read_text().splitlines() == ["teh", "wrod"]
    assert out_right.read_text().splitlines() == ["the", "word"]


def test_unzip_cli_aliases(tmp_path):
    pairs_file = tmp_path / "pairs.csv"
    pairs_file.write_text("teh,the\nwrod,word\n")

    out_left = tmp_path / "left.txt"
    out_right1 = tmp_path / "right1.txt"
    out_right2 = tmp_path / "right2.txt"

    # Test --output2
    test_args1 = [
        "multitool.py",
        "unzip",
        str(pairs_file),
        "-o",
        str(out_left),
        "--output2",
        str(out_right1),
    ]
    with patch.object(sys, "argv", test_args1):
        multitool.main()

    assert out_right1.read_text().splitlines() == ["the", "word"]

    # Test --output-right
    test_args2 = [
        "multitool.py",
        "unzip",
        str(pairs_file),
        "-o",
        str(out_left),
        "--output-right",
        str(out_right2),
    ]
    with patch.object(sys, "argv", test_args2):
        multitool.main()

    assert out_right2.read_text().splitlines() == ["the", "word"]
