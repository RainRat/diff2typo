import sys
import os
import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable=None, *_, **__: iterable if iterable is not None else MagicMock())

def test_unzip_mode_default_left(tmp_path):
    input_file = tmp_path / "pairs.txt"
    input_file.write_text("apple -> red\nbanana -> yellow\ncherry -> red\n")
    output_file = tmp_path / "output.txt"

    multitool.unzip_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        right_side=False,
        output_format='line',
        quiet=True
    )

    content = output_file.read_text().splitlines()
    assert content == ["apple", "banana", "cherry"]

def test_unzip_mode_right(tmp_path):
    input_file = tmp_path / "pairs.txt"
    input_file.write_text("apple -> red\nbanana -> yellow\ncherry -> red\n")
    output_file = tmp_path / "output.txt"

    multitool.unzip_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        right_side=True,
        output_format='line',
        quiet=True
    )

    content = output_file.read_text().splitlines()
    assert content == ["red", "yellow", "red"]

def test_unzip_mode_csv(tmp_path):
    input_file = tmp_path / "pairs.csv"
    input_file.write_text("apple,red\nbanana,yellow\n")
    output_file = tmp_path / "output.txt"

    multitool.unzip_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        right_side=True,
        output_format='line',
        quiet=True
    )

    content = output_file.read_text().splitlines()
    assert content == ["red", "yellow"]

def test_unzip_mode_json(tmp_path):
    input_file = tmp_path / "pairs.json"
    data = {"apple": "red", "banana": "yellow"}
    input_file.write_text(json.dumps(data))
    output_file = tmp_path / "output.txt"

    multitool.unzip_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        right_side=False,
        output_format='line',
        quiet=True
    )

    content = output_file.read_text().splitlines()
    # JSON dict keys are sorted in some versions, but unzip_mode uses _extract_pairs
    # which for JSON dict yields k, v in order.
    assert sorted(content) == ["apple", "banana"]

def test_unzip_mode_process_output(tmp_path):
    input_file = tmp_path / "pairs.txt"
    # Duplicate and unsorted
    input_file.write_text("cherry -> red\napple -> red\napple -> red\n")
    output_file = tmp_path / "output.txt"

    multitool.unzip_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True, # Sort and dedup
        right_side=False,
        output_format='line',
        quiet=True
    )

    content = output_file.read_text().splitlines()
    assert content == ["apple", "cherry"]

def test_unzip_mode_filtering(tmp_path):
    input_file = tmp_path / "pairs.txt"
    input_file.write_text("a -> red\napple -> red\n")
    output_file = tmp_path / "output.txt"

    multitool.unzip_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3, # Skip 'a'
        max_length=100,
        process_output=False,
        right_side=False,
        output_format='line',
        quiet=True
    )

    content = output_file.read_text().splitlines()
    assert content == ["apple"]
