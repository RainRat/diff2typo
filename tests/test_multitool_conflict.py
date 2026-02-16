import pytest
from multitool import conflict_mode
import io
import json
import csv
import os

def test_conflict_mode_basic(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("teh -> the\nteh -> ten\nfoo -> bar\n")

    output_file = tmp_path / "output.txt"

    conflict_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=1000,
        process_output=True,
        output_format='arrow',
        quiet=True,
        clean_items=True
    )

    result = output_file.read_text()
    assert "teh -> ten, the" in result
    assert "foo" not in result

def test_conflict_mode_csv(tmp_path):
    input_file = tmp_path / "test.csv"
    with open(input_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["teh", "the"])
        writer.writerow(["teh", "ten"])
        writer.writerow(["abc", "def"])

    output_file = tmp_path / "output.json"

    conflict_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=1000,
        process_output=True,
        output_format='json',
        quiet=True,
        clean_items=True
    )

    with open(output_file, 'r') as f:
        data = json.load(f)

    assert "teh" in data
    assert data["teh"] == "ten, the"
    assert "abc" not in data

def test_conflict_mode_cleaning(tmp_path):
    input_file = tmp_path / "test.txt"
    # Casing and punctuation should be cleaned by default
    input_file.write_text("Teh -> the\nteh! -> ten\n")

    output_file = tmp_path / "output.txt"

    conflict_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=1000,
        process_output=True,
        output_format='arrow',
        quiet=True,
        clean_items=True
    )

    result = output_file.read_text()
    assert "teh -> ten, the" in result

def test_conflict_mode_raw(tmp_path):
    input_file = tmp_path / "test.txt"
    # With --raw, Teh and teh are different
    input_file.write_text("Teh -> the\nteh -> ten\nTeh -> then\n")

    output_file = tmp_path / "output.txt"

    conflict_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=1000,
        process_output=True,
        output_format='arrow',
        quiet=True,
        clean_items=False
    )

    result = output_file.read_text()
    assert "Teh -> the, then" in result
    assert "teh" not in result # teh is not a conflict because it only has one correction here

def test_conflict_mode_min_length(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("a -> b\na -> c\nteh -> the\nteh -> ten\n")

    output_file = tmp_path / "output.txt"

    # min_length=3 should filter out 'a'
    conflict_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=1000,
        process_output=True,
        output_format='arrow',
        quiet=True,
        clean_items=True
    )

    result = output_file.read_text()
    assert "teh -> ten, the" in result
    assert "a ->" not in result
