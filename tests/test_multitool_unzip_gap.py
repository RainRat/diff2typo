import json
import pytest
from multitool import unzip_mode

def test_unzip_mode_left_extraction(tmp_path):
    input_file = tmp_path / "pairs.txt"
    input_file.write_text("apple -> red\nbanana -> yellow\ncherry -> red")
    output_file = tmp_path / "output.txt"

    unzip_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        right_side=False,
        output_format='line'
    )

    assert output_file.exists()
    content = output_file.read_text().splitlines()
    assert content == ["apple", "banana", "cherry"]

def test_unzip_mode_right_extraction(tmp_path):
    input_file = tmp_path / "pairs.txt"
    input_file.write_text("apple -> red\nbanana -> yellow\ncherry -> red")
    output_file = tmp_path / "output.txt"

    unzip_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        right_side=True,
        output_format='line'
    )

    assert output_file.exists()
    content = output_file.read_text().splitlines()
    assert content == ["red", "yellow", "red"]

def test_unzip_mode_json_keys_extraction(tmp_path):
    input_file = tmp_path / "data.json"
    data = {"keyb": "valb", "keya": "vala"}
    input_file.write_text(json.dumps(data))
    output_file = tmp_path / "output.txt"

    unzip_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        right_side=False,
        output_format='line'
    )

    content = output_file.read_text().splitlines()
    assert content == ["keya", "keyb"]

def test_unzip_mode_json_values_extraction(tmp_path):
    input_file = tmp_path / "data.json"
    data = {"keya": "valb", "keyb": "vala"}
    input_file.write_text(json.dumps(data))
    output_file = tmp_path / "output.txt"

    unzip_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        right_side=True,
        output_format='line'
    )

    content = output_file.read_text().splitlines()
    assert content == ["vala", "valb"]
