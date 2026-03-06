import subprocess
import os
import pytest

def test_scrub_basic(tmp_path):
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\nrecieve,receive\n")

    input_file = tmp_path / "input.txt"
    input_file.write_text("I recieve teh message.\n")

    output_file = tmp_path / "output.txt"

    subprocess.run(
        ["python3", "multitool.py", "scrub", str(input_file), "--mapping", str(mapping_file), "--output", str(output_file)],
        check=True
    )

    assert output_file.read_text() == "I receive the message.\n"

def test_scrub_preserve_context(tmp_path):
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\n")

    input_file = tmp_path / "input.txt"
    # Testing punctuation and mixed case (if clean_items is True, 'Teh' matches 'teh')
    input_file.write_text("Teh! (teh) \"teh\"...\n")

    output_file = tmp_path / "output.txt"

    subprocess.run(
        ["python3", "multitool.py", "scrub", str(input_file), "--mapping", str(mapping_file), "--output", str(output_file)],
        check=True
    )

    # Note: currently scrub mode replaces with exact mapping value.
    # If mapping is teh->the, then 'Teh' becomes 'the' if clean_items is True.
    assert output_file.read_text() == "the! (the) \"the\"...\n"

def test_scrub_subword(tmp_path):
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\n")

    input_file = tmp_path / "input.txt"
    input_file.write_text("my_teh_var = teh_value\n")

    output_file = tmp_path / "output.txt"

    subprocess.run(
        ["python3", "multitool.py", "scrub", str(input_file), "--mapping", str(mapping_file), "--output", str(output_file)],
        check=True
    )

    assert output_file.read_text() == "my_the_var = the_value\n"

def test_scrub_raw_mode(tmp_path):
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("Teh,the\n") # Case sensitive mapping

    input_file = tmp_path / "input.txt"
    input_file.write_text("Teh teh\n")

    output_file = tmp_path / "output.txt"

    # Run with --raw
    subprocess.run(
        ["python3", "multitool.py", "scrub", str(input_file), "--mapping", str(mapping_file), "--output", str(output_file), "--raw"],
        check=True
    )

    # Only 'Teh' should be replaced
    assert output_file.read_text() == "the teh\n"
