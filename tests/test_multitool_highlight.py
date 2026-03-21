import os
import pytest
from multitool import highlight_mode, YELLOW, RESET

def test_highlight_mode_basic(tmp_path):
    # Create input file
    input_file = tmp_path / "input.txt"
    input_file.write_text("Hello world, this is a test.\nAnother line with teh typo.")

    # Create mapping file
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\nworld,Earth")

    output_file = tmp_path / "output.txt"

    highlight_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        clean_items=True,
        smart=False
    )

    content = output_file.read_text()
    assert f"{YELLOW}world{RESET}" in content
    assert f"{YELLOW}teh{RESET}" in content
    assert "Hello" in content
    assert "typo" in content

def test_highlight_mode_list(tmp_path):
    # Create input file
    input_file = tmp_path / "input.txt"
    input_file.write_text("The quick brown fox jumps over the lazy dog.")

    # Create word list file
    list_file = tmp_path / "words.txt"
    list_file.write_text("quick\nfox\nlazy")

    output_file = tmp_path / "output.txt"

    highlight_mode(
        input_files=[str(input_file)],
        mapping_file=str(list_file),
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        clean_items=True,
        smart=False
    )

    content = output_file.read_text()
    assert f"{YELLOW}quick{RESET}" in content
    assert f"{YELLOW}fox{RESET}" in content
    assert f"{YELLOW}lazy{RESET}" in content
    assert "brown" in content

def test_highlight_mode_smart(tmp_path):
    # Create input file
    input_file = tmp_path / "input.txt"
    input_file.write_text("This is a tehVariable and anotherTehExample.")

    # Create mapping file
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the")

    output_file = tmp_path / "output.txt"

    # Run with smart=True
    highlight_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        clean_items=True,
        smart=True
    )

    content = output_file.read_text()
    assert f"{YELLOW}teh{RESET}Variable" in content
    assert f"another{YELLOW}Teh{RESET}Example" in content

def test_highlight_mode_raw(tmp_path):
    # Create input file
    input_file = tmp_path / "input.txt"
    input_file.write_text("Hello WORLD.")

    # Create mapping file (case sensitive if raw=True)
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("WORLD")

    output_file = tmp_path / "output.txt"

    # Run with clean_items=False (Raw mode)
    highlight_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        clean_items=False,
        smart=False
    )

    content = output_file.read_text()
    assert f"{YELLOW}WORLD{RESET}" in content

    # Now run with clean_items=True (Default) but mapping is uppercase
    highlight_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        clean_items=True,
        smart=False
    )

    content = output_file.read_text()
    # WORLD (input) should be cleaned to world,
    # WORLD (mapping) should be cleaned to world.
    # Match should happen.
    assert f"{YELLOW}WORLD{RESET}" in content
