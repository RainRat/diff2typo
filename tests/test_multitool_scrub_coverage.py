import sys
import pytest
from pathlib import Path

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable=None, *_, **__: iterable if iterable is not None else MagicMock())

@pytest.fixture
def mapping_file(tmp_path):
    f = tmp_path / "mapping.csv"
    f.write_text("teh,the\nrecieve,receive\napple,fruit\nv1,version1\nvar,variable\n")
    return str(f)

def test_scrub_mode_basic_mixed_case_replacement(tmp_path, mapping_file):
    input_file = tmp_path / "input.txt"
    input_file.write_text("I recieve teh apple.\nTeh apple is red.\n")
    output_file = tmp_path / "output.txt"

    multitool.scrub_mode(
        [str(input_file)], mapping_file, str(output_file),
        min_length=1, max_length=100, process_output=False
    )
    assert output_file.read_text() == "I receive the fruit.\nthe fruit is red.\n"

def test_scrub_mode_subword_replacement_camel_and_snake(tmp_path, mapping_file):
    input_file = tmp_path / "input.txt"
    input_file.write_text("myTehVariable = some_apple_value\n")
    output_file = tmp_path / "output.txt"

    multitool.scrub_mode(
        [str(input_file)], mapping_file, str(output_file),
        min_length=1, max_length=100, process_output=False
    )
    assert output_file.read_text() == "mytheVariable = some_fruit_value\n"

def test_scrub_mode_raw_case_sensitive_replacement(tmp_path, mapping_file):
    input_file = tmp_path / "input.txt"
    input_file.write_text("I recieve teh apple.\nTeh apple is red.\n")
    output_file = tmp_path / "output.txt"

    multitool.scrub_mode(
        [str(input_file)], mapping_file, str(output_file),
        min_length=1, max_length=100, process_output=False, clean_items=False
    )
    assert output_file.read_text() == "I receive the fruit.\nTeh fruit is red.\n"

def test_scrub_mode_line_limit(tmp_path, mapping_file):
    input_file = tmp_path / "input.txt"
    input_file.write_text("Line 1 teh\nLine 2 teh\n")
    output_file = tmp_path / "output.txt"

    multitool.scrub_mode(
        [str(input_file)], mapping_file, str(output_file),
        min_length=1, max_length=100, process_output=False, limit=1
    )
    assert output_file.read_text() == "Line 1 the\n"

def test_scrub_mode_multiple_input_files(tmp_path, mapping_file):
    input1 = tmp_path / "input1.txt"
    input1.write_text("file1 teh\n")
    input2 = tmp_path / "input2.txt"
    input2.write_text("file2 teh\n")
    output_file = tmp_path / "output.txt"

    multitool.scrub_mode(
        [str(input1), str(input2)], mapping_file, str(output_file),
        min_length=1, max_length=100, process_output=False
    )
    assert output_file.read_text() == "file1 the\nfile2 the\n"

def test_scrub_mode_empty_mapping(tmp_path):
    mapping = tmp_path / "empty.csv"
    mapping.write_text("")
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh\n")
    output_file = tmp_path / "output.txt"

    multitool.scrub_mode(
        [str(input_file)], str(mapping), str(output_file),
        min_length=1, max_length=100, process_output=False
    )
    assert output_file.read_text() == "teh\n"

def test_scrub_mode_missing_trailing_newline_restoration(tmp_path, mapping_file):
    input_file = tmp_path / "no_nl.txt"
    input_file.write_text("teh")
    output_file = tmp_path / "output.txt"

    multitool.scrub_mode(
        [str(input_file)], mapping_file, str(output_file),
        min_length=1, max_length=100, process_output=False
    )
    assert output_file.read_text() == "the\n"

def test_scrub_mode_complex_subword_delimiters(tmp_path, mapping_file):
    input_file = tmp_path / "input.txt"
    input_file.write_text("MyTehVar_123! @special\n")
    output_file = tmp_path / "output.txt"

    multitool.scrub_mode(
        [str(input_file)], mapping_file, str(output_file),
        min_length=1, max_length=100, process_output=False
    )
    assert output_file.read_text() == "Mythevariable_123! @special\n"

def test_scrub_mode_numeric_tokens(tmp_path, mapping_file):
    input_file = tmp_path / "input.txt"
    input_file.write_text("App v1.0 released (v1)\n")
    output_file = tmp_path / "output.txt"

    multitool.scrub_mode(
        [str(input_file)], mapping_file, str(output_file),
        min_length=1, max_length=100, process_output=False
    )
    assert output_file.read_text() == "App version1.0 released (version1)\n"
