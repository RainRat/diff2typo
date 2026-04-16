import sys
from pathlib import Path

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import multitool

def test_casing_mode_basic(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello Hello HELLO world world")

    output_file = tmp_path / "output.txt"

    multitool.casing_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=True,
        output_format='arrow',
        quiet=True
    )

    content = output_file.read_text()
    assert "hello" in content and "HELLO, Hello, hello" in content
    assert "world" not in content

def test_casing_mode_smart_split(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("CamelCase camelcase camelCase")

    output_file = tmp_path / "output.txt"

    # Without smart split, they are different words but filter_to_letters might normalize them
    # Actually filter_to_letters("CamelCase") -> "camelcase"
    multitool.casing_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=True,
        smart=False,
        output_format='arrow',
        quiet=True
    )
    content = output_file.read_text()
    assert "camelcase" in content and "CamelCase, camelCase, camelcase" in content

    # With smart split
    multitool.casing_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=True,
        smart=True,
        output_format='arrow',
        quiet=True
    )
    content = output_file.read_text()
    # "CamelCase" -> ["Camel", "Case"]
    # "camelcase" -> ["camelcase"]
    # "camelCase" -> ["camel", "Case"]
    # Normalized: "camel", "case", "camelcase"
    # "camel" appears as "Camel" and "camel"
    # "case" appears as "Case" and "Case" (no conflict for 'case' if only one variant)
    assert "camel" in content and "Camel, camel" in content

def test_casing_mode_md_table(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("test Test")

    output_file = tmp_path / "output.md"

    multitool.casing_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=True,
        output_format='md-table',
        quiet=True
    )

    content = output_file.read_text()
    assert "| Normalized | Variations |" in content
    assert "| test | Test, test |" in content
