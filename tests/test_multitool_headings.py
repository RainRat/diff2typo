import os
import pytest
from multitool import headings_mode

def test_headings_mode_basic(tmp_path, capsys):
    md_file = tmp_path / "test.md"
    md_file.write_text("# H1 Title\nSome text\n## H2 Subtitle\n### H3 Another\n# Another H1 #")

    output_file = tmp_path / "output.txt"

    # Test all headings
    headings_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        clean_items=False,
    )

    content = output_file.read_text().splitlines()
    assert content == ["H1 Title", "H2 Subtitle", "H3 Another", "Another H1"]

def test_headings_mode_level_filter(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# H1\n## H2\n# H1 again")

    output_file = tmp_path / "output.txt"

    # Test level 1 only
    headings_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        level=1,
        clean_items=False,
    )

    content = output_file.read_text().splitlines()
    assert content == ["H1", "H1 again"]

def test_headings_mode_pairs(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# H1\n## H2")

    output_file = tmp_path / "output.txt"

    # Test pairs (default format is line which uses -> for pairs)
    headings_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        pairs=True,
        clean_items=False,
        output_format='line'
    )

    content = output_file.read_text().splitlines()
    assert content == ["1 -> H1", "2 -> H2"]

def test_headings_mode_clean_and_length(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Title 1\n## AB\n### Very Long Heading Here")

    output_file = tmp_path / "output.txt"

    # Test length filter (min 3) and cleaning (remove numbers/spaces)
    headings_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=10,
        process_output=False,
        clean_items=True, # filter_to_letters
    )

    content = output_file.read_text().splitlines()
    # "Title 1" -> "title" (length 5) -> OK
    # "AB" -> "ab" (length 2) -> filtered out
    # "Very Long Heading Here" -> "verylongheadinghere" (length > 10) -> filtered out
    assert content == ["title"]
