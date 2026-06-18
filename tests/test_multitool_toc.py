import os
import pytest
from multitool import toc_mode, _slugify

def test_slugify():
    assert _slugify("Hello World") == "hello-world"
    assert _slugify("Markdown Header!!") == "markdown-header"
    assert _slugify("Multiple   Spaces") == "multiple-spaces"
    assert _slugify("Underscore_Test") == "underscore_test"
    assert _slugify("Numbers 123") == "numbers-123"
    assert _slugify("-Leading and Trailing-") == "leading-and-trailing"
    assert _slugify("Special !@#$%^&*() Characters") == "special-characters"

def test_toc_mode_basic(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Title\n## Section 1\n### Subsection\n## Section 2")

    output_file = tmp_path / "output.txt"

    toc_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        clean_items=False,
    )

    content = output_file.read_text().splitlines()
    expected = [
        "- [Title](#title)",
        "  - [Section 1](#section-1)",
        "    - [Subsection](#subsection)",
        "  - [Section 2](#section-2)"
    ]
    assert content == expected

def test_toc_mode_duplicates(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Section\n# Section\n# Section")

    output_file = tmp_path / "output.txt"

    toc_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        clean_items=False,
    )

    content = output_file.read_text().splitlines()
    expected = [
        "- [Section](#section)",
        "- [Section](#section-1)",
        "- [Section](#section-2)"
    ]
    assert content == expected

def test_toc_mode_no_links(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Title\n## Section")

    output_file = tmp_path / "output.txt"

    toc_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        no_links=True,
        clean_items=False,
    )

    content = output_file.read_text().splitlines()
    expected = [
        "- Title",
        "  - Section"
    ]
    assert content == expected

def test_toc_mode_level_filter(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# H1\n## H2\n### H3")

    output_file = tmp_path / "output.txt"

    # Filter for level 2 only
    toc_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        level=2,
        clean_items=False,
    )

    content = output_file.read_text().splitlines()
    expected = ["  - [H2](#h2)"]
    assert content == expected

def test_toc_mode_clean_and_length(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Title 1\n## AB\n### Long Header")

    output_file = tmp_path / "output.txt"

    toc_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=10,
        process_output=False,
        clean_items=True,
    )

    content = output_file.read_text().splitlines()
    # "Title 1" -> "title" (len 5) -> OK
    # "AB" -> "ab" (len 2) -> Too short
    # "Long Header" -> "longheader" (len 10) -> OK
    expected = [
        "- [title](#title-1)",
        "    - [longheader](#long-header)"
    ]
    assert content == expected
