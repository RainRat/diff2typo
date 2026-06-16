import pytest
import os
from multitool import frontmatter_mode

def test_frontmatter_extraction(tmp_path):
    """Test basic frontmatter extraction."""
    md_file = tmp_path / "test.md"
    md_file.write_text("---\ntitle: Hello\ntags:\n  - tech\n  - python\n---\nContent here.")

    output_file = tmp_path / "output.txt"

    # Extract all top-level keys
    frontmatter_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=True,
        key="",
        output_format="line",
        quiet=True
    )

    with open(output_file, 'r') as f:
        lines = f.read().splitlines()

    assert sorted(lines) == ["tags", "title"]

def test_frontmatter_specific_key(tmp_path):
    """Test extraction of a specific top-level key."""
    md_file = tmp_path / "test.md"
    md_file.write_text("---\ntitle: Hello\ntags:\n  - tech\n  - python\n---\nContent here.")

    output_file = tmp_path / "output.txt"

    frontmatter_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=True,
        key="title",
        output_format="line",
        quiet=True
    )

    with open(output_file, 'r') as f:
        lines = f.read().splitlines()

    assert lines == ["hello"]

def test_frontmatter_nested_key(tmp_path):
    """Test extraction of a nested key."""
    md_file = tmp_path / "test.md"
    md_file.write_text("---\nmetadata:\n  author: Alice\n  date: 2023-01-01\n---\nContent.")

    output_file = tmp_path / "output.txt"

    frontmatter_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=True,
        key="metadata.author",
        output_format="line",
        quiet=True
    )

    with open(output_file, 'r') as f:
        lines = f.read().splitlines()

    assert lines == ["alice"]

def test_frontmatter_missing(tmp_path):
    """Test handling of missing frontmatter."""
    md_file = tmp_path / "test.md"
    md_file.write_text("No frontmatter here.")

    output_file = tmp_path / "output.txt"

    frontmatter_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=True,
        key="",
        output_format="line",
        quiet=True
    )

    assert os.path.exists(output_file)
    with open(output_file, 'r') as f:
        assert f.read().strip() == ""

def test_frontmatter_malformed(tmp_path):
    """Test handling of malformed YAML in frontmatter."""
    md_file = tmp_path / "test.md"
    md_file.write_text("---\nmalformed: : yaml\n---\nContent.")

    output_file = tmp_path / "output.txt"

    frontmatter_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=True,
        key="",
        output_format="line",
        quiet=True
    )

    assert os.path.exists(output_file)
    with open(output_file, 'r') as f:
        assert f.read().strip() == ""
