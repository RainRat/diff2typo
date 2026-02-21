import pytest
from multitool import main
import sys
import os

def test_md_table_single_column(tmp_path):
    input_file = tmp_path / "items.txt"
    input_file.write_text("apple\nbanana\n")
    output_file = tmp_path / "output.md"

    sys.argv = ["multitool.py", "line", str(input_file), "-o", str(output_file), "-f", "md-table"]
    main()

    content = output_file.read_text()
    assert "| Item |" in content
    assert "| :--- |" in content
    assert "| apple |" in content
    assert "| banana |" in content

def test_md_table_pairs(tmp_path):
    input_file = tmp_path / "pairs.txt"
    input_file.write_text("teh -> the\nwod -> word\n")
    output_file = tmp_path / "output.md"

    sys.argv = ["multitool.py", "pairs", str(input_file), "-o", str(output_file), "-f", "md-table"]
    main()

    content = output_file.read_text()
    assert "| Typo | Correction |" in content
    assert "| :--- | :--- |" in content
    assert "| teh | the |" in content
    assert "| wod | word |" in content

def test_md_table_count(tmp_path):
    input_file = tmp_path / "items.txt"
    input_file.write_text("apple\napple\nbanana\n")
    output_file = tmp_path / "output.md"

    sys.argv = ["multitool.py", "count", str(input_file), "-o", str(output_file), "-f", "md-table"]
    main()

    content = output_file.read_text()
    assert "| Item | Count |" in content
    assert "| apple | 2 |" in content
    assert "| banana | 1 |" in content

def test_md_table_stats(tmp_path):
    input_file = tmp_path / "items.txt"
    input_file.write_text("apple\nbanana\n")
    output_file = tmp_path / "output.md"

    sys.argv = ["multitool.py", "stats", str(input_file), "-o", str(output_file), "-f", "md-table"]
    main()

    content = output_file.read_text()
    assert "### ANALYSIS STATISTICS" in content
    assert "| Metric | Value |" in content
    assert "| Total items encountered | 2 |" in content
    assert "| Unique items | 2 |" in content

def test_markdown_format_stats(tmp_path):
    # Verify that existing 'markdown' format also works for stats and produces md tables
    input_file = tmp_path / "items.txt"
    input_file.write_text("apple\n")
    output_file = tmp_path / "output.md"

    sys.argv = ["multitool.py", "stats", str(input_file), "-o", str(output_file), "-f", "markdown"]
    main()

    content = output_file.read_text()
    assert "### ANALYSIS STATISTICS" in content
    assert "| Metric | Value |" in content

def test_md_table_conflict(tmp_path):
    input_file = tmp_path / "conflicts.txt"
    input_file.write_text("teh -> the\nteh -> tha\n")
    output_file = tmp_path / "output.md"

    sys.argv = ["multitool.py", "conflict", str(input_file), "-o", str(output_file), "-f", "md-table"]
    main()

    content = output_file.read_text()
    assert "| Typo | Corrections |" in content
    assert "| teh | tha, the |" in content

def test_md_table_near_duplicates(tmp_path):
    input_file = tmp_path / "words.txt"
    input_file.write_text("apple\napply\n")
    output_file = tmp_path / "output.md"

    sys.argv = ["multitool.py", "near_duplicates", str(input_file), "-o", str(output_file), "-f", "md-table", "--max-dist", "1"]
    main()

    content = output_file.read_text()
    assert "| Word 1 | Word 2 |" in content
    assert "| apple | apply |" in content
