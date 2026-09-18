import json
import pytest
from multitool import markdown_mode, _extract_markdown_items_detailed

def test_extract_markdown_items_detailed(tmp_path):
    md_file = tmp_path / "test_extract.md"
    md_file.write_text("- teh -> the\n* apple: fruit\n+ standalone item\n- invalid: test\n- ")

    items = list(_extract_markdown_items_detailed(str(md_file)))
    assert items == [
        ("teh", "the"),
        ("apple", "fruit"),
        ("standalone item", ""),
        ("invalid", "test"),
    ]

def test_markdown_mode_pairs_basic(tmp_path):
    md_file = tmp_path / "notes.md"
    md_file.write_text("- teh -> the\n* adn: and\n+ item without pair")

    out_arrow = tmp_path / "out.arrow"
    markdown_mode(
        input_files=[str(md_file)],
        output_file=str(out_arrow),
        min_length=1,
        max_length=100,
        process_output=False,
        pairs=True,
        output_format='arrow',
        clean_items=False,
    )
    content = out_arrow.read_text()
    assert "teh" in content
    assert "the" in content
    assert "adn" in content
    assert "and" in content
    assert "item without pair" in content

def test_markdown_mode_pairs_csv(tmp_path):
    md_file = tmp_path / "notes.md"
    md_file.write_text("- typo1 -> fix1\n- typo2: fix2")

    out_csv = tmp_path / "out.csv"
    markdown_mode(
        input_files=[str(md_file)],
        output_file=str(out_csv),
        min_length=1,
        max_length=100,
        process_output=False,
        pairs=True,
        output_format='csv',
        clean_items=False,
    )
    lines = out_csv.read_text().splitlines()
    assert lines == ["typo1,fix1", "typo2,fix2"]

def test_markdown_mode_pairs_json(tmp_path):
    md_file = tmp_path / "notes.md"
    md_file.write_text("- left -> right")

    out_json = tmp_path / "out.json"
    markdown_mode(
        input_files=[str(md_file)],
        output_file=str(out_json),
        min_length=1,
        max_length=100,
        process_output=False,
        pairs=True,
        output_format='json',
        clean_items=False,
    )
    data = json.loads(out_json.read_text())
    assert data == {"left": "right"}

def test_markdown_mode_pairs_clean_and_filter(tmp_path):
    md_file = tmp_path / "notes.md"
    md_file.write_text("- AB -> CD\n- longwordhere -> fix")

    out_txt = tmp_path / "out.txt"
    markdown_mode(
        input_files=[str(md_file)],
        output_file=str(out_txt),
        min_length=3,
        max_length=10,
        process_output=False,
        pairs=True,
        output_format='arrow',
        clean_items=True,
    )
    lines = out_txt.read_text().splitlines()
    # AB -> ab (len 2, filtered by min_length=3)
    # longwordhere -> longwordhere (len 12, filtered by max_length=10)
    assert lines == []


def test_markdown_mode_pairs_process_output_sorting_and_dedup(tmp_path):
    md_file = tmp_path / "dup.md"
    md_file.write_text("- zebra -> zoo\n- apple -> fruit\n- zebra -> zoo")

    out_csv = tmp_path / "out.csv"
    markdown_mode(
        input_files=[str(md_file)],
        output_file=str(out_csv),
        min_length=1,
        max_length=100,
        process_output=True,
        pairs=True,
        output_format='csv',
        clean_items=True,
    )
    lines = out_csv.read_text().splitlines()
    assert lines == ["apple,fruit", "zebra,zoo"]
