import pytest
from multitool import headings_mode, toc_mode, links_mode

def test_headings_mode_sorting_and_uniqueness(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Beta\n# Alpha\n# Beta")
    output_file = tmp_path / "output.txt"

    headings_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        clean_items=False,
        quiet=True
    )

    content = output_file.read_text().splitlines()
    assert content == ["Alpha", "Beta"]

def test_toc_mode_sorting_and_uniqueness(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Beta\n# Alpha")
    output_file = tmp_path / "output.txt"

    toc_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        no_links=True,
        clean_items=False,
        quiet=True
    )

    content = output_file.read_text().splitlines()
    assert content == ["- Alpha", "- Beta"]

def test_links_mode_pairs_filtering_and_sorting(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("[Beta](url2)\n[Alpha](url1)\n[Beta](url2)\n[X](url3)")
    output_file = tmp_path / "output.txt"

    links_mode(
        input_files=[str(md_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=True,
        pairs=True,
        clean_items=False,
        quiet=True
    )

    content = output_file.read_text().splitlines()
    assert content == ["Alpha -> url1", "Beta -> url2"]
