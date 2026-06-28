import os
import json
import pytest
from multitool import orphans_mode

def test_orphans_mode_basic(tmp_path):
    # Create a project structure
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()

    index_md = doc_dir / "index.md"
    index_md.write_text("See [Guide](guide.md) and ![Logo](img/logo.png)")

    guide_md = doc_dir / "guide.md"
    guide_md.write_text("Welcome to the guide. [Back home](index.md)")

    orphan_md = doc_dir / "orphan.md"
    orphan_md.write_text("I am not linked by anyone.")

    img_dir = doc_dir / "img"
    img_dir.mkdir()
    logo_png = img_dir / "logo.png"
    logo_png.write_text("fake png content")

    unused_png = img_dir / "unused.png"
    unused_png.write_text("unused fake png")

    output_file = tmp_path / "output.json"

    # We pass all files as input_files
    input_files = [
        str(index_md),
        str(guide_md),
        str(orphan_md),
        str(logo_png),
        str(unused_png)
    ]

    orphans_mode(
        input_files=input_files,
        output_file=str(output_file),
        output_format='json',
        quiet=True
    )

    with open(output_file, "r") as f:
        orphans = json.load(f)

    # Expected orphans: orphan.md and unused.png
    assert len(orphans) == 2
    assert any(o.endswith("orphan.md") for o in orphans)
    assert any(o.endswith("unused.png") for o in orphans)
    assert not any(o.endswith("index.md") for o in orphans)
    assert not any(o.endswith("guide.md") for o in orphans)
    assert not any(o.endswith("logo.png") for o in orphans)

def test_orphans_mode_no_orphans(tmp_path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()

    index_md = doc_dir / "index.md"
    index_md.write_text("[About](about.md)")

    about_md = doc_dir / "about.md"
    about_md.write_text("[Back](index.md)")

    output_file = tmp_path / "output.json"

    input_files = [str(index_md), str(about_md)]

    orphans_mode(
        input_files=input_files,
        output_file=str(output_file),
        output_format='json',
        quiet=True
    )

    with open(output_file, "r") as f:
        orphans = json.load(f)

    assert orphans == []

def test_orphans_mode_arrow(tmp_path, capsys):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()

    index_md = doc_dir / "index.md"
    index_md.write_text("[Link](linked.md)")

    linked_md = doc_dir / "linked.md"
    linked_md.write_text("Linked")

    orphan_md = doc_dir / "orphan.md"
    orphan_md.write_text("Orphan")

    input_files = [str(index_md), str(linked_md), str(orphan_md)]

    # Run in arrow mode to stdout
    orphans_mode(
        input_files=input_files,
        output_file='-',
        output_format='arrow',
        quiet=True
    )

    captured = capsys.readouterr()
    assert "ORPHANED FILES" in captured.out
    assert "orphan.md" in captured.out
    assert "index.md" in captured.out
    assert "linked.md" not in captured.out
    assert "ORPHANS ANALYSIS" in captured.out
