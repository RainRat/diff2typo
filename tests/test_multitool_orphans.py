import os
import pytest
from multitool import orphans_mode

def test_orphans_mode_basic(tmp_path):
    # Create a project structure
    docs = tmp_path / "docs"
    docs.mkdir()

    readme = docs / "README.md"
    readme.write_text("[Link](referenced.md) and ![Image](image.png)")

    referenced = docs / "referenced.md"
    referenced.write_text("I am referenced, and I link back to [Home](README.md)")

    image = docs / "image.png"
    image.write_text("binary data")

    orphan = docs / "orphan.md"
    orphan.write_text("I am an orphan")

    unused_img = docs / "unused.png"
    unused_img.write_text("more binary data")

    output_file = tmp_path / "orphans.txt"

    # Run orphans mode
    input_files = [str(readme), str(referenced), str(image), str(orphan), str(unused_img)]
    orphans_mode(input_files, str(output_file), output_format='line')

    # Check results
    orphans = output_file.read_text().splitlines()
    assert len(orphans) == 2
    assert any("orphan.md" in o for o in orphans)
    assert any("unused.png" in o for o in orphans)
    assert not any("referenced.md" in o for o in orphans)
    assert not any("image.png" in o for o in orphans)
    assert not any("README.md" in o for o in orphans)

def test_orphans_mode_arrow_format(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()

    readme = docs / "README.md"
    readme.write_text("[Link](referenced.md)")

    referenced = docs / "referenced.md"
    referenced.write_text("referenced")

    orphan = docs / "orphan.md"
    orphan.write_text("orphan")

    # Redirect output to stdout (using '-') to check formatting
    input_files = [str(readme), str(referenced), str(orphan)]

    output_file = tmp_path / "output.arrow"
    orphans_mode(input_files, str(output_file), output_format='arrow')

    content = output_file.read_text()
    assert "orphan.md" in content
    assert "ORPHAN FILES ANALYSIS" in content
    assert "Total files analyzed" in content
    assert "Total links scanned" in content

def test_orphans_mode_empty(tmp_path):
    output_file = tmp_path / "orphans.txt"
    orphans_mode([], str(output_file), output_format='line')
    assert output_file.read_text() == ""
