import os
import subprocess
import pytest

def test_orphans_unreferenced_file(tmp_path):
    # Setup: Create a markdown file and an unreferenced image
    # doc.md links to other.md, and other.md links to doc.md
    doc = tmp_path / "doc.md"
    doc.write_text("# Test\n[link](other.md)", encoding="utf-8")

    other = tmp_path / "other.md"
    other.write_text("# Other\n[back](doc.md)", encoding="utf-8")

    orphan = tmp_path / "orphan.png"
    orphan.write_text("image data", encoding="utf-8")

    # Run orphans mode
    result = subprocess.run(
        ["python3", "multitool.py", "orphans", str(doc), str(other), str(orphan), "--output-format", "line"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    # The orphan file should be reported. The others should not.
    # Note: orphans mode uses _write_paired_output for non-arrow formats,
    # which in 'line' (default for paired) uses "Left -> Right Attr"
    # For orphans, it's (location [Type], Target (Reason))
    assert str(orphan) in result.stdout
    assert "Unreferenced file" in result.stdout
    assert "doc.md" not in result.stdout
    assert "other.md" not in result.stdout

def test_orphans_unused_definition(tmp_path):
    # Setup: Markdown with an unused reference definition
    doc = tmp_path / "doc.md"
    doc.write_text(
        "# Test\n"
        "[Used][used]\n\n"
        "[used]: https://example.com\n"
        "[unused]: https://orphan.com\n",
        encoding="utf-8"
    )

    # Run orphans mode
    result = subprocess.run(
        ["python3", "multitool.py", "orphans", str(doc), "--output-format", "line"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert "doc.md:5" in result.stdout
    assert "[unused]" in result.stdout
    assert "https://orphan.com" in result.stdout
    assert "Unused definition" in result.stdout
    assert "[used]" not in result.stdout

def test_orphans_arrow_format(tmp_path):
    # Setup: Create an orphan
    orphan = tmp_path / "orphan.txt"
    orphan.write_text("dead asset", encoding="utf-8")

    # Run orphans mode in arrow format
    result = subprocess.run(
        ["python3", "multitool.py", "orphans", str(orphan), "--output-format", "arrow"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert "ORPHANS ANALYSIS" in result.stdout
    assert "Unreferenced file" in result.stdout
    assert str(orphan) in result.stdout
    assert "│" in result.stdout # Check for table separator

def test_orphans_mixed_assets(tmp_path):
    # Setup: referenced and unreferenced files
    doc = tmp_path / "doc.md"
    doc.write_text("[img](img/ref.png)", encoding="utf-8")

    img_dir = tmp_path / "img"
    img_dir.mkdir()

    ref_img = img_dir / "ref.png"
    ref_img.write_text("data", encoding="utf-8")

    orphan_img = img_dir / "orphan.png"
    orphan_img.write_text("data", encoding="utf-8")

    # Run orphans mode
    result = subprocess.run(
        ["python3", "multitool.py", "orphans", str(doc), str(ref_img), str(orphan_img), "--output-format", "line"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert str(orphan_img) in result.stdout
    assert str(ref_img) not in result.stdout

def test_orphans_shortcut_links(tmp_path):
    # Setup: Markdown with a shortcut reference link [label]
    doc = tmp_path / "doc.md"
    doc.write_text(
        "# Test\n"
        "[used] and ![img-used]\n\n"
        "[used]: https://example.com\n"
        "[img-used]: img.png\n"
        "[unused]: https://orphan.com\n",
        encoding="utf-8"
    )

    # Run orphans mode
    result = subprocess.run(
        ["python3", "multitool.py", "orphans", str(doc), "--output-format", "line"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert "Unused definition" in result.stdout
    assert "[unused]" in result.stdout
    assert "[used]" not in result.stdout
    assert "![img-used]" not in result.stdout
