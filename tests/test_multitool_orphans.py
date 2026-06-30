import os
import subprocess
import pytest
import shutil

def test_multitool_orphans_basic(tmp_path):
    # Setup test directory
    test_dir = tmp_path / "test_orphans"
    test_dir.mkdir()

    index_md = test_dir / "index.md"
    index_md.write_text("# Index\n[link](linked.md)")

    linked_md = test_dir / "linked.md"
    linked_md.write_text("# Linked\n[back](index.md)")

    orphan_md = test_dir / "orphan.md"
    orphan_md.write_text("# Orphan")

    asset_png = test_dir / "asset.png"
    asset_png.write_text("DUMMY PNG")

    # Run orphans mode
    result = subprocess.run(
        ["python3", "multitool.py", "orphans", str(test_dir), "--format", "line"],
        capture_output=True,
        text=True
    )

    output = result.stdout.strip().split('\n')

    found_orphans = set()
    for line in output:
        if line.strip() and not line.startswith("[Orphans Mode]") and not line.startswith("Loaded"):
            found_orphans.add(os.path.abspath(line.strip()))

    expected_abs = {os.path.abspath(str(test_dir / "orphan.md")), os.path.abspath(str(test_dir / "asset.png"))}

    assert found_orphans == expected_abs

def test_multitool_orphans_chain(tmp_path):
    test_dir = tmp_path / "test_chain"
    test_dir.mkdir()

    a_md = test_dir / "a.md"
    a_md.write_text("[b](b.md)")

    b_md = test_dir / "b.md"
    b_md.write_text("[c](c.md)")

    c_md = test_dir / "c.md"
    c_md.write_text("# C")

    result = subprocess.run(
        ["python3", "multitool.py", "orphans", str(test_dir), "--format", "line"],
        capture_output=True,
        text=True
    )

    output = [l for l in result.stdout.strip().split('\n') if l.strip() and not l.startswith("[Orphans Mode]") and not l.startswith("Loaded")]
    # a.md is an orphan because nothing links TO it.
    found_abs = {os.path.abspath(l.strip()) for l in output}
    assert found_abs == {os.path.abspath(str(a_md))}
