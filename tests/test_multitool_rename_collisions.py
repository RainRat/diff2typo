import os
import pytest
from multitool import main
import logging
from unittest.mock import patch
import sys

def test_rename_collisions_dry_run_reporting(tmp_path, caplog):
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / "b.txt").write_text("2")
    (tmp_path / "d.txt").write_text("3")
    (tmp_path / "existing.txt").write_text("e")
    (tmp_path / "f.txt").write_text("4")

    args = [
        "multitool.py", "rename", str(tmp_path),
        "--add", "a:collision", "b:collision", "d:existing", "f:success",
        "--dry-run"
    ]

    with patch.object(sys, 'argv', args):
        with caplog.at_level(logging.WARNING):
            main()

    assert "Total renames that would be made: 1" in caplog.text
    assert "Total collisions detected (would be skipped): 3" in caplog.text

def test_rename_collisions_in_place_skipping(tmp_path, caplog):
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / "b.txt").write_text("2")
    (tmp_path / "d.txt").write_text("3")
    (tmp_path / "existing.txt").write_text("e")
    (tmp_path / "f.txt").write_text("4")

    args = [
        "multitool.py", "rename", str(tmp_path),
        "--add", "a:collision", "b:collision", "d:existing", "f:success",
        "--in-place"
    ]

    with patch.object(sys, 'argv', args):
        with caplog.at_level(logging.INFO):
            main()

    assert "Successfully made 1 rename(s)." in caplog.text
    assert "Skipped 3 collision(s)." in caplog.text

    assert (tmp_path / "success.txt").exists()
    assert (tmp_path / "a.txt").exists()
    assert (tmp_path / "b.txt").exists()
    assert (tmp_path / "d.txt").exists()
    assert (tmp_path / "collision.txt").exists() is False
