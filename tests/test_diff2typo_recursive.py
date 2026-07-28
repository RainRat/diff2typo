import os
import shutil
import pytest
import diff2typo

def test_recursive_scan_finds_files(tmp_path):
    # Set up folders and files
    dummy_dir = tmp_path / "dummy_diffs"
    dummy_dir.mkdir()

    sub_dir = dummy_dir / "sub"
    sub_dir.mkdir()

    file1 = dummy_dir / "patch1.diff"
    file1.write_bytes(b"hello -> hallo")

    file2 = sub_dir / "patch2.txt"
    file2.write_bytes(b"wrod -> word")

    # We also write a non-supported extension to ensure it is ignored
    file3 = sub_dir / "patch3.ignored"
    file3.write_bytes(b"foo -> bar")

    # Run the read_diff_sources function
    result = diff2typo._read_diff_sources([str(dummy_dir)])

    # Assertions
    assert "hello -> hallo" in result
    assert "wrod -> word" in result
    assert "foo -> bar" not in result

def test_recursive_scan_ignores_folders(tmp_path):
    dummy_dir = tmp_path / "dummy_diffs"
    dummy_dir.mkdir()

    # A standard ignored folder
    git_dir = dummy_dir / ".git"
    git_dir.mkdir()

    file_ignored = git_dir / "patch_ignored.txt"
    file_ignored.write_bytes(b"git_typo -> git_correct")

    file_kept = dummy_dir / "patch_kept.diff"
    file_kept.write_bytes(b"kept_typo -> kept_correct")

    # Run the read_diff_sources function
    result = diff2typo._read_diff_sources([str(dummy_dir)])

    # Assertions
    assert "kept_typo -> kept_correct" in result
    assert "git_typo -> git_correct" not in result
