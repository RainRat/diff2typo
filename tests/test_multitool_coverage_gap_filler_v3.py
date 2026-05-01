import sys
import io
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable=None, *_, **__: iterable if iterable is not None else MagicMock())

def test_read_file_lines_robust_directory(tmp_path, caplog):
    # Covers lines 269-270
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()

    with caplog.at_level(logging.WARNING):
        lines = multitool._read_file_lines_robust(str(dir_path))

    assert lines == []
    assert f"Input path '{dir_path}' is a directory. Skipping." in caplog.text

def test_main_input_expansion_excluded_folder(tmp_path, capsys):
    # Covers lines 5868-5869
    # Pass an excluded folder DIRECTLY as input match
    exclude_dir = tmp_path / "node_modules"
    exclude_dir.mkdir()
    (exclude_dir / "skipped.txt").write_text("skipped_content")

    with patch("sys.argv", ["multitool.py", "words", str(exclude_dir), "--quiet"]):
        with patch("sys.stdin", io.StringIO("")):
            multitool._STDIN_CACHE = None
            multitool.main()

    captured = capsys.readouterr()
    assert "skippedcontent" not in captured.out

def test_main_input_expansion_prune_dirs(tmp_path, capsys):
    # Covers line 5873: dirs[:] = [d for d in dirs if d not in exclude]
    base_dir = tmp_path / "base_prune"
    base_dir.mkdir()

    valid_dir = base_dir / "valid_dir"
    valid_dir.mkdir()
    (valid_dir / "file.txt").write_text("found_me")

    exclude_dir = base_dir / "node_modules"
    exclude_dir.mkdir()
    (exclude_dir / "skipped.txt").write_text("skipped_me")

    with patch("sys.argv", ["multitool.py", "words", str(base_dir), "--quiet"]):
        multitool._STDIN_CACHE = None
        multitool.main()

    captured = capsys.readouterr()
    assert "foundme" in captured.out
    assert "skippedme" not in captured.out

def test_main_rename_mode_include_directories(tmp_path):
    # Covers lines 5877-5878: expanded_paths.append(os.path.join(root, d))
    base_dir = tmp_path / "rename_test"
    base_dir.mkdir()
    sub_dir = base_dir / "sub_dir"
    sub_dir.mkdir()

    # rename mode needs a mapping
    mapping = tmp_path / "map.txt"
    mapping.write_text("sub_dir -> new_dir")

    # We want to verify that sub_dir itself is added to input_paths
    with patch("sys.argv", ["multitool.py", "rename", str(base_dir), "--mapping", str(mapping), "--dry-run", "--quiet"]):
        with patch("multitool.rename_mode") as mock_rename_mode:
            multitool._STDIN_CACHE = None
            multitool.main()

            # The input_files passed to rename_mode should include the directory
            called_input_files = mock_rename_mode.call_args[1]['input_files']
            assert any(str(sub_dir) == str(p) for p in called_input_files)

def test_main_delimiter_empty_string(tmp_path, capsys):
    # Covers line 5947: delimiter = None
    input_file = tmp_path / "input.csv"
    input_file.write_text("a,b\nc,d")

    # Use words mode but disable cleaning so we see the comma
    with patch("sys.argv", ["multitool.py", "words", str(input_file), "--delimiter", "", "--raw", "--quiet"]):
        multitool._STDIN_CACHE = None
        multitool.main()

    captured = capsys.readouterr()
    # If delimiter is None, it doesn't split by comma, so "a,b" is one word
    assert "a,b" in captured.out
