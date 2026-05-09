
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable=None, *_, **__: iterable if iterable is not None else MagicMock())

def test_count_mode_audit_by_file(tmp_path):
    """
    Test count_mode with mapping and by_file=True.
    Also test length filtering in audit mode.
    """
    file1 = tmp_path / "file1.txt"
    file1.write_text("apple apple banana x")
    file2 = tmp_path / "file2.txt"
    file2.write_text("apple cherry")

    output_file = tmp_path / "output.txt"

    # Ad-hoc mapping: apple:fruit1, banana:fruit2, cherry:fruit3, x:toolong
    multitool.count_mode(
        input_files=[str(file1), str(file2)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=False,
        ad_hoc=["apple:fruit1", "banana:fruit2", "cherry:fruit3", "x:ignore"],
        by_file=True,
        output_format='line'
    )

    content = output_file.read_text().splitlines()
    # apple is in 2 files, banana in 1 file, cherry in 1 file. 'x' is filtered by length.
    assert "apple -> fruit1: 2" in content
    assert "banana -> fruit2: 1" in content
    assert "cherry -> fruit3: 1" in content
    assert "x -> ignore" not in output_file.read_text()

def test_count_mode_pairs_by_file(tmp_path):
    """
    Test count_mode with --pairs and --by-file.
    """
    file1 = tmp_path / "file1.csv"
    file1.write_text("apple,fruit\nbanana,fruit")
    file2 = tmp_path / "file2.csv"
    file2.write_text("apple,fruit\ncherry,fruit")

    output_file = tmp_path / "output.txt"

    multitool.count_mode(
        input_files=[str(file1), str(file2)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        pairs=True,
        by_file=True,
        output_format='line'
    )

    content = output_file.read_text().splitlines()
    assert "apple -> fruit: 2" in content
    assert "banana -> fruit: 1" in content
    assert "cherry -> fruit: 1" in content

def test_resolve_mode_filtering_and_raw(tmp_path):
    """
    Test resolve_mode with length filtering and raw=False (default) vs True.
    """
    input_file = tmp_path / "pairs.csv"
    # apple -> fruit (ok)
    # a -> fruit (too short typo)
    # banana -> f (too short correction)
    # ! -> ? (becomes empty after cleaning)
    input_file.write_text("apple,fruit\na,fruit\nbanana,f\n!,?")

    output_file = tmp_path / "output.txt"

    # 1. Standard (cleaning and filtering)
    multitool.resolve_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=False,
        clean_items=True,
        output_format='line'
    )
    assert output_file.read_text().strip() == "apple -> fruit"

    # 2. Raw (no cleaning, but still filtering)
    multitool.resolve_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        clean_items=False,
        output_format='line'
    )
    content = output_file.read_text().splitlines()
    assert "apple -> fruit" in content
    assert "a -> fruit" in content
    assert "banana -> f" in content
    assert "! -> ?" in content

def test_rename_mode_directory_expansion(tmp_path, monkeypatch):
    """
    Test directory expansion logic specifically for 'rename' mode in main().
    It should use bottom-up traversal and include directories themselves.
    """
    base = tmp_path / "base"
    sub = base / "sub"
    sub.mkdir(parents=True)
    # Add an excluded directory to cover the 'if d not in exclude' branch
    excluded = sub / ".git"
    excluded.mkdir()

    file1 = sub / "file1.txt"
    file1.write_text("content")

    with patch("sys.argv", ["multitool.py", "rename", str(base), "--mapping", "m.csv"]), \
         patch("multitool.rename_mode") as mock_rename:
        try:
            multitool.main()
        except SystemExit:
            pass

        _, kwargs = mock_rename.call_args
        input_paths = kwargs["input_files"]

        assert str(file1) in input_paths
        assert str(sub) in input_paths
        assert str(base) in input_paths
        assert str(excluded) not in input_paths

        idx_file = input_paths.index(str(file1))
        idx_sub = input_paths.index(str(sub))
        idx_base = input_paths.index(str(base))

        assert idx_file < idx_sub
        assert idx_sub < idx_base

def test_main_directory_expansion_exclude(tmp_path):
    """
    Test directory expansion skipping excluded folders like .git
    """
    base = tmp_path / "project"
    base.mkdir()
    git_dir = base / ".git"
    git_dir.mkdir()
    git_file = git_dir / "config"
    git_file.write_text("secret")

    src_dir = base / "src"
    src_dir.mkdir()
    src_file = src_dir / "main.py"
    src_file.write_text("print(1)")

    with patch("sys.argv", ["multitool.py", "words", str(base)]), \
         patch("multitool.words_mode") as mock_words:
        try:
            multitool.main()
        except SystemExit:
            pass

        _, kwargs = mock_words.call_args
        input_paths = kwargs["input_files"]

        assert str(src_file) in input_paths
        assert str(git_file) not in input_paths
        assert str(git_dir) not in input_paths
