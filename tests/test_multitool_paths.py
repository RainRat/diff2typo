import pytest
import os
from multitool import paths_mode

def test_paths_mode_basic(tmp_path):
    output_file = tmp_path / "output.txt"
    input_files = ["src/main.py", "docs/readme.md"]

    paths_mode(
        input_files=input_files,
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        output_format='line',
        quiet=True,
        clean_items=False
    )

    content = output_file.read_text().splitlines()
    assert "src/main.py" in content
    assert "docs/readme.md" in content

def test_paths_mode_components(tmp_path):
    output_file = tmp_path / "output.txt"
    input_files = ["src/module/sub.py"]

    # Test basename
    paths_mode(
        input_files=input_files,
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        basename=True,
        quiet=True,
        clean_items=False
    )
    assert output_file.read_text().strip() == "sub.py"

    # Test dirname
    paths_mode(
        input_files=input_files,
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        dirname=True,
        quiet=True,
        clean_items=False
    )
    assert output_file.read_text().strip() == "src/module"

    # Test extension
    paths_mode(
        input_files=input_files,
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        extension=True,
        quiet=True,
        clean_items=False
    )
    assert output_file.read_text().strip() == ".py"

def test_paths_mode_smart_split(tmp_path):
    output_file = tmp_path / "output.txt"
    input_files = ["MyCamelCaseFile.txt"]

    paths_mode(
        input_files=input_files,
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=True,
        basename=True,
        smart=True,
        quiet=True,
        clean_items=False
    )

    content = sorted(output_file.read_text().splitlines())
    assert "My" in content
    assert "Camel" in content
    assert "Case" in content
    assert "File" in content

def test_paths_mode_filtering(tmp_path):
    output_file = tmp_path / "output.txt"
    input_files = ["a.py", "abcde.py"]

    # a.py length is 4
    # abcde.py length is 8

    paths_mode(
        input_files=input_files,
        output_file=str(output_file),
        min_length=5,
        max_length=1000,
        process_output=False,
        basename=True,
        quiet=True,
        clean_items=False
    )

    content = output_file.read_text().splitlines()
    assert "abcde.py" in content
    assert "a.py" not in content

def test_paths_mode_cleaning(tmp_path):
    output_file = tmp_path / "output.txt"
    input_files = ["My-File_Name.txt"]

    paths_mode(
        input_files=input_files,
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        basename=True,
        quiet=True,
        clean_items=True
    )

    # filter_to_letters removes everything except a-z and lowercases
    # "My-File_Name.txt" -> "myfilenametxt"
    assert output_file.read_text().strip() == "myfilenametxt"

def test_paths_mode_directory_inclusion(tmp_path):
    # Create a structure:
    # tmp_path/
    #   dir1/
    #     file1.txt
    #     dir2/
    #       file2.txt

    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    (dir1 / "file1.txt").touch()
    dir2 = dir1 / "dir2"
    dir2.mkdir()
    (dir2 / "file2.txt").touch()

    import subprocess
    import sys

    # Run via subprocess to test main() and directory expansion
    result = subprocess.run(
        [sys.executable, "multitool.py", "paths", str(dir1), "--basename", "--raw", "-P"],
        capture_output=True,
        text=True
    )

    output = result.stdout.splitlines()
    # Expecting: dir1, dir2, file1.txt, file2.txt (sorted due to -P)
    assert "dir1" in output
    assert "dir2" in output
    assert "file1.txt" in output
    assert "file2.txt" in output
