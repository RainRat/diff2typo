import os
import subprocess
import pytest
from pathlib import Path

def test_scrub_diff(tmp_path):
    # Get absolute path to multitool.py
    root_dir = Path(__file__).resolve().parents[1]
    multitool_path = root_dir / "multitool.py"

    input_file = tmp_path / "test.txt"
    input_file.write_text("This is teh test file with a typo.\nAnother line without change.")

    # Run scrub mode with --diff
    result = subprocess.run(
        ["python3", str(multitool_path), "scrub", str(input_file), "--add", "teh:the", "--diff"],
        capture_output=True,
        text=True,
        check=True
    )

    output = result.stdout
    assert "--- a/" + str(input_file) in output
    assert "+++ b/" + str(input_file) in output
    assert "-This is teh test file with a typo." in output
    assert "+This is the test file with a typo." in output
    assert " Another line without change." in output

def test_standardize_diff(tmp_path):
    # Get absolute path to multitool.py
    root_dir = Path(__file__).resolve().parents[1]
    multitool_path = root_dir / "multitool.py"

    file1 = tmp_path / "file1.txt"
    file1.write_text("Database database database.")
    file2 = tmp_path / "file2.txt"
    file2.write_text("DATABASE database database.")

    # Run standardize mode with --diff
    # database is most frequent (4 times), Database and DATABASE are once each.
    result = subprocess.run(
        ["python3", str(multitool_path), "standardize", str(tmp_path), "--diff"],
        capture_output=True,
        text=True,
        check=True
    )

    output = result.stdout
    assert "--- a/" + str(file1) in output
    assert "-Database database database." in output
    assert "+database database database." in output

    assert "--- a/" + str(file2) in output
    assert "-DATABASE database database." in output
    assert "+database database database." in output

def test_diff_patch_compatibility(tmp_path):
    # Save original working directory
    old_cwd = os.getcwd()
    try:
        # Use relative paths for better patch compatibility in tests
        os.chdir(tmp_path)
        input_file_name = "target.txt"
        input_file = Path(input_file_name)
        original_content = "Line 1: teh typo\nLine 2: stable"
        input_file.write_text(original_content)

        # Generate diff using relative path
        root_dir = Path(__file__).resolve().parents[1]
        multitool_path = root_dir / "multitool.py"

        diff_result = subprocess.run(
            ["python3", str(multitool_path), "scrub", input_file_name, "--add", "teh:the", "--diff"],
            capture_output=True,
            text=True,
            check=True
        )

        patch_content = diff_result.stdout

        # Verify with patch --dry-run
        patch_check = subprocess.run(
            ["patch", "--dry-run", "-p1"],
            input=patch_content,
            capture_output=True,
            text=True
        )

        assert patch_check.returncode == 0
        assert "checking file target.txt" in patch_check.stdout.lower()
    finally:
        os.chdir(old_cwd)
