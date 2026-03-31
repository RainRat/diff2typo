import os
from multitool import main
import sys
from unittest.mock import patch

def test_scrub_inplace(tmp_path):
    # Create a mapping file
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\n")

    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("teh quick brown fox")

    # Run scrub with --in-place
    test_args = ["multitool.py", "scrub", str(mapping_file), str(test_file), "--in-place"]
    with patch.object(sys, 'argv', test_args):
        main()

    # Verify file was updated
    assert test_file.read_text() == "the quick brown fox\n"
    assert not os.path.exists(str(test_file) + ".bak")

def test_scrub_inplace_backup(tmp_path):
    # Create a mapping file
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\n")

    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("teh quick brown fox")

    # Run scrub with --in-place .bak
    test_args = ["multitool.py", "scrub", str(mapping_file), str(test_file), "--in-place", ".bak"]
    with patch.object(sys, 'argv', test_args):
        main()

    # Verify file was updated and backup created
    assert test_file.read_text() == "the quick brown fox\n"
    backup_file = tmp_path / "test.txt.bak"
    assert backup_file.exists()
    assert backup_file.read_text() == "teh quick brown fox"

def test_scrub_inplace_dry_run(tmp_path, caplog):
    # Create a mapping file
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\n")

    # Create a test file
    test_file = tmp_path / "test.txt"
    original_content = "teh quick brown fox"
    test_file.write_text(original_content)

    # Run scrub with --in-place and --dry-run
    test_args = ["multitool.py", "scrub", str(mapping_file), str(test_file), "--in-place", "--dry-run"]
    with patch.object(sys, 'argv', test_args):
        main()

    # Verify file was NOT updated
    assert test_file.read_text() == original_content
    assert "[Dry Run] Would make 1 replacement(s) in" in caplog.text

def test_scrub_dry_run_accumulated(tmp_path, caplog):
    # Create a mapping file
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\n")

    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("teh quick brown fox")

    # Run scrub with --dry-run (no in-place)
    test_args = ["multitool.py", "scrub", str(mapping_file), str(test_file), "--dry-run"]
    with patch.object(sys, 'argv', test_args):
        main()

    # Verify log contains total replacements
    assert "[Dry Run] Total replacements that would be made: 1" in caplog.text

def test_scrub_multiple_inplace(tmp_path):
    # Create a mapping file
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\n")

    # Create test files
    file1 = tmp_path / "file1.txt"
    file1.write_text("teh one")
    file2 = tmp_path / "file2.txt"
    file2.write_text("teh two")

    # Run scrub with --in-place on multiple files
    test_args = ["multitool.py", "scrub", str(mapping_file), str(file1), str(file2), "--in-place"]
    with patch.object(sys, 'argv', test_args):
        main()

    # Verify both files updated
    assert file1.read_text() == "the one\n"
    assert file2.read_text() == "the two\n"
