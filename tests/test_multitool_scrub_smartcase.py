from multitool import main
import sys
from unittest.mock import patch

def test_scrub_smart_case(tmp_path):
    # Create a mapping file (lowercase)
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\n")

    # Create a test file with various casings
    test_file = tmp_path / "test.txt"
    test_file.write_text("teh Teh TEH")

    # Run scrub with --smart-case and --in-place
    test_args = ["multitool.py", "scrub", str(test_file), str(mapping_file), "--smart-case", "--in-place"]
    with patch.object(sys, 'argv', test_args):
        main()

    # Verify casing preservation
    # Note: scrub mode adds a newline at the end when writing to file
    assert test_file.read_text() == "the The THE\n"

def test_scrub_smart_case_subwords(tmp_path):
    # Create a mapping file
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\n")

    # Create a test file with CamelCase/PascalCase subwords
    test_file = tmp_path / "test.txt"
    test_file.write_text("MyTehWord PascalTeh")

    # Run scrub with --smart-case and --in-place
    test_args = ["multitool.py", "scrub", str(test_file), str(mapping_file), "--smart-case", "--in-place"]
    with patch.object(sys, 'argv', test_args):
        main()

    # Verify casing preservation in subwords
    assert test_file.read_text() == "MyTheWord PascalThe\n"

def test_scrub_no_smart_case(tmp_path):
    # Create a mapping file
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\n")

    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Teh TEH")

    # Run scrub WITHOUT --smart-case but with --in-place
    test_args = ["multitool.py", "scrub", str(test_file), str(mapping_file), "--in-place"]
    with patch.object(sys, 'argv', test_args):
        main()

    # Verify replacements are lowercase (default behavior)
    assert test_file.read_text() == "the the\n"

def test_scrub_smart_case_raw_mode(tmp_path):
    # In raw mode, matching is case-sensitive.
    # Create a mapping file
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("Teh,The\n")

    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Teh TEH")

    # Run scrub with --raw, --smart-case and --in-place
    test_args = ["multitool.py", "scrub", str(test_file), str(mapping_file), "--raw", "--smart-case", "--in-place"]
    with patch.object(sys, 'argv', test_args):
        main()

    # Only "Teh" matches in raw mode. "TEH" does not.
    assert test_file.read_text() == "The TEH\n"

def test_scrub_smart_case_empty_replacement(tmp_path):
    # Create a mapping file with an empty replacement
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,\n")

    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Teh TEH")

    # Run scrub with --smart-case and --in-place
    test_args = ["multitool.py", "scrub", str(test_file), str(mapping_file), "--smart-case", "--in-place"]
    with patch.object(sys, 'argv', test_args):
        main()

    # Verify no crash and empty strings used (with newlines from file writing)
    assert test_file.read_text() == " \n"
