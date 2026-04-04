import sys
import os
import pytest
from unittest.mock import patch
from io import StringIO
import multitool

def test_verify_mode_basic(tmp_path):
    # Setup mapping file
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("teh -> the\nrecieve -> receive\n", encoding="utf-8")

    # Setup input file
    input_file = tmp_path / "input.txt"
    input_file.write_text("This is teh house.\n", encoding="utf-8")

    # Call verify mode without prune
    output_file = tmp_path / "output.txt"

    with patch("sys.argv", ["multitool.py", "verify", str(mapping_file), str(input_file), "-o", str(output_file)]):
        multitool.main()

    content = output_file.read_text(encoding="utf-8")
    assert "teh -> the [FOUND]" in content
    assert "recieve -> receive [MISSING]" in content

def test_verify_mode_prune(tmp_path):
    # Setup mapping file
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("teh -> the\nrecieve -> receive\n", encoding="utf-8")

    # Setup input file
    input_file = tmp_path / "input.txt"
    input_file.write_text("This is teh house.\n", encoding="utf-8")

    # Call verify mode with prune
    output_file = tmp_path / "output.txt"

    with patch("sys.argv", ["multitool.py", "verify", str(mapping_file), str(input_file), "--prune", "-o", str(output_file)]):
        multitool.main()

    content = output_file.read_text(encoding="utf-8")
    assert "teh -> the" in content
    assert "recieve" not in content

def test_verify_mode_smart(tmp_path):
    # Setup mapping file
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("teh -> the\n", encoding="utf-8")

    # Setup input file with compound word
    input_file = tmp_path / "input.txt"
    input_file.write_text("This is tehHouse.\n", encoding="utf-8")

    # Call verify mode without smart
    output_file_no_smart = tmp_path / "output_no_smart.txt"
    with patch("sys.argv", ["multitool.py", "verify", str(mapping_file), str(input_file), "-o", str(output_file_no_smart)]):
        multitool.main()
    assert "teh -> the [MISSING]" in output_file_no_smart.read_text(encoding="utf-8")

    # Call verify mode with smart
    output_file_smart = tmp_path / "output_smart.txt"
    with patch("sys.argv", ["multitool.py", "verify", str(mapping_file), str(input_file), "--smart", "-o", str(output_file_smart)]):
        multitool.main()
    assert "teh -> the [FOUND]" in output_file_smart.read_text(encoding="utf-8")

def test_verify_mode_adhoc(tmp_path):
    # Setup input file
    input_file = tmp_path / "input.txt"
    input_file.write_text("Found teh typo.\n", encoding="utf-8")

    # Call verify mode with ad-hoc pairs
    output_file = tmp_path / "output.txt"

    with patch("sys.argv", ["multitool.py", "verify", str(input_file), "--add", "teh:the", "missing:word", "-o", str(output_file)]):
        multitool.main()

    content = output_file.read_text(encoding="utf-8")
    assert "teh -> the [FOUND]" in content
    assert "missing -> word [MISSING]" in content or "missing ->  [MISSING]" in content # ad-hoc "missing" might result in empty correction

def test_verify_mode_no_mapping_fails():
    with patch("sys.argv", ["multitool.py", "verify"]), pytest.raises(SystemExit) as e:
        multitool.main()
    assert e.value.code == 1
