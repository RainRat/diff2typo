import io
import os
import sys
import json
from unittest.mock import patch
import pytest
import gentypos

def test_gentypos_non_interactive_reads_stdin_instead_of_default_config(capsys, monkeypatch):
    # Mock os.path.exists to return True for gentypos.yaml
    original_exists = os.path.exists
    def mock_exists(path):
        if path == "gentypos.yaml":
            return True
        return original_exists(path)
    monkeypatch.setattr(os.path, "exists", mock_exists)

    # Mock parse_yaml_config to return the default config dictionary
    default_config_data = {
        "input_file": "wordlist_small.txt",
        "dictionary_file": None,
        "output_format": "list",
    }
    monkeypatch.setattr(gentypos, "parse_yaml_config", lambda path: default_config_data)

    # Mock stdin content
    mock_stdin = io.StringIO("hello\n")

    # Mock argv with no config file argument (uses default gentypos.yaml)
    test_args = [
        "gentypos.py",
        "-f", "list",
        "--no-filter",
        "-m", "3",
    ]

    with patch("sys.stdin.isatty", return_value=False), \
         patch("sys.stdin", mock_stdin), \
         patch("sys.argv", test_args):
        gentypos.main()

    captured = capsys.readouterr()
    stdout_lines = captured.out.splitlines()
    assert len(stdout_lines) > 0
    # verify that typos like "ehllo" (transposition) or "gello" (adjacent replacement of 'h') are generated
    assert "ehllo" in stdout_lines or "gello" in stdout_lines


def test_gentypos_interactive_missing_input_exits_gracefully(caplog, monkeypatch):
    # Ensure config loading returns empty or references a non-existent file
    monkeypatch.setattr(gentypos, "parse_yaml_config", lambda path: {})
    # Mock exists to return False for configuration files or input files
    monkeypatch.setattr(os.path, "exists", lambda path: False)

    test_args = ["gentypos.py"]

    with patch("sys.stdin.isatty", return_value=True), \
         patch("sys.argv", test_args), \
         pytest.raises(SystemExit) as excinfo:
        gentypos.main()

    assert excinfo.value.code == 1
    # Check that our user-friendly error is in logs
    assert any("No input words or valid input file provided" in record.message for record in caplog.records)
