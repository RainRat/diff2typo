import pytest
import os
import json
from unittest.mock import patch
import multitool

@pytest.fixture
def sample_file(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("Hello World\nuser@example.com\nAnother line with foo@bar.org\n12345\n", encoding="utf-8")
    return str(p)

def test_regex_extraction(sample_file, capsys):
    """Test extracting emails with regex (raw, no cleaning)."""
    with patch("sys.argv", ["multitool.py", "regex", sample_file, "--pattern", r"[\w\.-]+@[\w\.-]+"]):
        multitool.main()

    captured = capsys.readouterr()
    output = captured.out.strip().split('\n')

    # filter_to_letters should NOT be applied.
    expected = ["user@example.com", "foo@bar.org"]
    assert sorted(output) == sorted(expected)

def test_regex_extraction_groups(sample_file, capsys):
    """Test extracting groups with regex."""
    # Pattern to extract just the user part of email: (user)@...
    with patch("sys.argv", ["multitool.py", "regex", sample_file, "--pattern", r"([\w\.-]+)@[\w\.-]+"]):
        multitool.main()

    captured = capsys.readouterr()
    output = captured.out.strip().split('\n')

    # user -> user, foo -> foo
    expected = ["user", "foo"]
    assert sorted(output) == sorted(expected)

def test_regex_invalid_pattern(sample_file, capsys):
    """Test handling of invalid regex pattern."""
    with patch("sys.argv", ["multitool.py", "regex", sample_file, "--pattern", r"[invalid"]):
        with pytest.raises(SystemExit) as excinfo:
            multitool.main()
        assert excinfo.value.code == 1

    captured = capsys.readouterr()
