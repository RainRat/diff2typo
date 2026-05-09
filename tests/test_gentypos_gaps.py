import json
import csv
import sys
import logging
import importlib
from pathlib import Path
from unittest.mock import patch

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

def test_color_initialization_enabled(monkeypatch):
    """Test branch where colors are NOT disabled (Line 50)."""
    # Mock sys.stdout.isatty to return True
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    # Ensure NO_COLOR is not in environment
    monkeypatch.delenv("NO_COLOR", raising=False)

    # Reload the module to trigger the top-level color initialization
    importlib.reload(gentypos)

    # Check that colors are not empty
    assert gentypos.RED != ""
    assert gentypos.RESET != ""

def test_minimal_formatter_no_color_for_level(monkeypatch):
    """Test branch in MinimalFormatter where color is None (Line 71)."""
    formatter = gentypos.MinimalFormatter()
    # Mock sys.stderr.isatty to return True
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)

    # Create a record with a level NOT in LEVEL_COLORS, e.g., DEBUG
    # DEBUG = 10
    record = logging.LogRecord(
        name="test",
        level=logging.DEBUG,
        pathname="test.py",
        lineno=1,
        msg="debug message",
        args=(),
        exc_info=None
    )

    formatted = formatter.format(record)
    # levelname should be "DEBUG" without color codes
    assert "DEBUG: debug message" == formatted
    assert gentypos.RED not in formatted

def test_load_substitutions_json_list(tmp_path):
    """Test branch where JSON data is a list instead of a dict (Line 184)."""
    path = tmp_path / "list.json"
    path.write_text(json.dumps(["a", "b"]))
    result = gentypos._load_substitutions_file(str(path))
    assert result == {}

def test_load_substitutions_json_missing_keys(tmp_path):
    """Test branch where JSON replacements item is missing keys (Line 181)."""
    path = tmp_path / "missing.json"
    data = {
        "replacements": [
            {"correct": "a"},  # missing 'typo'
            {"typo": "e"},     # missing 'correct'
            {"correct": "i", "typo": "o"} # valid
        ]
    }
    path.write_text(json.dumps(data))
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"i": ["o"]}

def test_load_substitutions_csv_missing_fields(tmp_path):
    """Test branch where CSV row is missing correct_char or typo_char (Line 198)."""
    path = tmp_path / "missing.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["correct_char", "typo_char"])
        writer.writeheader()
        writer.writerow({"correct_char": "a", "typo_char": ""})
        writer.writerow({"correct_char": "", "typo_char": "e"})
        writer.writerow({"correct_char": "i", "typo_char": "o"})

    result = gentypos._load_substitutions_file(str(path))
    assert result == {"i": ["o"]}

def test_load_substitutions_yaml_list(tmp_path):
    """Test branch where YAML data is a list instead of a dict (Line 223)."""
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b")
    result = gentypos._load_substitutions_file(str(path))
    assert result == {}

def test_load_file_non_ascii(tmp_path):
    """Test branch where line with non-ASCII characters is skipped (Line 466)."""
    path = tmp_path / "words.txt"
    # µ is non-ASCII (ord(µ) = 181)
    path.write_bytes("apple\nµ\nbanana\n".encode("utf-8"))

    result = gentypos.load_file(str(path))
    assert "apple" in result
    assert "banana" in result
    assert "µ" not in result
    assert len(result) == 2

def test_format_typos_list_format():
    """Test branch for 'list' output format (Line 555)."""
    typo_to_correct = {"teh": "the"}
    result = gentypos.format_typos(typo_to_correct, output_format="list")
    assert result == ["teh"]

def test_format_typos_fallback():
    """Test fallback branch for unknown output format."""
    typo_to_correct = {"teh": "the"}
    result = gentypos.format_typos(typo_to_correct, output_format="unknown")
    assert result == ["teh"]

def test_main_max_length_only(tmp_path, monkeypatch):
    """Test branch in main where min_length is None but max_length is set (Line 921)."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello")

    # We need a dictionary or --no-filter
    # Mocking sys.argv
    monkeypatch.setattr(sys, "argv", [
        "gentypos.py",
        str(input_file),
        "--max-length", "10",
        "--no-filter",
        "--format", "list"
    ])

    # Mock _run_typo_generation and validate_config to verify config building
    with patch("gentypos._run_typo_generation"), \
         patch("gentypos.validate_config") as mock_validate:
        gentypos.main()

        # Get the config passed to validate_config
        config = mock_validate.call_args[0][0]

        # Verify that max_length is set
        assert config['word_length']['max_length'] == 10
        # min_length is set to 0 by default in CLI mode if not provided
        assert config['word_length']['min_length'] == 0
