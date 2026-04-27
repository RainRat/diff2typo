import sys
import os
import logging
import json
import csv
from pathlib import Path
from unittest.mock import patch
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

def test_color_initialization_enabled(monkeypatch):
    """Hits branch 50->54 (now in init_colors) by ensuring color codes are NOT empty when enabled."""
    with patch("sys.stdout.isatty", return_value=True):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("FORCE_COLOR", "1")
        gentypos.init_colors()
        assert gentypos.RED == "\033[1;31m"

    # Reset to disabled for other tests
    monkeypatch.setenv("NO_COLOR", "1")
    gentypos.init_colors()
    assert gentypos.RED == ""

def test_minimal_formatter_no_color():
    """Hits branch 71->74 (now in _should_enable_color) by using a level without a color (DEBUG)."""
    formatter = gentypos.MinimalFormatter()
    record = logging.LogRecord(
        name="test", level=logging.DEBUG, pathname="", lineno=0,
        msg="debug message", args=None, exc_info=None
    )
    # Ensure _should_enable_color(sys.stderr) is True, but 71 will be False for DEBUG
    with patch("sys.stderr.isatty", return_value=True), patch.dict(os.environ, {"FORCE_COLOR": "1"}):
        formatted = formatter.format(record)
        assert "DEBUG: debug message" in formatted

def test_load_substitutions_json_invalid_item(tmp_path):
    """Hits branch 181->180: item in replacements missing 'typo' key."""
    path = tmp_path / "subs.json"
    data = {"replacements": [{"correct": "a"}]} # missing 'typo'
    path.write_text(json.dumps(data))

    result = gentypos._load_substitutions_file(str(path))
    assert result == {}

def test_load_substitutions_json_not_dict(tmp_path):
    """Hits branch 184->233: root JSON is a list."""
    path = tmp_path / "subs_list.json"
    path.write_text(json.dumps(["a", "b"]))

    result = gentypos._load_substitutions_file(str(path))
    assert result == {}

def test_load_substitutions_csv_invalid_typostats(tmp_path):
    """Hits branch 198->197: typostats CSV with missing typo_char."""
    path = tmp_path / "subs.csv"
    path.write_text("correct_char,typo_char\na,\n") # second col empty

    result = gentypos._load_substitutions_file(str(path))
    assert result == {}

def test_load_substitutions_yaml_not_dict(tmp_path):
    """Hits branch 223->233: root YAML is a list."""
    path = tmp_path / "subs.yaml"
    path.write_text("- a\n- b")

    result = gentypos._load_substitutions_file(str(path))
    assert result == {}

def test_load_file_non_ascii_and_empty(tmp_path):
    """Hits branch 466->464: non-ASCII characters or empty lines."""
    path = tmp_path / "words.txt"
    # ñ is non-ASCII (ord > 128)
    path.write_bytes("hello\n\nñonascii\nworld\n".encode('utf-8'))

    result = gentypos.load_file(str(path))
    assert result == {"hello", "world"}

def test_format_typos_unknown_format():
    """Hits branch 555->548: unsupported output format."""
    result = gentypos.format_typos({"typo": "correct"}, "unknown")
    assert result == []

def test_main_max_length_only(monkeypatch):
    """Hits branch 921->923: --max-length provided but --min-length NOT provided."""
    monkeypatch.setattr(sys, "argv", ["gentypos.py", "word", "--max-length", "10", "--no-filter"])

    with patch("gentypos._run_typo_generation", return_value={}):
        gentypos.main()

def test_main_quiet_mode_debug_log(monkeypatch, caplog):
    """Hits branch for Quiet mode debug log in main."""
    monkeypatch.setattr(sys, "argv", ["gentypos.py", "word", "--quiet", "--no-filter"])
    with patch("gentypos._run_typo_generation", return_value={}):
        with caplog.at_level(logging.DEBUG):
            gentypos.main()
    assert "Quiet mode enabled." in caplog.text

def test_main_config_already_has_word_length(monkeypatch):
    """Target line 919->920 False branch (if 'word_length' already in config)."""
    monkeypatch.setattr(sys, "argv", ["gentypos.py", "word", "--min-length", "5", "--no-filter"])

    with patch("os.path.exists", return_value=True):
        with patch("gentypos.parse_yaml_config", return_value={'word_length': {}}):
            with patch("gentypos._run_typo_generation", return_value={}):
                gentypos.main()
