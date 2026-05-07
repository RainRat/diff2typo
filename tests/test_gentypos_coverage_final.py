import logging
import sys
import runpy
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

def test_minimal_formatter_colorization():
    with patch("sys.stderr.isatty", return_value=True), \
         patch("gentypos.RESET", "\033[0m"), \
         patch.dict(gentypos.MinimalFormatter.LEVEL_COLORS, {logging.WARNING: "\033[33m"}):
        formatter = gentypos.MinimalFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=10,
            msg="warning message",
            args=None,
            exc_info=None
        )
        formatted = formatter.format(record)
        assert "\033[33mWARNING\033[0m" in formatted

def test_load_substitutions_csv_empty_rows(tmp_path):
    path = tmp_path / "subs_empty.csv"
    # Row 1: Header (to be skipped)
    # Row 2: Empty
    # Row 3: Single column (len < 2)
    # Row 4: Valid
    path.write_text("typo,correction\n\nonly_one\ne,a\n")

    result = gentypos._load_substitutions_file(str(path))
    # 'a' is the correction, 'e' is the typo. Mapping should be correction -> [typo]
    assert result == {"a": ["e"]}

def test_main_output_header_file(tmp_path, monkeypatch):
    out_file = tmp_path / "out_with_header.txt"
    monkeypatch.setattr(sys, "argv", [
        "gentypos.py", "word",
        "--output", str(out_file),
        "--no-filter",
        "--format", "list"
    ])

    # We need to ensure output_header is set in config
    # Since main() loads config and merges defaults, we can mock parse_yaml_config
    # or just let it use defaults if we use a format that has a default header.
    # 'table' format has a default header "[default.extend-words]"

    monkeypatch.setattr(sys, "argv", [
        "gentypos.py", "word",
        "--output", str(out_file),
        "--no-filter",
        "--format", "table"
    ])

    with patch("gentypos._run_typo_generation", return_value={"wrd": "word"}):
        gentypos.main()

    content = out_file.read_text()
    assert "[default.extend-words]" in content
    assert 'wrd = "word"' in content

def test_gentypos_main_entry_point():
    with patch("sys.argv", ["gentypos.py", "--help"]):
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_path("gentypos.py", run_name="__main__")
        assert excinfo.value.code == 0
