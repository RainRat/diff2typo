from unittest.mock import patch
import sys
import logging
from pathlib import Path
import pytest
import gentypos

@pytest.fixture
def empty_config_file(tmp_path):
    config = tmp_path / "test_config.yaml"
    config.write_text("{}", encoding="utf-8")
    return str(config)

def test_gentypos_dry_run_cli_output(caplog, empty_config_file):
    test_args = [
        "gentypos.py",
        "hello",
        "world",
        "-c", empty_config_file,
        "--dry-run"
    ]
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit) as exc_info:
            with caplog.at_level(logging.INFO):
                gentypos.main()

    assert exc_info.value.code == 0

    log_text = "\n".join(record.message for record in caplog.records)

    assert "--- GENTYPOS DRY RUN ---" in log_text
    assert "CLI Words: ['hello', 'world']" in log_text
    assert "Sample Typo Generation Preview:" in log_text
    assert "hello" in log_text
    assert "world" in log_text
    assert "Dry run complete. No files were written." in log_text

def test_gentypos_dry_run_does_not_write_files(tmp_path, empty_config_file):
    out_file = tmp_path / "typos.txt"
    test_args = [
        "gentypos.py",
        "hello",
        "-c", empty_config_file,
        "--dry-run",
        "-o", str(out_file)
    ]
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit) as exc_info:
            gentypos.main()

    assert exc_info.value.code == 0
    assert not out_file.exists()


def test_gentypos_dry_run_with_dictionary(caplog, empty_config_file, tmp_path):
    dict_file = tmp_path / "dict.txt"
    dict_file.write_text("hllo\nhello\nworld", encoding="utf-8")

    test_args = [
        "gentypos.py",
        "hello",
        "-c", empty_config_file,
        "-d", str(dict_file),
        "--dry-run"
    ]
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit) as exc_info:
            with caplog.at_level(logging.INFO):
                gentypos.main()

    assert exc_info.value.code == 0

    log_text = "\n".join(record.message for record in caplog.records)
    assert "Kept:" in log_text
    assert "Filtered:" in log_text
    assert "hllo" in log_text
