import sys
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos


def test_gentypos_limit_option(capsys, monkeypatch):
    """Verify that -L / --limit restricts output typos count."""
    test_args = ["gentypos.py", "hello", "-L", "3", "--no-filter", "-q"]
    monkeypatch.setattr(sys, "argv", test_args)

    gentypos.main()

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]
    assert len(lines) == 3


def test_gentypos_limit_dry_run(caplog, monkeypatch):
    """Verify that --limit is reported in dry-run output."""
    import logging
    test_args = ["gentypos.py", "hello", "--limit", "5", "--dry-run"]
    monkeypatch.setattr(sys, "argv", test_args)

    with caplog.at_level(logging.INFO):
        with pytest.raises(SystemExit) as exc_info:
            gentypos.main()

    assert exc_info.value.code == 0
    assert any("Limit: 5" in msg for msg in caplog.messages)


def test_gentypos_limit_from_config(tmp_path, capsys, monkeypatch):
    """Verify that limit specified in config file is respected."""
    import yaml
    words_file = tmp_path / "words.txt"
    words_file.write_text("hello\nworld\n", encoding="utf-8")
    config_file = tmp_path / "custom_gentypos.yaml"
    config_data = {
        "input_file": str(words_file),
        "output_file": "-",
        "limit": 2,
        "word_length": {"min_length": 3},
        "no_filter": True,
        "quiet": True,
    }
    config_file.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    test_args = ["gentypos.py", "--config", str(config_file), "-q", "--no-filter"]
    monkeypatch.setattr(sys, "argv", test_args)

    gentypos.main()

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]
    assert len(lines) == 2
