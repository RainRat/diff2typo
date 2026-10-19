import sys
import logging
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos


def test_missing_custom_config_file(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    missing_config = tmp_path / "non_existent_config.yaml"
    test_args = ["gentypos.py", "--config", str(missing_config)]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        gentypos.main()

    assert exc_info.value.code == 1
    assert f"Configuration file '{missing_config}' not found." in caplog.text


def test_transposition_options_non_dict_override(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "custom.yaml"
    config_file.write_text("transposition_options: invalid_string\n", encoding="utf-8")

    test_args = [
        "gentypos.py",
        "hello",
        "--config", str(config_file),
        "-T", "2",
        "--no-filter",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    gentypos.main()

    captured = capsys.readouterr()
    assert "hlelo" in captured.out or "hleol" in captured.out or "hello" in captured.out or "-> hello" in captured.out


def test_word_length_non_dict_override(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "custom.yaml"
    config_file.write_text("word_length: 12345\n", encoding="utf-8")

    test_args = [
        "gentypos.py",
        "testing",
        "--config", str(config_file),
        "-m", "3",
        "-M", "10",
        "--no-filter",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    gentypos.main()

    captured = capsys.readouterr()
    assert "testing" in captured.out


def test_cli_mode_existing_dictionary_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    dict_file = tmp_path / "large_dictionary.txt"
    dict_file.write_text("fello\ntesting\n", encoding="utf-8")

    test_args = [
        "gentypos.py",
        "hello",
        "-d", str(dict_file),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    gentypos.main()

    captured = capsys.readouterr()
    assert "fello" not in captured.out


def test_cli_mode_substitutions_file_and_format(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sub_file = tmp_path / "subs.json"
    sub_file.write_text('{"e": ["a"]}', encoding="utf-8")

    test_args = [
        "gentypos.py",
        "hello",
        "-s", str(sub_file),
        "--format", "csv",
        "--no-filter",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    gentypos.main()

    captured = capsys.readouterr()
    assert "hallo,hello" in captured.out


def test_minimal_formatter_color_output():
    formatter = gentypos.MinimalFormatter()
    record = logging.LogRecord(
        name="gentypos",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="Test warning message",
        args=(),
        exc_info=None,
    )

    with patch("gentypos._should_enable_color", return_value=True):
        formatted = formatter.format(record)

    assert "\033[1;33mWARNING\033[0m: Test warning message" in formatted
