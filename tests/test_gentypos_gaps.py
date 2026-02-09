import sys
import logging
import pytest
import json
import csv
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(gentypos, "tqdm", lambda iterable, *_, **__: iterable)

def test_is_cli_mode_adhoc_words(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.txt"
    # Provide adhoc words as positional arguments
    # We use --no-filter to avoid needing a large dictionary file
    monkeypatch.setattr(sys, 'argv', ['gentypos.py', 'hello', 'world', '--output', str(output_file), '--no-filter'])
    gentypos.main()
    # Check that output contains typos for 'hello' and 'world'
    content = output_file.read_text()
    assert len(content.splitlines()) > 0
    # Minimal check for 'hello' typo (deletion of 'h' -> 'ello')
    assert "ello" in content

def test_legacy_word_flag(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.txt"
    # Provide adhoc words via legacy --word flag
    monkeypatch.setattr(sys, 'argv', ['gentypos.py', '--word', 'test', '--output', str(output_file), '--no-filter'])
    gentypos.main()
    content = output_file.read_text()
    assert len(content.splitlines()) > 0
    # 'test' -> 'est' (deletion)
    assert "est" in content

def test_load_substitutions_csv_with_header(tmp_path):
    p = tmp_path / "subs.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["typo", "correction"]) # Recognized as a header
        writer.writerow(["a", "e"])
    subs = gentypos._load_substitutions_file(str(p))
    assert subs["a"] == ["e"]
    assert "typo" not in subs

def test_cli_mode_missing_dictionary_warning(monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)
    # Specified dictionary file that does not exist
    dict_file = tmp_path / "nonexistent_dict.txt"

    config = {
        'dictionary_file': str(dict_file),
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))

    monkeypatch.setattr(sys, 'argv', ['gentypos.py', 'test', '--config', str(config_file)])
    with caplog.at_level(logging.WARNING):
        gentypos.main()
    assert "Dictionary file" in caplog.text
    assert "not found. Skipping filtering in CLI mode." in caplog.text

def test_extract_config_settings_unknown_format(caplog):
    config = {'output_format': 'invalid_format'}
    with caplog.at_level(logging.WARNING):
        settings = gentypos._extract_config_settings(config)
    assert "Unknown output format 'invalid_format'" in caplog.text
    assert settings.output_format == 'arrow'

def test_extract_config_settings_table_header():
    config = {'output_format': 'table'}
    settings = gentypos._extract_config_settings(config)
    assert settings.output_header == "[default.extend-words]"

def test_load_substitutions_file_not_found(caplog):
    with pytest.raises(SystemExit):
        with caplog.at_level(logging.ERROR):
            gentypos._load_substitutions_file("nonexistent_subs.json")
    assert "not found" in caplog.text

def test_main_invalid_config_file(monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['gentypos.py', '--config', 'missing_config.yaml'])
    with pytest.raises(SystemExit):
        with caplog.at_level(logging.ERROR):
            gentypos.main()
    assert "Configuration file 'missing_config.yaml' not found" in caplog.text

def test_main_stdout_output(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    # Use '-' for stdout
    # Ensure min_length is small enough for 'cat'
    monkeypatch.setattr(sys, 'argv', ['gentypos.py', 'cat', '--output', '-', '--no-filter'])
    gentypos.main()
    captured = capsys.readouterr()
    # 'cat' -> 'at' (deletion)
    assert "at -> cat" in captured.out

def test_main_cli_substitutions_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    subs_file = tmp_path / "extra.yaml"
    subs_file.write_text(yaml.dump({"a": "z"}))
    output_file = tmp_path / "output.txt"
    # CLI mode
    monkeypatch.setattr(sys, 'argv', ['gentypos.py', 'abc', '--substitutions', str(subs_file), '--output', str(output_file), '--no-filter'])
    gentypos.main()
    content = output_file.read_text()
    assert "zbc -> abc" in content

def test_main_non_cli_overrides(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    input_file = tmp_path / "input.txt"
    input_file.write_text("cat\n")
    output_file = tmp_path / "output.txt"
    subs_file = tmp_path / "extra.yaml"
    subs_file.write_text(yaml.dump({"c": "k"}))

    config = {
        'input_file': str(input_file),
        'dictionary_file': None,
        'output_file': 'ignored.txt',
        'output_format': 'list',
        'word_length': {'min_length': 0}
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))

    monkeypatch.setattr(sys, 'argv', ['gentypos.py', '--config', str(config_file), '--output', str(output_file), '--substitutions', str(subs_file)])
    gentypos.main()
    content = output_file.read_text()
    assert "kat" in content

def test_main_output_header_stdout(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['gentypos.py', 'cat', '--output', '-', '--no-filter'])
    # Need a config to set output_header or use table format
    config = {'output_format': 'table', 'output_header': 'MY HEADER'}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))
    monkeypatch.setattr(sys, 'argv', ['gentypos.py', 'cat', '--output', '-', '--no-filter', '--config', str(config_file)])

    gentypos.main()
    captured = capsys.readouterr()
    assert "MY HEADER" in captured.out

def test_main_output_header_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.txt"
    config = {'output_format': 'table', 'output_header': 'MY FILE HEADER'}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))
    monkeypatch.setattr(sys, 'argv', ['gentypos.py', 'cat', '--output', str(output_file), '--no-filter', '--config', str(config_file)])

    gentypos.main()
    content = output_file.read_text()
    assert "MY FILE HEADER" in content

def test_main_dictionary_filtering_stats(monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello\n")
    dict_file = tmp_path / "dict.txt"
    dict_file.write_text("helloo\n") # 'helloo' is a typo of 'hello' (duplication)
    output_file = tmp_path / "output.txt"

    config = {
        'input_file': str(input_file),
        'dictionary_file': str(dict_file),
        'output_file': str(output_file),
        'output_format': 'arrow',
        'typo_types': {'duplication': True, 'deletion': False, 'transposition': False, 'replacement': False},
        'word_length': {'min_length': 0}
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))

    monkeypatch.setattr(sys, 'argv', ['gentypos.py', '--config', str(config_file), '--verbose'])
    with caplog.at_level(logging.DEBUG):
        gentypos.main()

    assert "Filtered out typo 'helloo' as it exists in the large dictionary." in caplog.text

def test_main_output_error(monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.txt"
    # Make output_file a directory to cause open() to fail for writing
    output_file.mkdir()

    monkeypatch.setattr(sys, 'argv', ['gentypos.py', 'cat', '--output', str(output_file), '--no-filter'])
    with pytest.raises(SystemExit):
        with caplog.at_level(logging.ERROR):
            gentypos.main()
    assert "Error writing to" in caplog.text

def test_load_substitutions_file_error(tmp_path, caplog):
    p = tmp_path / "error.json"
    p.write_text("{") # Invalid JSON
    with pytest.raises(SystemExit):
        with caplog.at_level(logging.ERROR):
            gentypos._load_substitutions_file(str(p))
    assert "Error loading substitutions" in caplog.text

def test_extract_config_settings_quiet_flag():
    config = {}
    settings = gentypos._extract_config_settings(config, quiet=True)
    assert settings.quiet is True

def test_main_verbose_quiet_logs(monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)

    # Test verbose
    monkeypatch.setattr(sys, 'argv', ['gentypos.py', 'test', '--verbose', '--no-filter'])
    with caplog.at_level(logging.DEBUG):
        gentypos.main()
    assert "Verbose mode enabled." in caplog.text

    caplog.clear()
    # Test quiet
    monkeypatch.setattr(sys, 'argv', ['gentypos.py', 'test', '--quiet', '--no-filter'])
    with caplog.at_level(logging.DEBUG):
        gentypos.main()
    assert "Quiet mode enabled." in caplog.text
