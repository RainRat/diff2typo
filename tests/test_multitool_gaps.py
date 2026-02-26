import json
import logging
import sys
import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_write_output_json(tmp_path):
    output_json = tmp_path / "output.json"
    items = ["apple", "banana"]
    multitool.write_output(items, str(output_json), output_format='json')
    with open(output_json) as f:
        data = json.load(f)
    assert data == ["apple", "banana"]

def test_write_output_csv(tmp_path):
    output_csv = tmp_path / "output.csv"
    items = ["apple", "banana"]
    multitool.write_output(items, str(output_csv), output_format='csv')
    content_csv = output_csv.read_text()
    assert "apple" in content_csv
    assert "banana" in content_csv

def test_write_output_markdown(tmp_path):
    output_md = tmp_path / "output.md"
    items = ["apple", "banana"]
    multitool.write_output(items, str(output_md), output_format='markdown')
    content_md = output_md.read_text()
    assert "- apple\n" in content_md
    assert "- banana\n" in content_md

def test_write_output_fallback_to_line(tmp_path):
    output_line = tmp_path / "output.txt"
    items = ["apple", "banana"]
    multitool.write_output(items, str(output_line), output_format='unknown')
    content_line = output_line.read_text()
    assert "apple\nbanana\n" == content_line

def test_load_mapping_file_invalid_json(tmp_path, caplog):
    invalid_json = tmp_path / "mapping.json"
    invalid_json.write_text("{")
    with caplog.at_level(logging.ERROR):
        mapping = multitool._load_mapping_file(str(invalid_json))
    assert mapping == {}
    assert "Failed to parse JSON" in caplog.text

def test_load_mapping_file_invalid_yaml(tmp_path, caplog):
    invalid_yaml = tmp_path / "mapping.yaml"
    invalid_yaml.write_text(":")
    with caplog.at_level(logging.ERROR):
        mapping = multitool._load_mapping_file(str(invalid_yaml))
    assert mapping == {}
    assert "Failed to parse YAML" in caplog.text

def test_load_mapping_file_yaml_non_dict(tmp_path):
    list_yaml = tmp_path / "mapping.yml"
    list_yaml.write_text("- item1\n- item2")
    mapping = multitool._load_mapping_file(str(list_yaml))
    assert mapping == {}

def test_load_mapping_file_empty(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    mapping = multitool._load_mapping_file(str(empty_file))
    assert mapping == {}

def test_load_mapping_file_comments_only(tmp_path):
    comment_file = tmp_path / "comments.txt"
    comment_file.write_text("# comment\n  \n# another")
    mapping = multitool._load_mapping_file(str(comment_file))
    assert mapping == {}

def test_load_mapping_file_table_heuristic(tmp_path):
    table_file = tmp_path / "mapping.toml"
    table_file.write_text('key = "value"\n# comment\nfoo = "bar"')
    mapping = multitool._load_mapping_file(str(table_file))
    assert mapping == {"key": "value", "foo": "bar"}

def test_load_mapping_file_yaml_missing_dependency(tmp_path, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "yaml", None)
    yaml_file = tmp_path / "mapping.yaml"
    yaml_file.write_text("a: b")
    with caplog.at_level(logging.ERROR):
        mapping = multitool._load_mapping_file(str(yaml_file))
    assert mapping == {}
    assert "PyYAML not installed" in caplog.text

def test_mode_help_action_unknown_mode_error():
    parser = argparse.ArgumentParser()
    action = multitool.ModeHelpAction(option_strings=['--mode-help'], dest='mode_help')
    with patch.object(parser, 'error', side_effect=SystemExit(2)) as mock_error:
        with pytest.raises(SystemExit):
            action(parser, argparse.Namespace(), 'unknown_mode')
        mock_error.assert_called_once()
        assert "Unknown mode: unknown_mode" in mock_error.call_args[0][0]

def test_mode_help_action_valid_mode_with_tty(capsys):
    parser = argparse.ArgumentParser()
    action = multitool.ModeHelpAction(option_strings=['--mode-help'], dest='mode_help')
    valid_mode = list(multitool.MODE_DETAILS.keys())[0]
    with patch('sys.stdout.isatty', return_value=True):
        with pytest.raises(SystemExit):
            action(parser, argparse.Namespace(), valid_mode)
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert f"MODE:" in output
    assert valid_mode.upper() in output

def test_mode_help_action_valid_mode_no_tty(capsys):
    parser = argparse.ArgumentParser()
    action = multitool.ModeHelpAction(option_strings=['--mode-help'], dest='mode_help')
    valid_mode = list(multitool.MODE_DETAILS.keys())[0]
    with patch('sys.stdout.isatty', return_value=False):
        with pytest.raises(SystemExit):
            action(parser, argparse.Namespace(), valid_mode)
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert f"MODE:" in output
    assert valid_mode.upper() in output

def test_main_exit_on_file_not_found(monkeypatch, caplog):
    monkeypatch.setattr(sys, 'argv', ['multitool.py', 'line', '--input', 'nonexistent.txt', '--output', 'out.txt'])
    with pytest.raises(SystemExit) as excinfo:
        multitool.main()
    assert excinfo.value.code == 1
    assert "File not found: 'nonexistent.txt'" in caplog.text

def test_main_exit_on_generic_file_not_found(monkeypatch, caplog):
    def mock_line_mode(*args, **kwargs):
        raise FileNotFoundError("Generic error")
    monkeypatch.setattr(multitool, 'line_mode', mock_line_mode)
    monkeypatch.setattr(sys, 'argv', ['multitool.py', 'line', '--input', 'dummy.txt', '--output', 'out.txt'])
    with pytest.raises(SystemExit) as excinfo:
        multitool.main()
    assert excinfo.value.code == 1
    assert "File not found: Generic error" in caplog.text

def test_main_exit_on_unexpected_exception(monkeypatch, caplog):
    def mock_line_mode(*args, **kwargs):
        raise Exception("Boom!")
    monkeypatch.setattr(multitool, 'line_mode', mock_line_mode)
    monkeypatch.setattr(sys, 'argv', ['multitool.py', 'line', '--input', 'dummy.txt', '--output', 'out.txt'])
    with pytest.raises(SystemExit) as excinfo:
        multitool.main()
    assert excinfo.value.code == 1
    assert "An unexpected error occurred: Boom!" in caplog.text

def test_load_and_clean_file_skips_empty_lines(tmp_path):
    f = tmp_path / "empty_lines.txt"
    f.write_text("apple\n\nbanana")
    _, cleaned, _ = multitool._load_and_clean_file(str(f), 1, 100)
    assert cleaned == ["apple", "banana"]

def test_extract_table_items_handles_missing_closing_quote(tmp_path):
    f = tmp_path / "table.txt"
    f.write_text('key = "value_without_closing_quote')
    out = tmp_path / "out.txt"
    multitool.table_mode([str(f)], str(out), 1, 100, False, right_side=True, clean_items=False)
    assert out.read_text().strip() == "value_without_closing_quote"

def test_extract_table_items_left_side(tmp_path):
    f = tmp_path / "table.txt"
    f.write_text('key1 = "val1"')
    out = tmp_path / "out.txt"
    multitool.table_mode([str(f)], str(out), 1, 100, False, right_side=False, clean_items=False)
    assert out.read_text().strip() == "key1"

def test_extract_backtick_items_ignores_empty_backticks(tmp_path):
    f = tmp_path / "backtick.txt"
    f.write_text("Some `` empty backticks")
    out = tmp_path / "out.txt"
    multitool.backtick_mode([str(f)], str(out), 1, 100, False)
    assert out.read_text().strip() == ""

def test_extract_yaml_items_skips_none_documents(tmp_path):
    f = tmp_path / "docs.yaml"
    f.write_text("---\nkey: val\n---\n")
    out = tmp_path / "out.txt"
    multitool.yaml_mode([str(f)], str(out), 1, 100, False, key="key")
    assert out.read_text().strip() == "val"

def test_extract_regex_items_with_multiple_groups(tmp_path):
    f = tmp_path / "regex.txt"
    f.write_text("apple:orange banana:grape")
    out = tmp_path / "out.txt"
    multitool.regex_mode([str(f)], str(out), 1, 100, False, pattern=r"(\w+):(\w+)")
    results = out.read_text().splitlines()
    assert results == ["apple", "orange", "banana", "grape"]

def test_minimal_formatter_non_info():
    formatter = multitool.MinimalFormatter()
    # INFO level should be clean
    record_info = logging.LogRecord("name", logging.INFO, "path", 10, "info message", None, None)
    assert formatter.format(record_info) == "info message"
    # WARNING level should have prefix
    record_warn = logging.LogRecord("name", logging.WARNING, "path", 10, "warn message", None, None)
    assert formatter.format(record_warn) == "WARNING: warn message"

def test_write_output_yaml_fallback(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)
    output_yaml = tmp_path / "output.yaml"
    items = ["apple", "banana"]
    multitool.write_output(items, str(output_yaml), output_format='yaml')
    content = output_yaml.read_text()
    assert "- apple\n- banana\n" == content

def test_write_paired_output_yaml_fallback(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)
    output_yaml = tmp_path / "output.yaml"
    pairs = [("teh", "the")]
    multitool._write_paired_output(pairs, str(output_yaml), output_format='yaml', mode_label="Test")
    content = output_yaml.read_text()
    assert "teh: the\n" == content

def test_extract_markdown_items_empty(tmp_path):
    f = tmp_path / "empty_items.md"
    f.write_text("- \n-    \n- item")
    out = tmp_path / "out.txt"
    multitool.markdown_mode([str(f)], str(out), 1, 100, False)
    assert out.read_text().strip() == "item"

def test_write_output_with_limit(tmp_path):
    output_line = tmp_path / "output.txt"
    items = ["apple", "banana", "cherry"]
    multitool.write_output(items, str(output_line), output_format='line', limit=2)
    content = output_line.read_text().splitlines()
    assert content == ["apple", "banana"]

def test_write_paired_output_with_limit(tmp_path):
    output_arrow = tmp_path / "output.txt"
    pairs = [("a", "1"), ("b", "2"), ("c", "3")]
    multitool._write_paired_output(pairs, str(output_arrow), output_format='arrow', mode_label="Test", limit=2)
    content = output_arrow.read_text().splitlines()
    assert content == ["a -> 1", "b -> 2"]

def test_count_mode_with_limit(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("apple apple banana cherry")
    out = tmp_path / "out.txt"
    multitool.count_mode([str(f)], str(out), 1, 100, False, limit=2)
    # count_mode sorts by count descending. apple: 2, banana: 1, cherry: 1.
    # limit=2 should give apple and banana (or cherry).
    results = out.read_text().splitlines()
    assert len(results) == 2
    assert "apple" in results[0]
