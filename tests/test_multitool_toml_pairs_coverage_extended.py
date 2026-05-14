import multitool
from unittest.mock import patch
import io
import pytest
import logging

def test_toml_replacements_list_of_tables(tmp_path):
    toml_content = """
[[replacements]]
typo = "teh"
correct = "the"

[[replacements]]
typo = "recieve"
correction = "receive"
"""
    toml_file = tmp_path / "list_of_tables.toml"
    toml_file.write_text(toml_content)

    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        with patch('sys.argv', ['multitool.py', 'pairs', str(toml_file), '-f', 'csv', '-R']):
            multitool.main()
            output = fake_out.getvalue()
            assert "teh,the" in output
            assert "recieve,receive" in output

def test_toml_replacements_dict_of_lists(tmp_path):
    toml_content = """
[replacements]
common = [
    { typo = "teh", correct = "the" }
]
rare = [
    { typo = "recieve", correction = "receive" }
]
"""
    toml_file = tmp_path / "dict_of_lists.toml"
    toml_file.write_text(toml_content)

    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        with patch('sys.argv', ['multitool.py', 'pairs', str(toml_file), '-f', 'csv', '-R']):
            multitool.main()
            output = fake_out.getvalue()
            assert "teh,the" in output
            assert "recieve,receive" in output

def test_toml_replacements_dict_simple(tmp_path):
    toml_content = """
[replacements]
teh = "the"
recieve = "receive"
"""
    toml_file = tmp_path / "dict_simple.toml"
    toml_file.write_text(toml_content)

    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        with patch('sys.argv', ['multitool.py', 'pairs', str(toml_file), '-f', 'csv', '-R']):
            multitool.main()
            output = fake_out.getvalue()
            assert "teh,the" in output
            assert "recieve,receive" in output

def test_toml_flat_mapping(tmp_path):
    toml_content = """
teh = "the"
recieve = "receive"
"""
    toml_file = tmp_path / "flat.toml"
    toml_file.write_text(toml_content)

    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        with patch('sys.argv', ['multitool.py', 'pairs', str(toml_file), '-f', 'csv', '-R']):
            multitool.main()
            output = fake_out.getvalue()
            assert "teh,the" in output
            assert "recieve,receive" in output

def test_toml_fallback_to_toml_package(tmp_path):
    toml_content = """
teh = "the"
"""
    toml_file = tmp_path / "fallback.toml"
    toml_file.write_text(toml_content)

    with patch('multitool._TOMLLIB_AVAILABLE', False):
        with patch('multitool._TOML_AVAILABLE', True):
            with patch('sys.stdout', new=io.StringIO()) as fake_out:
                with patch('sys.argv', ['multitool.py', 'pairs', str(toml_file), '-f', 'csv', '-R']):
                    multitool.main()
                    output = fake_out.getvalue()
                    assert "teh,the" in output

def test_toml_no_support(tmp_path, caplog):
    toml_content = "teh = 'the'"
    toml_file = tmp_path / "no_support.toml"
    toml_file.write_text(toml_content)

    with patch('multitool._TOMLLIB_AVAILABLE', False):
        with patch('multitool._TOML_AVAILABLE', False):
            with caplog.at_level(logging.ERROR):
                with patch('sys.argv', ['multitool.py', 'pairs', str(toml_file), '-f', 'csv', '-R']):
                    multitool.main()
                    assert "TOML support requires Python 3.11+ or the 'toml' package." in caplog.text

def test_toml_malformed(tmp_path, caplog):
    toml_file = tmp_path / "malformed.toml"
    toml_file.write_text("invalid toml content")

    with caplog.at_level(logging.ERROR):
        with patch('sys.argv', ['multitool.py', 'pairs', str(toml_file), '-f', 'csv', '-R']):
            multitool.main()
            assert "Failed to parse TOML" in caplog.text
