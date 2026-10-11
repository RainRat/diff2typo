import tempfile
import os
import sys
import subprocess
import pytest
import gentypos


def test_load_file_json():
    with tempfile.TemporaryDirectory() as td:
        file_path = os.path.join(td, "words.json")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write('["apple", "banana", "cherry"]')
        words = gentypos.load_file(file_path)
        assert words == {"apple", "banana", "cherry"}

        # Test dictionary structure in JSON
        file_path_dict = os.path.join(td, "dict.json")
        with open(file_path_dict, "w", encoding="utf-8") as f:
            f.write('{"word1": "orange", "words": ["grape", "melon"]}')
        words_dict = gentypos.load_file(file_path_dict)
        assert words_dict == {"word1", "orange", "words", "grape", "melon"}


def test_load_file_yaml():
    if not gentypos._YAML_AVAILABLE:
        pytest.skip("PyYAML not installed")

    with tempfile.TemporaryDirectory() as td:
        file_path = os.path.join(td, "words.yaml")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("- apple\n- banana\n- cherry\n")
        words = gentypos.load_file(file_path)
        assert words == {"apple", "banana", "cherry"}


def test_load_file_toml():
    if not gentypos._TOMLLIB_AVAILABLE and not gentypos._TOML_AVAILABLE:
        pytest.skip("TOML not supported in current environment")

    with tempfile.TemporaryDirectory() as td:
        file_path = os.path.join(td, "words.toml")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write('[fruits]\nwords = ["apple", "banana"]\n')
        words = gentypos.load_file(file_path)
        assert words == {"fruits", "words", "apple", "banana"}


def test_load_file_csv():
    with tempfile.TemporaryDirectory() as td:
        file_path = os.path.join(td, "words.csv")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("typo,correction\nhelo,hello\nwrd,word\n")
        words = gentypos.load_file(file_path)
        assert "hello" in words
        assert "helo" in words
        assert "word" in words
        assert "wrd" in words


def test_load_file_markdown():
    with tempfile.TemporaryDirectory() as td:
        file_path = os.path.join(td, "words.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(
                "# Wordlist\n\n"
                "- apple\n"
                "- banana\n\n"
                "| Word | Description |\n"
                "| --- | --- |\n"
                "| cherry | fruit |\n"
            )
        words = gentypos.load_file(file_path)
        assert "apple" in words
        assert "banana" in words
        assert "cherry" in words
        assert "fruit" in words


def test_load_file_html():
    with tempfile.TemporaryDirectory() as td:
        file_path = os.path.join(td, "words.html")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(
                "<!DOCTYPE html><html><body>\n"
                "<table><tr><td>apple</td><td>banana</td></tr></table>\n"
                "<ul><li>cherry</li></ul>\n"
                "</body></html>"
            )
        words = gentypos.load_file(file_path)
        assert "apple" in words
        assert "banana" in words
        assert "cherry" in words


def test_load_file_plaintext():
    with tempfile.TemporaryDirectory() as td:
        file_path = os.path.join(td, "words.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("apple\nbanana\ncherry\n")
        words = gentypos.load_file(file_path)
        assert words == {"apple", "banana", "cherry"}


def test_cli_integration_json_input():
    with tempfile.TemporaryDirectory() as td:
        json_file = os.path.join(td, "words.json")
        with open(json_file, "w", encoding="utf-8") as f:
            f.write('["hello", "world"]')

        res = subprocess.run(
            [sys.executable, "gentypos.py", "-i", json_file, "--no-filter", "-f", "arrow"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "hello" in res.stdout
        assert "world" in res.stdout
        assert "Loaded 2 words from the small dictionary" in res.stderr
