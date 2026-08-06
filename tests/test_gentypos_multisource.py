import os
import sys
import types
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

def test_resolve_input_sources_files_and_globs(tmp_path):
    f1 = tmp_path / "words1.txt"
    f1.write_text("hello\n")
    f2 = tmp_path / "words2.csv"
    f2.write_text("world\n")

    inputs = [str(f1), str(tmp_path / "*.csv")]
    resolved = gentypos._resolve_input_sources(inputs)

    assert str(f1) in resolved
    assert str(f2) in resolved
    assert len(resolved) == 2

def test_resolve_input_sources_recursive_dir(tmp_path):
    sub_dir = tmp_path / "nested"
    sub_dir.mkdir()

    f1 = sub_dir / "words.txt"
    f1.write_text("apple\n")

    ignored_dir = tmp_path / "node_modules"
    ignored_dir.mkdir()
    f2 = ignored_dir / "ignored.txt"
    f2.write_text("banana\n")

    f3 = sub_dir / "words.exe"
    f3.write_text("cherry\n")

    resolved = gentypos._resolve_input_sources([str(tmp_path)])

    assert str(f1) in resolved
    assert str(f2) not in resolved
    assert str(f3) not in resolved

def test_load_words_from_sources(tmp_path):
    f1 = tmp_path / "words1.txt"
    f1.write_text("apple\n")
    f2 = tmp_path / "words2.txt"
    f2.write_text("banana\n")

    words = gentypos.load_words_from_sources([str(f1), str(f2)])
    assert words == {"apple", "banana"}

def test_extract_config_settings_single_vs_list():
    config1 = {"input_file": "words.txt"}
    settings1 = gentypos._extract_config_settings(config1)
    assert settings1.input_files == ["words.txt"]

    config2 = {"input_file": ["words1.txt", "words2.txt"]}
    settings2 = gentypos._extract_config_settings(config2)
    assert settings2.input_files == ["words1.txt", "words2.txt"]

    config3 = {}
    settings3 = gentypos._extract_config_settings(config3)
    assert settings3.input_files == []
