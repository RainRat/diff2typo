import json
import os
import pytest
from types import SimpleNamespace
import gentypos


def test_substitutions_multiple_files_cli(tmp_path):
    f1 = tmp_path / "subs1.json"
    f2 = tmp_path / "subs2.json"

    f1.write_text(json.dumps({"ph": ["f"]}), encoding="utf-8")
    f2.write_text(json.dumps({"th": ["teh"]}), encoding="utf-8")

    settings = SimpleNamespace(
        custom_substitutions_config={},
        substitutions_file=[str(f1), str(f2)],
        ad_hoc=None,
        enable_custom_substitutions=True,
        enable_adjacent_substitutions=False,
    )

    _, custom_subs = gentypos._setup_generation_tools(settings)

    assert "ph" in custom_subs
    assert "f" in custom_subs["ph"]
    assert "th" in custom_subs
    assert "teh" in custom_subs["th"]


def test_substitutions_directory_scan(tmp_path):
    sub_dir = tmp_path / "patterns"
    sub_dir.mkdir()

    f1 = sub_dir / "subs1.json"
    f2 = sub_dir / "subs2.txt"

    f1.write_text(json.dumps({"e": ["a"]}), encoding="utf-8")
    f2.write_text("i -> u\n", encoding="utf-8")

    settings = SimpleNamespace(
        custom_substitutions_config={},
        substitutions_file=[str(sub_dir)],
        ad_hoc=None,
        enable_custom_substitutions=True,
        enable_adjacent_substitutions=False,
    )

    _, custom_subs = gentypos._setup_generation_tools(settings)

    assert "e" in custom_subs
    assert "a" in custom_subs["e"]
    assert "i" in custom_subs
    assert "u" in custom_subs["i"]


def test_substitutions_glob_pattern(tmp_path):
    sub_dir = tmp_path / "glob_test"
    sub_dir.mkdir()

    f1 = sub_dir / "p1.json"
    f2 = sub_dir / "p2.json"
    f3 = sub_dir / "ignored.other"

    f1.write_text(json.dumps({"a": ["e"]}), encoding="utf-8")
    f2.write_text(json.dumps({"b": ["v"]}), encoding="utf-8")
    f3.write_text("x -> y\n", encoding="utf-8")

    glob_pattern = str(sub_dir / "*.json")

    settings = SimpleNamespace(
        custom_substitutions_config={},
        substitutions_file=[glob_pattern],
        ad_hoc=None,
        enable_custom_substitutions=True,
        enable_adjacent_substitutions=False,
    )

    _, custom_subs = gentypos._setup_generation_tools(settings)

    assert "a" in custom_subs
    assert "e" in custom_subs["a"]
    assert "b" in custom_subs
    assert "v" in custom_subs["b"]


def test_substitutions_yaml_list_config(tmp_path):
    f1 = tmp_path / "c1.json"
    f2 = tmp_path / "c2.json"

    f1.write_text(json.dumps({"oo": ["u"]}), encoding="utf-8")
    f2.write_text(json.dumps({"ee": ["i"]}), encoding="utf-8")

    config = {
        "substitutions_file": [str(f1), str(f2)],
        "input_file": None,
        "dictionary_file": None,
        "output_file": "-",
        "output_format": "arrow",
    }

    settings = gentypos._extract_config_settings(config)

    assert settings.substitutions_file == [str(f1), str(f2)]

    _, custom_subs = gentypos._setup_generation_tools(settings)

    assert "oo" in custom_subs
    assert "u" in custom_subs["oo"]
    assert "ee" in custom_subs
    assert "i" in custom_subs["ee"]
