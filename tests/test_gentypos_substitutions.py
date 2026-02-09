import json
import csv
import os
import yaml
import pytest
from gentypos import _load_substitutions_file, _setup_generation_tools, _run_typo_generation
from types import SimpleNamespace

def test_load_substitutions_json_typostats(tmp_path):
    # Format matching typostats.py output
    data = {
        "replacements": [
            {"correct": "a", "typo": "e", "count": 10},
            {"correct": "t", "typo": "th", "count": 5}
        ]
    }
    p = tmp_path / "subs.json"
    p.write_text(json.dumps(data))

    subs = _load_substitutions_file(str(p))
    assert subs["a"] == ["e"]
    assert subs["t"] == ["th"]

def test_load_substitutions_json_plain(tmp_path):
    data = {"e": ["a", "i"], "o": "0"}
    p = tmp_path / "subs.json"
    p.write_text(json.dumps(data))

    subs = _load_substitutions_file(str(p))
    assert subs["e"] == ["a", "i"]
    assert subs["o"] == ["0"]

def test_load_substitutions_csv_typostats(tmp_path):
    p = tmp_path / "subs.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["correct_char", "typo_char", "count"])
        writer.writerow(["s", "z", "3"])
        writer.writerow(["m", "rn", "2"])

    subs = _load_substitutions_file(str(p))
    assert subs["s"] == ["z"]
    assert subs["m"] == ["rn"]

def test_load_substitutions_csv_plain(tmp_path):
    p = tmp_path / "subs.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["a", "4"])
        writer.writerow(["i", "1"])

    subs = _load_substitutions_file(str(p))
    assert subs["a"] == ["4"]
    assert subs["i"] == ["1"]

def test_load_substitutions_yaml(tmp_path):
    data = {"l": ["1", "ll"], "c": "k"}
    p = tmp_path / "subs.yaml"
    p.write_text(yaml.dump(data))

    subs = _load_substitutions_file(str(p))
    assert "1" in subs["l"]
    assert "ll" in subs["l"]
    assert subs["c"] == ["k"]

def test_integration_substitutions_file(tmp_path):
    # Create a substitutions file
    subs_data = {"replacements": [{"correct": "x", "typo": "ks"}]}
    subs_file = tmp_path / "my_subs.json"
    subs_file.write_text(json.dumps(subs_data))

    settings = SimpleNamespace(
        enable_custom_substitutions=True,
        custom_substitutions_config={}, # No config substitutions
        substitutions_file=str(subs_file),
        enable_adjacent_substitutions=False, # Disable adjacent to isolate custom
        include_diagonals=False,
        typo_types={'replacement': True}, # Only replacement
        repeat_modifications=1,
        min_length=0,
        max_length=100,
        transposition_distance=1,
        quiet=True,
        dictionary_file=None
    )

    adjacent_keys, custom_subs = _setup_generation_tools(settings)

    assert "x" in custom_subs
    assert "ks" in custom_subs["x"]

    word_list = ["box"]
    all_words = set()

    results = _run_typo_generation(word_list, all_words, settings, adjacent_keys, custom_subs, quiet=True)

    # "box" should generate "boks"
    assert "boks" in results
    assert results["boks"] == "box"

def test_merge_substitutions(tmp_path):
    # Config has 'a' -> 'e'
    # File has 'a' -> 'i' and 'b' -> 'v'
    subs_file = tmp_path / "extra.yaml"
    subs_file.write_text(yaml.dump({"a": "i", "b": "v"}))

    settings = SimpleNamespace(
        enable_custom_substitutions=True,
        custom_substitutions_config={"a": ["e"]},
        substitutions_file=str(subs_file),
        enable_adjacent_substitutions=False,
        include_diagonals=False,
        quiet=True
    )

    _, custom_subs = _setup_generation_tools(settings)

    assert "e" in custom_subs["a"]
    assert "i" in custom_subs["a"]
    assert "v" in custom_subs["b"]
