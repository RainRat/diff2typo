import os
import json
import csv
import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

def test_load_substitutions_json_typostats(tmp_path):
    path = tmp_path / "subs.json"
    data = {
        "replacements": [
            {"correct": "a", "typo": "e"},
            {"correct": "i", "typo": "o"}
        ]
    }
    path.write_text(json.dumps(data))
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"a": ["e"], "i": ["o"]}

def test_load_substitutions_json_plain(tmp_path):
    path = tmp_path / "subs.json"
    data = {
        "ph": ["f", "v"],
        "sh": "s"
    }
    path.write_text(json.dumps(data))
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"ph": ["f", "v"], "sh": ["s"]}

def test_load_substitutions_csv_typostats(tmp_path):
    path = tmp_path / "subs.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["correct_char", "typo_char", "count"])
        writer.writeheader()
        writer.writerow({"correct_char": "a", "typo_char": "e", "count": 10})
        writer.writerow({"correct_char": "i", "typo_char": "o", "count": 5})

    result = gentypos._load_substitutions_file(str(path))
    assert result == {"a": ["e"], "i": ["o"]}

def test_load_substitutions_csv_plain_header(tmp_path):
    path = tmp_path / "subs.csv"
    path.write_text("typo,correction\ne,a\no,i\n")
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"e": ["a"], "o": ["i"]}

def test_load_substitutions_csv_plain_no_header(tmp_path):
    path = tmp_path / "subs.csv"
    path.write_text("x,y\nz,w\n")
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"x": ["y"], "z": ["w"]}

@pytest.mark.skipif(not gentypos._YAML_AVAILABLE, reason="PyYAML not installed")
def test_load_substitutions_yaml(tmp_path):
    path = tmp_path / "subs.yaml"
    path.write_text("a: [e, i]\nph: f\n")
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"a": ["e", "i"], "ph": ["f"]}

def test_load_substitutions_yaml_no_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(gentypos, "_YAML_AVAILABLE", False)
    path = tmp_path / "subs.yaml"
    path.write_text("a: e")
    with pytest.raises(SystemExit):
        gentypos._load_substitutions_file(str(path))

def test_load_substitutions_missing_file():
    with pytest.raises(SystemExit):
        gentypos._load_substitutions_file("nonexistent.json")

def test_load_substitutions_malformed_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{invalid json}")
    with pytest.raises(SystemExit):
        gentypos._load_substitutions_file(str(path))
