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
    # 'a' is the correction, 'e' is the typo. Mapping should be correction -> [typo]
    assert result == {"a": ["e"], "i": ["o"]}

def test_load_substitutions_csv_reversed_header(tmp_path):
    path = tmp_path / "subs.csv"
    path.write_text("correction,typo\na,e\ni,o\n")
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"a": ["e"], "i": ["o"]}

def test_load_substitutions_csv_both_typo_headers(tmp_path):
    path = tmp_path / "subs.csv"
    path.write_text("typo,before\na,e\ni,o\n")
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"a": ["e"], "i": ["o"]}

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

def test_load_substitutions_arrow(tmp_path):
    path = tmp_path / "subs.txt"
    path.write_text("ph -> f\nth -> teh\n")
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"ph": ["f"], "th": ["teh"]}

def test_load_substitutions_colon(tmp_path):
    path = tmp_path / "subs.txt"
    path.write_text("ph: f\nth: teh\n")
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"ph": ["f"], "th": ["teh"]}

def test_load_substitutions_toml_table(tmp_path):
    path = tmp_path / "subs.toml"
    path.write_text('ph = "f"\nth = ["teh", "t"]\n')
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"ph": ["f"], "th": ["teh", "t"]}

def test_load_substitutions_markdown_table(tmp_path):
    path = tmp_path / "subs.md"
    content = """
| typo | correction |
|------|------------|
| teh  | the        |
| wrod | word       |
"""
    path.write_text(content)
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"the": ["teh"], "word": ["wrod"]}

def test_load_substitutions_typostats_table(tmp_path):
    path = tmp_path / "subs.txt"
    content = """
  Typo │ Correction │ Count │      % │ Visual
───────┼────────────┼───────┼────────┼────────
  o    │ e          │     5 │  15.0% │ ▊
  teh  │ the        │    10 │  30.0% │ █
"""
    path.write_text(content)
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"e": ["o"], "the": ["teh"]}

def test_load_substitutions_latin1_fallback(tmp_path):
    path = tmp_path / "subs.txt"
    content = "é -> e\n"
    path.write_bytes(content.encode("latin-1"))
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"é": ["e"]}
