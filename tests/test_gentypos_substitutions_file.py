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

def test_load_substitutions_txt_arrow(tmp_path):
    path = tmp_path / "subs.txt"
    path.write_text("e -> a\no -> i\n")
    # Default is typo -> correction, meaning correct is on the right, typo is on the left.
    # So mapping correct -> [typo] will be a -> [e], i -> [o]
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"a": ["e"], "i": ["o"]}

def test_load_substitutions_txt_equal(tmp_path):
    path = tmp_path / "subs.txt"
    path.write_text('e = "a"\no = "i"\n')
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"a": ["e"], "i": ["o"]}

def test_load_substitutions_txt_colon(tmp_path):
    path = tmp_path / "subs.txt"
    path.write_text("e: a\no: i\n")
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"a": ["e"], "i": ["o"]}

def test_load_substitutions_txt_csv_fallback(tmp_path):
    path = tmp_path / "subs.txt"
    path.write_text("e,a\no,i\n")
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"a": ["e"], "i": ["o"]}

def test_load_substitutions_txt_header_correct_left(tmp_path):
    path = tmp_path / "subs.txt"
    path.write_text("correct -> typo\na -> e\ni -> o\n")
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"a": ["e"], "i": ["o"]}

def test_load_substitutions_txt_header_typo_left(tmp_path):
    path = tmp_path / "subs.txt"
    path.write_text("typo -> correction\ne -> a\no -> i\n")
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"a": ["e"], "i": ["o"]}

def test_load_substitutions_md_table(tmp_path):
    path = tmp_path / "subs.md"
    content = (
        "| Typo | Correction |\n"
        "| :--- | :--- |\n"
        "| e | a |\n"
        "| o | i |\n"
    )
    path.write_text(content)
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"a": ["e"], "i": ["o"]}

def test_load_substitutions_md_table_correct_left(tmp_path):
    path = tmp_path / "subs.md"
    content = (
        "| Correct | Typo |\n"
        "| :--- | :--- |\n"
        "| a | e |\n"
        "| i | o |\n"
    )
    path.write_text(content)
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"a": ["e"], "i": ["o"]}

def test_load_substitutions_toml_plain(tmp_path):
    path = tmp_path / "subs.toml"
    path.write_text('ph = ["f", "v"]\nsh = "s"\n')
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"ph": ["f", "v"], "sh": ["s"]}

def test_load_substitutions_toml_replacements(tmp_path):
    path = tmp_path / "subs.toml"
    content = """
    [[replacements]]
    typo = "e"
    correct = "a"

    [[replacements]]
    typo = "o"
    correction = "i"
    """
    path.write_text(content)
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"a": ["e"], "i": ["o"]}

def test_load_substitutions_toml_no_dependency(tmp_path, monkeypatch):
    import importlib.util
    original_find_spec = importlib.util.find_spec
    def mock_find_spec(name):
        if name == "toml":
            return None
        return original_find_spec(name)
    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)
    monkeypatch.setitem(sys.modules, "tomllib", None)
    monkeypatch.setitem(sys.modules, "toml", None)

    path = tmp_path / "subs.toml"
    path.write_text('ph = "f"')

    with pytest.raises(SystemExit):
        gentypos._load_substitutions_file(str(path))

def test_load_substitutions_txt_latin1(tmp_path):
    path = tmp_path / "subs.txt"
    # Write some characters that are valid in Latin-1 but not in UTF-8 (e.g. \xe9)
    path.write_bytes(b"e \xe9 -> a\n")
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"a": ["e \xe9"]}

def test_load_substitutions_invalid_txt(tmp_path):
    path = tmp_path / "subs.txt"
    path.write_text("invalid_line_without_separators\n")
    result = gentypos._load_substitutions_file(str(path))
    assert result == {}
