import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

def test_detect_format_from_extension_various():
    allowed = ['arrow', 'csv', 'table', 'list']
    default = 'arrow'

    assert gentypos._detect_format_from_extension("filename", allowed, default) == default
    assert gentypos._detect_format_from_extension("file.txt", allowed, default) == "arrow"
    assert gentypos._detect_format_from_extension("file.csv", allowed, default) == "csv"
    assert gentypos._detect_format_from_extension("file.table", allowed, default) == "table"
    assert gentypos._detect_format_from_extension("file.toml", allowed, default) == "table"
    assert gentypos._detect_format_from_extension("file.list", allowed, default) == "list"
    assert gentypos._detect_format_from_extension("file.arrow", allowed, default) == "arrow"
    assert gentypos._detect_format_from_extension("file.unknown", allowed, default) == default
    assert gentypos._detect_format_from_extension("file.csv", ['arrow'], default) == default

def test_load_substitutions_correct_typo_header(tmp_path):
    path = tmp_path / "subs.csv"
    path.write_text("correct,typo\na,e\ni,o\n")
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"a": ["e"], "i": ["o"]}

def test_load_substitutions_typo_correct_header(tmp_path):
    path = tmp_path / "subs.csv"
    path.write_text("typo,correct\ne,a\no,i\n")
    result = gentypos._load_substitutions_file(str(path))
    assert result == {"a": ["e"], "i": ["o"]}
