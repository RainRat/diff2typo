import sys
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

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

def test_main_extension_detection_csv(tmp_path):
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("{}", encoding="utf-8")
    output_file = tmp_path / "test.csv"
    test_args = [
        "gentypos.py",
        "-c", str(config_file),
        "hello",
        "--no-filter",
        "-o", str(output_file)
    ]
    with patch.object(sys, 'argv', test_args):
        try:
            gentypos.main()
        except SystemExit:
            pass
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "->" not in content

def test_main_extension_detection_unknown(tmp_path):
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("{}", encoding="utf-8")
    output_file = tmp_path / "test.unknown"
    test_args = [
        "gentypos.py",
        "-c", str(config_file),
        "hello",
        "--no-filter",
        "-o", str(output_file)
    ]
    with patch.object(sys, 'argv', test_args):
        try:
            gentypos.main()
        except SystemExit:
            pass
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "->" in content or not content

def test_main_extension_detection_no_ext(tmp_path):
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("{}", encoding="utf-8")
    output_file = tmp_path / "testfile"
    test_args = [
        "gentypos.py",
        "-c", str(config_file),
        "hello",
        "--no-filter",
        "-o", str(output_file)
    ]
    with patch.object(sys, 'argv', test_args):
        try:
            gentypos.main()
        except SystemExit:
            pass
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "->" in content or not content
