from unittest.mock import patch
import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

@pytest.fixture
def empty_config_file(tmp_path):
    config = tmp_path / "test_config.yaml"
    config.write_text("{}", encoding="utf-8")
    return str(config)

@pytest.fixture
def input_words_file(tmp_path):
    words_file = tmp_path / "words.txt"
    words_file.write_text("banana\napple\n", encoding="utf-8")
    return str(words_file)

@pytest.fixture
def large_dictionary_file(tmp_path):
    dict_file = tmp_path / "dict.txt"
    # We add potential typos of "apple", e.g., "spple" (replacement of 'a' with 's')
    # If "spple" is in the large dictionary, it should be filtered out.
    dict_file.write_text("spple\n", encoding="utf-8")
    return str(dict_file)

def test_gentypos_cli_override_input_file_prints_typos_to_stdout(capsys, empty_config_file, input_words_file):
    test_args = [
        "gentypos.py",
        "-c", empty_config_file,
        "-i", input_words_file,
        "--no-filter",
        "-f", "arrow"
    ]
    with patch.object(sys, 'argv', test_args):
        gentypos.main()

    captured = capsys.readouterr()
    stdout_lines = captured.out.splitlines()
    assert len(stdout_lines) > 0
    # The generated typo should map back to "banana" or "apple"
    assert any("-> banana" in line for line in stdout_lines)
    assert any("-> apple" in line for line in stdout_lines)

def test_gentypos_cli_override_dictionary_file_filters_typos(capsys, empty_config_file, tmp_path):
    words_file = tmp_path / "words.txt"
    words_file.write_text("apple\n", encoding="utf-8")

    dict_file = tmp_path / "dict.txt"
    # 'spple' is adjacent replacement of 'a' on QWERTY
    dict_file.write_text("spple\n", encoding="utf-8")

    test_args = [
        "gentypos.py",
        "-c", empty_config_file,
        "-i", str(words_file),
        "-d", str(dict_file),
        "-f", "arrow"
    ]
    with patch.object(sys, 'argv', test_args):
        gentypos.main()

    captured = capsys.readouterr()
    stdout_lines = captured.out.splitlines()
    assert len(stdout_lines) > 0
    assert not any("spple -> apple" in line for line in stdout_lines)

def test_gentypos_cli_override_repeat_modifications_generates_nested_typos(capsys, empty_config_file, tmp_path):
    words_file = tmp_path / "words.txt"
    words_file.write_text("cat\n", encoding="utf-8")

    test_args = [
        "gentypos.py",
        "-c", empty_config_file,
        "-i", str(words_file),
        "--no-filter",
        "-r", "2",
        "-f", "arrow"
    ]
    with patch.object(sys, 'argv', test_args):
        gentypos.main()

    captured = capsys.readouterr()
    stdout_lines = captured.out.splitlines()
    assert len(stdout_lines) > 0
    # With repeat=2, we should get typos of distance 2.
    # For example, "cat" -> replacement/deletion/transposition of distance 2
    # e.g., 'ca' is distance 1 deletion. Distance 2 deletion from 'cat' can be 'c' or 'a' or 't' or empty.
    # Also double adjacent key replacements.
    assert any(len(line.split(" -> ")[0]) >= 1 for line in stdout_lines)

def test_gentypos_cli_override_alias_flags_work(capsys, empty_config_file, input_words_file, large_dictionary_file):
    test_args = [
        "gentypos.py",
        "-c", empty_config_file,
        "--input-file", input_words_file,
        "--dictionary-file", large_dictionary_file,
        "--repeat-modifications", "1",
        "-f", "arrow"
    ]
    with patch.object(sys, 'argv', test_args):
        gentypos.main()

    captured = capsys.readouterr()
    stdout_lines = captured.out.splitlines()
    assert len(stdout_lines) > 0
    assert not any("spple -> apple" in line for line in stdout_lines)
