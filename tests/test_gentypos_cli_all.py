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

def test_gentypos_cli_all_long_flag(capsys, empty_config_file):
    test_args = [
        "gentypos.py",
        "word",
        "-c", empty_config_file,
        "--all",
        "--no-filter",
        "-f", "arrow"
    ]
    with patch.object(sys, 'argv', test_args):
        gentypos.main()

    captured = capsys.readouterr()
    stdout_lines = captured.out.splitlines()
    assert len(stdout_lines) > 0

    # 1. Deletion: 'wrd', 'wod', 'ord', 'wor'
    assert any("wrd -> word" in line or "wod -> word" in line for line in stdout_lines)

    # 2. Transposition: 'wrod', 'owrd', 'wodr'
    assert any("wrod -> word" in line for line in stdout_lines)

    # 3. Replacement (keyboard / substitution): 'wprd' (o -> p), 'wsrd' (o -> s)
    assert any("wprd -> word" in line for line in stdout_lines)

    # 4. Duplication: 'woord', 'wword', 'worrd', 'wordd'
    assert any("woord -> word" in line or "wword -> word" in line for line in stdout_lines)

def test_gentypos_cli_all_short_flag(capsys, empty_config_file):
    test_args = [
        "gentypos.py",
        "word",
        "-c", empty_config_file,
        "-A",
        "--no-filter",
        "-f", "arrow"
    ]
    with patch.object(sys, 'argv', test_args):
        gentypos.main()

    captured = capsys.readouterr()
    stdout_lines = captured.out.splitlines()
    assert len(stdout_lines) > 0

    # 1. Deletion
    assert any("wrd -> word" in line or "wod -> word" in line for line in stdout_lines)

    # 2. Transposition
    assert any("wrod -> word" in line for line in stdout_lines)

    # 3. Replacement
    assert any("wprd -> word" in line for line in stdout_lines)

    # 4. Duplication
    assert any("woord -> word" in line or "wword -> word" in line for line in stdout_lines)
