from pathlib import Path
import sys
from unittest.mock import patch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos


def test_cli_transposition_distance_long_flag(capsys):
    """Test that --transposition-distance sets the transposition distance correctly in CLI mode."""
    test_args = ["gentypos.py", "abcdef", "-t", "--transposition-distance", "2", "--no-filter"]
    with patch.object(sys, "argv", test_args):
        gentypos.main()

    captured = capsys.readouterr()
    # 'abcdef' with distance 2 yields transpositions: cbadef, adcbef, abedcf, abcfed
    assert "cbadef -> abcdef" in captured.out
    assert "adcbef -> abcdef" in captured.out
    assert "abedcf -> abcdef" in captured.out
    assert "abcfed -> abcdef" in captured.out


def test_cli_transposition_distance_short_flag(capsys):
    """Test that -T short flag sets the transposition distance correctly in CLI mode."""
    test_args = ["gentypos.py", "abcdef", "-t", "-T", "2", "--no-filter"]
    with patch.object(sys, "argv", test_args):
        gentypos.main()

    captured = capsys.readouterr()
    assert "cbadef -> abcdef" in captured.out
    assert "adcbef -> abcdef" in captured.out


def test_cli_transposition_distance_overrides_config(tmp_path, capsys):
    """Test that CLI transposition distance flag overrides distance specified in YAML config."""
    config_file = tmp_path / "custom_config.yaml"
    config_file.write_text(
        "transposition_options:\n"
        "  distance: 1\n"
        "typo_types:\n"
        "  deletion: false\n"
        "  transposition: true\n"
        "  replacement: false\n"
        "  duplication: false\n"
    )

    test_args = ["gentypos.py", "abcdef", "-c", str(config_file), "-T", "2", "--no-filter"]
    with patch.object(sys, "argv", test_args):
        gentypos.main()

    captured = capsys.readouterr()
    assert "cbadef -> abcdef" in captured.out


def test_cli_transposition_distance_null_config_override(tmp_path, capsys):
    """Test that CLI transposition distance handles None/null transposition_options gracefully."""
    config_file = tmp_path / "null_config.yaml"
    config_file.write_text(
        "transposition_options: null\n"
        "word_length: null\n"
        "typo_types:\n"
        "  transposition: true\n"
    )

    test_args = ["gentypos.py", "abcdef", "-c", str(config_file), "-T", "2", "-m", "3", "-M", "10", "--no-filter"]
    with patch.object(sys, "argv", test_args):
        gentypos.main()

    captured = capsys.readouterr()
    assert "cbadef -> abcdef" in captured.out
