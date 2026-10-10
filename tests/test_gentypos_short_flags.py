from pathlib import Path
import sys
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

def test_gentypos_deletion_short_flag(monkeypatch, capsys):
    test_args = ["gentypos.py", "word", "-D", "--no-filter", "-f", "arrow", "-q"]
    monkeypatch.setattr(sys, "argv", test_args)

    gentypos.main()

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]

    # 'word' -> deletions: 'ord', 'wod', 'wrd' (note: 'wor' is skipped since trailing 'd' deletion is ignored in gentypos)
    expected_typos = {"ord -> word", "wod -> word", "wrd -> word"}
    assert set(lines) == expected_typos


def test_gentypos_duplication_short_flag(monkeypatch, capsys):
    test_args = ["gentypos.py", "test", "-u", "--no-filter", "-f", "arrow", "-q"]
    monkeypatch.setattr(sys, "argv", test_args)

    gentypos.main()

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]

    # 'test' -> duplications: 'ttest', 'teest', 'tesst', 'testt'
    expected_typos = {"ttest -> test", "teest -> test", "tesst -> test", "testt -> test"}
    assert set(lines) == expected_typos


def test_gentypos_max_length_short_flag(monkeypatch, capsys):
    test_args = ["gentypos.py", "cat", "elephant", "-M", "4", "-D", "--no-filter", "-f", "arrow", "-q"]
    monkeypatch.setattr(sys, "argv", test_args)

    gentypos.main()

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]

    # 'elephant' (len 8) should be skipped due to -M 4; only 'cat' (len 3) processed
    expected_typos = {"at -> cat", "ct -> cat", "ca -> cat"}
    assert set(lines) == expected_typos


def test_gentypos_no_filter_short_flag(monkeypatch, capsys):
    test_args = ["gentypos.py", "test", "-u", "-N", "-f", "arrow", "-q"]
    monkeypatch.setattr(sys, "argv", test_args)

    gentypos.main()

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]

    # 'test' -> duplications with -N (no-filter): 'ttest', 'teest', 'tesst', 'testt'
    expected_typos = {"ttest -> test", "teest -> test", "tesst -> test", "testt -> test"}
    assert set(lines) == expected_typos


def test_gentypos_dry_run_short_flag(monkeypatch, caplog):
    test_args = ["gentypos.py", "hello", "-n"]
    monkeypatch.setattr(sys, "argv", test_args)

    with caplog.at_level("INFO"):
        with pytest.raises(SystemExit) as exc_info:
            gentypos.main()

    assert exc_info.value.code == 0
    assert "--- GENTYPOS DRY RUN ---" in caplog.text


def test_gentypos_plural_option_aliases(monkeypatch, capsys):
    # Test --deletions
    monkeypatch.setattr(sys, "argv", ["gentypos.py", "word", "--deletions", "--no-filter", "-f", "arrow", "-q"])
    gentypos.main()
    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    assert set(lines) == {"ord -> word", "wod -> word", "wrd -> word"}

    # Test --duplications
    monkeypatch.setattr(sys, "argv", ["gentypos.py", "test", "--duplications", "--no-filter", "-f", "arrow", "-q"])
    gentypos.main()
    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    assert set(lines) == {"ttest -> test", "teest -> test", "tesst -> test", "testt -> test"}

    # Test --transpositions
    monkeypatch.setattr(sys, "argv", ["gentypos.py", "word", "--transpositions", "--no-filter", "-f", "arrow", "-q"])
    gentypos.main()
    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    assert set(lines) == {"owrd -> word", "wrod -> word", "wodr -> word"}

    # Test --replacements
    monkeypatch.setattr(sys, "argv", ["gentypos.py", "a", "--replacements", "--no-filter", "-f", "arrow", "-q"])
    gentypos.main()
    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    assert len(lines) > 0
    assert all("-> a" in line for line in lines)


def test_gentypos_config_uppercase_short_flag(tmp_path, monkeypatch, capsys):
    cfg_file = tmp_path / "custom_gentypos.yaml"
    cfg_file.write_text(
        "output_format: arrow\n"
        "word_length:\n"
        "  min_length: 0\n"
        "typo_types:\n"
        "  deletion: true\n"
        "  transposition: false\n"
        "  replacement: false\n"
        "  duplication: false\n",
        encoding="utf-8",
    )

    test_args = ["gentypos.py", "test", "-C", str(cfg_file), "--no-filter", "-q"]
    monkeypatch.setattr(sys, "argv", test_args)

    gentypos.main()

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]

    # 'test' -> deletions: 'est', 'tst', 'tet', 'tes'
    expected_typos = {"est -> test", "tst -> test", "tet -> test", "tes -> test"}
    assert set(lines) == expected_typos


def test_gentypos_sort_short_flag(monkeypatch, capsys):
    # Test -S typo (alphabetical by typo)
    test_args_typo = ["gentypos.py", "apple", "banana", "-D", "-N", "-f", "arrow", "-S", "typo", "-q"]
    monkeypatch.setattr(sys, "argv", test_args_typo)
    gentypos.main()
    captured = capsys.readouterr()
    lines_typo = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    typos_only = [line.split(" -> ")[0] for line in lines_typo]
    assert typos_only == sorted(typos_only)

    # Test -S correct (alphabetical by correction word)
    test_args_correct = ["gentypos.py", "apple", "banana", "-D", "-N", "-f", "arrow", "-S", "correct", "-q"]
    monkeypatch.setattr(sys, "argv", test_args_correct)
    gentypos.main()
    captured = capsys.readouterr()
    lines_correct = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]
    corrections_only = [line.split(" -> ")[1] for line in lines_correct]
    assert corrections_only == sorted(corrections_only)


def test_gentypos_input_uppercase_short_flag(tmp_path, monkeypatch, capsys):
    input_file = tmp_path / "words.txt"
    input_file.write_text("banana\n", encoding="utf-8")

    test_args = ["gentypos.py", "-I", str(input_file), "-D", "-N", "-f", "arrow", "-q"]
    monkeypatch.setattr(sys, "argv", test_args)

    gentypos.main()

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines() if line.strip()]

    # Deletions for 'banana': 'anana', 'bnana', 'baana', 'banna', 'banaa', 'banan'
    expected_typos = {
        "anana -> banana",
        "bnana -> banana",
        "baana -> banana",
        "banna -> banana",
        "banaa -> banana",
        "banan -> banana",
    }
    assert set(lines) == expected_typos
