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
