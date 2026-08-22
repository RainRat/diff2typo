import sys
import pytest
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
