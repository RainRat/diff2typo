import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import diff2typo
from diff2typo import find_typos, process_diff_block, _compare_word_lists


def test_compare_word_lists_max_length():
    before = ["shortword", "veryveryverylongword"]
    after = ["shortwrd", "veryveryverylongwrd"]

    # Without max_length constraint
    results = _compare_word_lists(before, after, min_length=2, max_length=None)
    assert "shortword -> shortwrd" in results
    assert "veryveryverylongword -> veryveryverylongwrd" in results

    # With max_length constraint (10)
    results_filtered = _compare_word_lists(before, after, min_length=2, max_length=10)
    assert "shortword -> shortwrd" in results_filtered
    assert "veryveryverylongword -> veryveryverylongwrd" not in results_filtered


def test_find_typos_max_length():
    diff_text = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
- shortword veryveryverylongword
+ shortwrd veryveryverylongwrd
"""
    results_all = find_typos(diff_text, min_length=2, max_length=None)
    assert "shortword -> shortwrd" in results_all
    assert "veryveryverylongword -> veryveryverylongwrd" in results_all

    results_limited = find_typos(diff_text, min_length=2, max_length=12)
    assert "shortword -> shortwrd" in results_limited
    assert "veryveryverylongword -> veryveryverylongwrd" not in results_limited


def test_diff2typo_cli_max_length(monkeypatch, capsys):
    diff_content = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
- shortword veryveryverylongword
+ shortwrd veryveryverylongwrd
"""
    monkeypatch.setattr(diff2typo, '_read_diff_sources', lambda _: diff_content)
    monkeypatch.setattr(diff2typo, 'read_words_mapping', lambda *a, **kw: {})
    monkeypatch.setattr(diff2typo, 'read_allowed_words', lambda *a, **kw: set())
    monkeypatch.setattr(diff2typo, 'filter_known_typos', lambda candidates, **kw: candidates)

    monkeypatch.setattr(sys, 'argv', ['diff2typo.py', 'test.diff', '--max-length', '12', '-q'])

    diff2typo.main()

    captured = capsys.readouterr()
    assert "shortword -> shortwrd" in captured.out
    assert "veryveryverylongword" not in captured.out
