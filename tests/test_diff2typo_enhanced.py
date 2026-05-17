import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import diff2typo

def test_main_min_count(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    # 3 occurrences of teh -> the, 1 occurrence of recieve -> receive
    diff_text = (
        '--- a/file1\n+++ b/file1\n@@\n-teh\n+the\n'
        '--- a/file2\n+++ b/file2\n@@\n-teh\n+the\n'
        '--- a/file3\n+++ b/file3\n@@\n-teh\n+the\n'
        '--- a/file4\n+++ b/file4\n@@\n-recieve\n+receive\n'
    )

    diff_file = tmp_path / 'diff.txt'
    diff_file.write_text(diff_text)

    output_file = tmp_path / 'output.txt'

    # Mocking external tools and files to avoid issues
    monkeypatch.setattr(diff2typo, 'read_words_mapping', lambda *a, **k: {})
    monkeypatch.setattr(diff2typo, 'read_allowed_words', lambda *a, **k: set())
    monkeypatch.setattr(diff2typo, 'filter_known_typos', lambda c, *a, **k: c)

    # Run with --min-count 2
    monkeypatch.setattr(sys, 'argv', [
        'diff2typo.py',
        '--input', str(diff_file),
        '--output', str(output_file),
        '--min-count', '2',
        '--quiet'
    ])

    diff2typo.main()

    results = output_file.read_text().strip().splitlines()
    assert results == ['teh -> the']
    assert 'recieve -> receive' not in results

def test_main_sort_count(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    # teh: 3, recieve: 2, eror: 1
    # Use words longer than default min-length=2
    diff_text = (
        '-teh\n+the\n' * 3 +
        '-recieve\n+receive\n' * 2 +
        '-eror\n+error\n' * 1
    )

    diff_file = tmp_path / 'diff.txt'
    diff_file.write_text(diff_text)
    output_file = tmp_path / 'output.txt'

    # Mocking as before
    monkeypatch.setattr(diff2typo, 'read_words_mapping', lambda *a, **k: {})
    monkeypatch.setattr(diff2typo, 'read_allowed_words', lambda *a, **k: set())
    monkeypatch.setattr(diff2typo, 'filter_known_typos', lambda c, *a, **k: c)

    # Sort by count
    monkeypatch.setattr(sys, 'argv', [
        'diff2typo.py',
        '--input', str(diff_file),
        '--output', str(output_file),
        '--sort', 'count',
        '--quiet'
    ])

    diff2typo.main()
    results = output_file.read_text().strip().splitlines()
    assert results == ['teh -> the', 'recieve -> receive', 'eror -> error']

    # Sort by alpha (default)
    monkeypatch.setattr(sys, 'argv', [
        'diff2typo.py',
        '--input', str(diff_file),
        '--output', str(output_file),
        '--sort', 'alpha',
        '--quiet'
    ])

    diff2typo.main()
    results = output_file.read_text().strip().splitlines()
    assert results == ['eror -> error', 'recieve -> receive', 'teh -> the']

def test_main_limit(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    # Need to meet min-length=2 by default
    diff_text = '-aa\n+bb\n-cc\n+dd\n-ee\n+ff\n'

    diff_file = tmp_path / 'diff.txt'
    diff_file.write_text(diff_text)
    output_file = tmp_path / 'output.txt'

    # Mocking
    monkeypatch.setattr(diff2typo, 'read_words_mapping', lambda *a, **k: {})
    monkeypatch.setattr(diff2typo, 'read_allowed_words', lambda *a, **k: set())
    monkeypatch.setattr(diff2typo, 'filter_known_typos', lambda c, *a, **k: c)

    monkeypatch.setattr(sys, 'argv', [
        'diff2typo.py',
        '--input', str(diff_file),
        '--output', str(output_file),
        '--limit', '2',
        '--quiet'
    ])

    diff2typo.main()
    results = output_file.read_text().strip().splitlines()
    assert len(results) == 2
