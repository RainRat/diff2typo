import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parents[1]))
import diff2typo

def test_process_audit_typos():
    args = SimpleNamespace(output_format='arrow')
    valid_words = {'apple', 'banana'}
    allowed_words = {'aple'}
    candidates = ['apple -> aple', 'apple -> aplle', 'banana -> bananaa']

    # Regression: apple -> aplle (aplle is not in valid_words or allowed_words)
    # Regression: banana -> bananaa (bananaa is not in valid_words or allowed_words)
    # Not a regression: apple -> aple (aple is in allowed_words)

    result = diff2typo.process_audit_typos(candidates, args, valid_words, allowed_words)
    assert 'apple -> aplle' in result
    assert 'banana -> bananaa' in result
    assert 'apple -> aple' not in result

def test_audit_mode_integration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    diff_file = tmp_path / 'diff.txt'
    # Case 1: Regression (apple -> aplle)
    # Case 2: Fix (aple -> apple) - should be ignored in audit mode
    diff_file.write_text('--- a/file\n+++ b/file\n@@\n-apple\n+aplle\n@@\n-aple\n+apple\n')

    dictionary_file = tmp_path / 'words.csv'
    dictionary_file.write_text('apple\nbanana\n')

    allowed_file = tmp_path / 'allowed.csv'
    allowed_file.write_text('aple\n')

    output_file = tmp_path / 'output.txt'

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'diff2typo.py',
            '--input',
            str(diff_file),
            '--output',
            str(output_file),
            '--dictionary',
            str(dictionary_file),
            '--allowed',
            str(allowed_file),
            '--mode',
            'audit',
            '--quiet',
        ],
    )

    diff2typo.main()

    results = output_file.read_text().strip().splitlines()
    assert results == ['apple -> aplle']
