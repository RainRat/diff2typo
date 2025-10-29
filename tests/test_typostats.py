import json
import sys
from pathlib import Path

import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import typostats


def test_is_one_letter_replacement_basic():
    assert typostats.is_one_letter_replacement('tezt', 'test') == ('s', 'z')
    assert typostats.is_one_letter_replacement('test', 'test') is None
    assert typostats.is_one_letter_replacement('abc', 'xyz') is None


def test_is_one_letter_replacement_one_to_two():
    assert typostats.is_one_letter_replacement('aa', 'a', allow_two_char=True) == ('a', 'aa')
    assert typostats.is_one_letter_replacement('aa', 'a', allow_two_char=False) is None


def test_process_typos_counts_and_filtering():
    lines = [
        'tezt, test',
        'lavel, level',
        'aa, a',
        'fóo, foo',  # non-ASCII; should be skipped
    ]
    counts = typostats.process_typos(lines, allow_two_char=True)
    assert counts == {('s', 'z'): 1, ('e', 'a'): 1, ('a', 'aa'): 1}


def test_generate_report_arrow(capsys):
    counts = {('s', 'z'): 3, ('e', 'a'): 1}
    typostats.generate_report(counts, min_occurrences=2, output_format='arrow')
    captured = capsys.readouterr().out
    assert 's -> z: 3' in captured
    assert 'e -> a' not in captured


def test_generate_report_json(capsys):
    counts = {('s', 'z'): 3, ('e', 'a'): 1}
    typostats.generate_report(counts, output_format='json')
    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert data["replacements"] == [
        {"correct": "s", "typo": "z", "count": 3},
        {"correct": "e", "typo": "a", "count": 1},
    ]
