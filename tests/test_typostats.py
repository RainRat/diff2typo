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


def test_generate_report_yaml_format(capsys):
    counts = {('c', 'x'): 2, ('a', 'b'): 1}
    typostats.generate_report(counts, output_format='yaml')
    output_lines = capsys.readouterr().out.splitlines()
    assert output_lines == ['  a:', '  - "b"', '  c:', '  - "x"']


def test_generate_report_sort_by_typo(capsys):
    counts = {('b', 'z'): 1, ('a', 'y'): 2, ('a', 'x'): 3}
    typostats.generate_report(counts, sort_by='typo', output_format='arrow')
    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert lines[1:] == ['a -> x: 3', 'a -> y: 2', 'b -> z: 1']


def test_generate_report_sort_by_correct(capsys):
    counts = {('b', 'z'): 1, ('a', 'y'): 2, ('c', 'x'): 3}
    typostats.generate_report(counts, sort_by='correct', output_format='arrow')
    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert lines[1:] == ['a -> y: 2', 'b -> z: 1', 'c -> x: 3']


def test_main_file_not_found(monkeypatch, tmp_path):
    output_file = tmp_path / 'out.txt'

    monkeypatch.setattr(
        sys,
        'argv',
        ['typostats.py', str(tmp_path / 'missing.csv'), '--output', str(output_file)],
    )

    with pytest.raises(SystemExit):
        typostats.main()


def test_main_encoding_fallback(monkeypatch, tmp_path):
    input_file = tmp_path / 'latin1.csv'
    input_file.write_text('fóo,foo\n', encoding='latin1')
    output_file = tmp_path / 'report.txt'

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'typostats.py',
            str(input_file),
            '--output',
            str(output_file),
            '--format',
            'arrow',
        ],
    )

    typostats.main()

    assert output_file.exists()
