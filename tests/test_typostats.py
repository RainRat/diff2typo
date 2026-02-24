import json
import sys
from pathlib import Path

import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import typostats


def test_is_one_letter_replacement_basic():
    assert typostats.is_one_letter_replacement('tezt', 'test') == [('s', 'z')]
    assert typostats.is_one_letter_replacement('test', 'test') == []
    assert typostats.is_one_letter_replacement('abc', 'xyz') == []


def test_is_one_letter_replacement_one_to_two():
    assert typostats.is_one_letter_replacement('aa', 'a', allow_two_char=True) == [('a', 'aa')]
    assert typostats.is_one_letter_replacement('aa', 'a', allow_two_char=False) == []


def test_is_one_letter_replacement_multiple_two_char():
    # Example: cabt -> cat
    # Two possible interpretations:
    # 1. 'a' in 'cat' is replaced by 'ab' in 'cabt'
    # 2. 't' in 'cat' is replaced by 'bt' in 'cabt'
    assert typostats.is_one_letter_replacement('cabt', 'cat', allow_two_char=True) == [
        ('a', 'ab'),
        ('t', 'bt'),
    ]


def test_is_one_letter_replacement_doubled_letter():
    # For a doubled letter, we capture all valid one-to-two interpretations.
    # While 'a' -> 'aa' is the most likely, 'c' -> 'ca' and 't' -> 'at' are also technically valid.
    assert typostats.is_one_letter_replacement('caat', 'cat', allow_two_char=True) == [
        ('a', 'aa'),
        ('c', 'ca'),
        ('t', 'at'),
    ]
    assert typostats.is_one_letter_replacement('catt', 'cat', allow_two_char=True) == [
        ('a', 'at'),
        ('t', 'tt'),
    ]


def test_is_transposition():
    assert typostats.is_transposition('teh', 'the') == [('he', 'eh')]
    assert typostats.is_transposition('tehs', 'thes') == [('he', 'eh')]
    assert typostats.is_transposition('test', 'test') == []
    assert typostats.is_transposition('tset', 'test') == [('es', 'se')]
    assert typostats.is_transposition('abcde', 'abced') == [('ed', 'de')]
    # Non-adjacent transposition should return []
    assert typostats.is_transposition('ecbad', 'abcde') == []


def test_process_typos_counts_and_filtering():
    lines = [
        'tezt, test',
        'lavel, level',
        'aa, a',
        'fóo, foo',  # non-ASCII; should be skipped
    ]
    counts, _, _ = typostats.process_typos(lines, allow_two_char=True)
    assert counts == {('s', 'z'): 1, ('e', 'a'): 1, ('a', 'aa'): 1}


def test_process_typos_table_format():
    lines = [
        'tezt = "test"',
        'lavel = "level"',
    ]
    counts, _, _ = typostats.process_typos(lines, allow_two_char=False)
    assert counts == {('s', 'z'): 1, ('e', 'a'): 1}


def test_process_typos_with_transposition():
    lines = [
        'teh -> the',
        'tset -> test',
    ]
    # Without allow_transposition, these should return nothing (multi-letter diff)
    counts, _, _ = typostats.process_typos(lines, allow_two_char=False, allow_transposition=False)
    assert counts == {}

    # With allow_transposition, they should be detected
    counts, _, _ = typostats.process_typos(lines, allow_two_char=False, allow_transposition=True)
    assert counts == {('he', 'eh'): 1, ('es', 'se'): 1}


def test_generate_report_arrow(capsys):
    counts = {('s', 'z'): 3, ('e', 'a'): 1}
    typostats.generate_report(counts, min_occurrences=2, output_format='arrow', quiet=True)
    captured = capsys.readouterr().out
    # 's' should be padded with 6 spaces (max_c=7)
    assert '       s -> z    :     3' in captured
    assert 'e' not in captured


def test_generate_report_limit(capsys):
    counts = {('a', 'b'): 10, ('c', 'd'): 5, ('e', 'f'): 2}
    typostats.generate_report(counts, limit=2, output_format='arrow', quiet=True)
    captured = capsys.readouterr().out
    assert '       a -> b    :    10' in captured
    assert '       c -> d    :     5' in captured
    assert 'e' not in captured


def test_generate_report_limit_with_typo_sort(capsys):
    # Counts are such that 'z' would be last in count sort, but first in reverse typo sort?
    # Actually sort_by='typo' sorts alphabetically by typo char.
    counts = {('a', 'z'): 10, ('b', 'x'): 5, ('c', 'y'): 2}
    # Sorted by typo: ('b', 'x'), ('c', 'y'), ('a', 'z')
    typostats.generate_report(counts, limit=2, sort_by='typo', output_format='arrow', quiet=True)
    captured = capsys.readouterr().out
    assert '       b -> x    :     5' in captured
    assert '       c -> y    :     2' in captured
    assert 'a -> z' not in captured


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


def test_generate_report_csv(capsys):
    counts = {('s', 'z'): 3, ('e', 'a'): 1}
    typostats.generate_report(counts, output_format='csv')
    captured = capsys.readouterr().out
    # The csv module in Python uses \r\n for line endings by default
    expected_csv = "correct_char,typo_char,count\r\ns,z,3\r\ne,a,1\r\n"
    assert captured == expected_csv


def test_generate_report_sort_by_typo(capsys):
    counts = {('b', 'z'): 1, ('a', 'y'): 2, ('a', 'x'): 3}
    typostats.generate_report(counts, sort_by='typo', output_format='arrow')
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line]
    # Header is now on stderr, so lines contains only data
    assert '       a -> x    :     3' in lines[0]
    assert '       a -> y    :     2' in lines[1]
    assert '       b -> z    :     1' in lines[2]
    assert "LETTER REPLACEMENTS" in captured.err


def test_generate_report_sort_by_correct(capsys):
    counts = {('b', 'z'): 1, ('a', 'y'): 2, ('c', 'x'): 3}
    typostats.generate_report(counts, sort_by='correct', output_format='arrow')
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line]
    # Header is now on stderr, so lines contains only data
    assert '       a -> y    :     2' in lines[0]
    assert '       b -> z    :     1' in lines[1]
    assert '       c -> x    :     3' in lines[2]
    assert "LETTER REPLACEMENTS" in captured.err


def test_main_file_not_found(monkeypatch, tmp_path):
    output_file = tmp_path / 'out.txt'

    monkeypatch.setattr(
        sys,
        'argv',
        ['typostats.py', str(tmp_path / 'missing.csv'), '--output', str(output_file)],
    )

    # In batch processing, missing files are logged as errors but do not stop execution
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


def test_main_cli_flags(monkeypatch, tmp_path):
    input_file = tmp_path / 'input.txt'
    input_file.write_text('tezt -> test\n')
    output_file = tmp_path / 'output.txt'

    # Test kebab-case flag and quiet flag
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'typostats.py',
            str(input_file),
            '--output',
            str(output_file),
            '--allow-two-char',
            '--quiet',
        ],
    )
    typostats.main()
    assert output_file.exists()


def test_process_typos_skips_non_ascii_corrections():
    lines = [
        'tezt, tézt, test',
    ]
    counts, _, _ = typostats.process_typos(lines, allow_two_char=False)
    assert counts == {('s', 'z'): 1}
