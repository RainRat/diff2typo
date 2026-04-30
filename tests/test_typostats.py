import json
import sys
import logging
import io
import runpy
import subprocess
import importlib
from pathlib import Path
from unittest.mock import patch, mock_open
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import typostats


def test_levenshtein_distance_basic():
    assert typostats.levenshtein_distance('test', 'test') == 0
    assert typostats.levenshtein_distance('test', 'tezt') == 1
    assert typostats.levenshtein_distance('test', 'tests') == 1
    assert typostats.levenshtein_distance('tests', 'test') == 1
    assert typostats.levenshtein_distance('', 'abc') == 3
    assert typostats.levenshtein_distance('abc', '') == 3


def test_is_transposition_basic():
    assert typostats.is_transposition('teh', 'the') == [('he', 'eh')]
    assert typostats.is_transposition('tehs', 'thes') == [('he', 'eh')]
    assert typostats.is_transposition('test', 'test') == []
    assert typostats.is_transposition('tset', 'test') == [('es', 'se')]
    assert typostats.is_transposition('abcde', 'abced') == [('ed', 'de')]
    assert typostats.is_transposition('ecbad', 'abcde') == []
    assert typostats.is_transposition("abc", "ab") == []


def test_is_one_letter_replacement_basic():
    assert typostats.is_one_letter_replacement('tezt', 'test') == [('s', 'z')]
    assert typostats.is_one_letter_replacement('test', 'test') == []
    assert typostats.is_one_letter_replacement('abc', 'xyz') == []


def test_is_one_letter_replacement_one_to_two():
    assert typostats.is_one_letter_replacement('aa', 'a', allow_1to2=True) == []
    assert typostats.is_one_letter_replacement('aa', 'a', allow_1to2=True, include_deletions=True) == [('a', 'aa')]
    assert typostats.is_one_letter_replacement('rn', 'm', allow_1to2=True) == [('m', 'rn')]
    assert typostats.is_one_letter_replacement('aa', 'a', allow_1to2=False) == []


def test_is_one_letter_replacement_multiple_two_char():
    assert typostats.is_one_letter_replacement('cabt', 'cat', allow_1to2=True) == []
    assert typostats.is_one_letter_replacement('cabt', 'cat', allow_1to2=True, include_deletions=True) == [
        ('a', 'ab'),
        ('t', 'bt'),
    ]


def test_is_one_letter_replacement_doubled_letter():
    assert typostats.is_one_letter_replacement('caat', 'cat', allow_1to2=True) == []
    assert typostats.is_one_letter_replacement('caat', 'cat', allow_1to2=True, include_deletions=True) == [
        ('a', 'aa'),
        ('c', 'ca'),
        ('t', 'at'),
    ]


def test_is_one_letter_replacement_two_to_one():
    assert typostats.is_one_letter_replacement('f', 'ph', allow_2to1=True) == [('ph', 'f')]
    assert typostats.is_one_letter_replacement('a', 'aa', allow_2to1=True, include_deletions=False) == []
    assert typostats.is_one_letter_replacement('a', 'aa', allow_2to1=True, include_deletions=True) == [('aa', 'a')]


def test_process_typos_formats():
    assert typostats.process_typos(['teh -> the'])[0] == {}
    assert typostats.process_typos(['teh -> the'], allow_transposition=True)[0] == {('he', 'eh'): 1}
    assert typostats.process_typos(['tezt = "test"'])[0] == {('s', 'z'): 1}
    assert typostats.process_typos(['tezt: test'])[0] == {('s', 'z'): 1}
    assert typostats.process_typos(['tezt, test'])[0] == {('s', 'z'): 1}
    counts, _, _ = typostats.process_typos(['tezt, test, toast'])
    assert counts == {('s', 'z'): 1}
    assert typostats.process_typos(['fóo, foo'])[0] == {}
    assert typostats.process_typos(['foo, fóo'])[0] == {}
    assert typostats.process_typos(['', 'tezt -> test'])[1] == 2


def test_generate_report_formats(capsys, tmp_path):
    counts = {('s', 'z'): 3, ('e', 'a'): 1}
    typostats.generate_report(counts, output_format='arrow', quiet=True)
    assert 's │ z' in capsys.readouterr().out
    typostats.generate_report(counts, output_format='json')
    assert len(json.loads(capsys.readouterr().out)["replacements"]) == 2
    typostats.generate_report(counts, output_format='csv')
    assert "correct_char,typo_char,count" in capsys.readouterr().out
    typostats.generate_report(counts, output_format='yaml')
    assert "  s:" in capsys.readouterr().out
    out_file = tmp_path / "report.txt"
    typostats.generate_report(counts, output_file=str(out_file))
    assert "ANALYSIS SUMMARY" in out_file.read_text()


def test_generate_report_sorting_and_filtering(capsys):
    counts = {('b', 'z'): 1, ('a', 'y'): 2, ('a', 'x'): 3}
    typostats.generate_report(counts, sort_by='count', output_format='arrow', quiet=True)
    lines = [l for l in capsys.readouterr().out.splitlines() if '│' in l and 'CORRECT' not in l and '─' not in l]
    assert 'x' in lines[0] and 'y' in lines[1] and 'z' in lines[2]
    typostats.generate_report(counts, sort_by='typo', output_format='arrow', quiet=True)
    lines = [l for l in capsys.readouterr().out.splitlines() if '│' in l and 'CORRECT' not in l and '─' not in l]
    assert 'x' in lines[0] and 'y' in lines[1] and 'z' in lines[2]
    typostats.generate_report(counts, sort_by='correct', output_format='arrow', quiet=True)
    lines = [l for l in capsys.readouterr().out.splitlines() if '│' in l and 'CORRECT' not in l and '─' not in l]
    assert lines[0].strip().startswith('a') and lines[2].strip().startswith('b')
    typostats.generate_report(counts, min_occurrences=2, output_format='arrow', quiet=True)
    assert len([l for l in capsys.readouterr().out.splitlines() if '│' in l and 'CORRECT' not in l and '─' not in l]) == 2


def test_generate_report_summaries(capsys):
    typostats.generate_report({('he', 'eh'): 1}, allow_transposition=True, quiet=False)
    assert "Transpositions [T]:" in capsys.readouterr().err
    typostats.generate_report({('a', 'b'): 1}, min_occurrences=2, quiet=False)
    assert "Patterns matching criteria:" in capsys.readouterr().err
    typostats.generate_report({('a', 'b'): 2, ('c', 'd'): 1}, limit=1, quiet=False)
    assert "Showing patterns:" in capsys.readouterr().err
    counts = {('a', 'aa'): 1, ('bb', 'b'): 1, ('m', 'rn'): 1, ('ph', 'f'): 1}
    typostats.generate_report(counts, include_deletions=True, allow_1to2=True, allow_2to1=True, quiet=False)
    err = capsys.readouterr().err
    assert "Insertions [Ins]:" in err and "Deletions [Del]:" in err
    assert "1-to-2 replacements [1:2]" in err and "2-to-1 replacements [2:1]" in err


def test_generate_report_keyboard(capsys):
    counts = {('q', 'w'): 5, ('q', 'p'): 1}
    typostats.generate_report(counts, keyboard=True, quiet=False)
    captured = capsys.readouterr()
    assert "Keyboard Adjacency" in captured.err
    assert "[K]" in captured.out
    assert "[K]" not in captured.out.splitlines()[-1]


def test_generate_report_markers(capsys):
    counts = {('a', 'ab'): 1, ('bc', 'b'): 1, ('m', 'rn'): 1, ('ph', 'f'): 1, ('he', 'eh'): 1}
    typostats.generate_report(counts, all=True, quiet=True)
    out = capsys.readouterr().out
    assert all(m in out for m in ["[Ins]", "[Del]", "[1:2]", "[2:1]", "[T]"])


def test_detect_encoding_variants(caplog):
    with patch('typostats._CHARDET_AVAILABLE', False):
        assert typostats.detect_encoding("dummy.txt") is None
        assert "chardet not installed" in caplog.text
    if typostats._CHARDET_AVAILABLE:
        with patch('builtins.open', mock_open(read_data=b'abc')), \
             patch('typostats.chardet.detect') as mock_detect:
            mock_detect.return_value = {'encoding': 'utf-8', 'confidence': 0.9}
            assert typostats.detect_encoding("dummy.txt") == 'utf-8'
            mock_detect.return_value = {'encoding': 'utf-8', 'confidence': 0.4}
            assert typostats.detect_encoding("dummy.txt") is None


def test_load_lines_from_file_variants(monkeypatch, tmp_path):
    monkeypatch.setattr(sys.stdin, 'readlines', lambda: ["line1\n"])
    assert typostats.load_lines_from_file('-') == ["line1\n"]
    assert typostats.load_lines_from_file(str(tmp_path / "nonexistent")) is None
    mock_files = {'dummy.txt': b'\xff'}
    def mocked_open(file, mode='r', encoding=None, **kwargs):
        if 'b' in mode: return io.BytesIO(mock_files['dummy.txt'])
        if encoding == 'utf-8': raise UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')
        if encoding == 'latin1': return io.StringIO("\xff")
        raise UnicodeDecodeError('other', b'', 0, 1, 'invalid')
    with patch('builtins.open', side_effect=mocked_open), \
         patch('typostats.detect_encoding', return_value=None):
        assert typostats.load_lines_from_file('dummy.txt') == ["\xff"]


def test_main_cli_functionality():
    with patch('sys.argv', ['typostats.py', '--help']):
        with pytest.raises(SystemExit): typostats.main()
    with patch('sys.argv', ['typostats.py', 'input.txt', '-a', '-q']), \
         patch('typostats.load_lines_from_file', return_value=["teh -> the"]), \
         patch('typostats.generate_report') as mock_report:
        typostats.main()
        assert mock_report.call_args[1]['keyboard'] is True and mock_report.call_args[1]['quiet'] is True
    with patch('sys.argv', ['typostats.py', 'input.txt', '--allow-two-char']), \
         patch('typostats.load_lines_from_file', return_value=["m -> rn"]), \
         patch('typostats.process_typos', return_value=({}, 0, 0)) as mock_process, \
         patch('typostats.generate_report'):
        typostats.main()
        assert mock_process.call_args[1]['allow_1to2'] is True and mock_process.call_args[1]['allow_2to1'] is True


def test_typostats_subprocess_all(tmp_path):
    typos_file = tmp_path / "typos.txt"
    typos_file.write_text("teh -> the\nrecieve -> receive\nm -> rn\nph -> f\nor -> o\na -> aa\n", encoding="utf-8")
    result = subprocess.run([sys.executable, "typostats.py", str(typos_file), "-a"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Enabled features:" in result.stderr
    assert "he" in result.stdout and "rn" in result.stdout


def test_tqdm_unavailable_fallback():
    initial_tqdm = typostats._TQDM_AVAILABLE
    try:
        with patch.dict(sys.modules, {'tqdm': None}):
            importlib.reload(typostats)
            assert typostats._TQDM_AVAILABLE is False
    finally:
        importlib.reload(typostats)
    assert typostats._TQDM_AVAILABLE == initial_tqdm


def test_minimal_formatter():
    formatter = typostats.MinimalFormatter()
    assert formatter.format(logging.LogRecord('n', logging.INFO, 'p', 1, 'msg', None, None)) == 'msg'
    with patch('typostats.sys.stderr.isatty', return_value=False):
        assert formatter.format(logging.LogRecord('n', logging.WARNING, 'p', 1, 'msg', None, None)) == 'WARNING: msg'


def test_format_analysis_summary_branches():
    # Retention bar branches
    # 100% retention
    report = "\n".join(typostats._format_analysis_summary(10, ["a"] * 10))
    assert "Retention rate:" in report and "100.0%" in report
    assert "████████████████████" in report

    # 0% retention
    report = "\n".join(typostats._format_analysis_summary(10, []))
    assert "Retention rate:" in report and "0.0%" in report
    assert "No items passed" in report

    # Non-hashable unique items
    report = "\n".join(typostats._format_analysis_summary(2, [["a"], ["a"]], item_label="list"))
    assert "Unique items:" in report or "Unique lists:" in report

    # Shortest/Longest with format_item
    report = "\n".join(typostats._format_analysis_summary(2, [("a", "bc"), ("def", "g")]))
    assert "Shortest item:" in report or "Shortest replacement:" in report
    assert "Longest item:" in report or "Longest replacement:" in report
    report = "\n".join(typostats._format_analysis_summary(10, ["a"], item_label="word"))
    assert "Total words encountered:" in report
