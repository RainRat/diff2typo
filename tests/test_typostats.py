import json
import csv
import sys
import logging
import io
import subprocess
import importlib
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
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


def test_levenshtein_distance_extra():
    assert typostats.levenshtein_distance('a', 'abc') == 2
    assert typostats.levenshtein_distance('abc', 'a') == 2


def test_is_transposition_basic():
    assert typostats.is_transposition('teh', 'the') == [('he', 'eh')]
    assert typostats.is_transposition('tehs', 'thes') == [('he', 'eh')]
    assert typostats.is_transposition('test', 'test') == []
    assert typostats.is_transposition('tset', 'test') == [('es', 'se')]
    assert typostats.is_transposition('abcde', 'abcle') == [] # Not a transposition
    assert typostats.is_transposition('abcde', 'abced') == [('ed', 'de')]
    assert typostats.is_transposition('ecbad', 'abcde') == []
    assert typostats.is_transposition("abc", "ab") == []
    # Case with exactly 2 differences but not adjacent
    assert typostats.is_transposition('axcye', 'abcde') == []
    # Case with exactly 2 differences adjacent but not a swap
    assert typostats.is_transposition('abxye', 'abcde') == []


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
    # Returns only the first valid interpretation to avoid double-counting
    assert typostats.is_one_letter_replacement('cabt', 'cat', allow_1to2=True, include_deletions=True) == [
        ('a', 'ab'),
    ]


def test_is_one_letter_replacement_doubled_letter():
    assert typostats.is_one_letter_replacement('caat', 'cat', allow_1to2=True) == []
    # Returns only the first valid interpretation to avoid double-counting
    assert typostats.is_one_letter_replacement('caat', 'cat', allow_1to2=True, include_deletions=True) == [
        ('c', 'ca'),
    ]


def test_is_one_letter_replacement_two_to_one():
    assert typostats.is_one_letter_replacement('f', 'ph', allow_2to1=True) == [('ph', 'f')]
    assert typostats.is_one_letter_replacement('a', 'aa', allow_2to1=True, include_deletions=False) == []
    assert typostats.is_one_letter_replacement('a', 'aa', allow_2to1=True, include_deletions=True) == [('aa', 'a')]


def test_is_one_letter_replacement_filtering():
    # 1-to-2 replacement, but it's an insertion: 'a' -> 'aa'
    assert typostats.is_one_letter_replacement('aa', 'a', allow_1to2=True, include_deletions=False) == []
    # 2-to-1 replacement, but it's a deletion: 'aa' -> 'a'
    assert typostats.is_one_letter_replacement('a', 'aa', allow_2to1=True, include_deletions=False) == []
    # 1-to-2 replacement, not an insertion: 'm' -> 'rn'
    assert typostats.is_one_letter_replacement('rn', 'm', allow_1to2=True, include_deletions=False) == [('m', 'rn')]
    # 2-to-1 replacement, not a deletion: 'ph' -> 'f'
    assert typostats.is_one_letter_replacement('f', 'ph', allow_2to1=True, include_deletions=False) == [('ph', 'f')]


def test_is_one_letter_replacement_edge_cases():
    # Suffix match failures
    assert typostats.is_one_letter_replacement('abc', 'a', allow_1to2=True) == []
    assert typostats.is_one_letter_replacement('a', 'abc', allow_2to1=True) == []
    assert typostats.is_one_letter_replacement('ab', 'abc', allow_2to1=True) == []


def test_process_typos_formats():
    # process_typos now takes pairs directly
    assert typostats.process_typos([('teh', 'the')])[0] == {}
    assert typostats.process_typos([('teh', 'the')], allow_transposition=True)[0] == {('he', 'eh'): 1}
    assert typostats.process_typos([('tezt', 'test')])[0] == {('s', 'z'): 1}

    # Test with multiple pairs
    counts, pairs_count = typostats.process_typos([('tezt', 'test'), ('tezt', 'tent')])
    assert counts == {('s', 'z'): 1, ('n', 'z'): 1}
    assert pairs_count == 2

    # Non-ASCII word filter
    assert typostats.process_typos([('fóo', 'foo')])[0] == {}
    assert typostats.process_typos([('foo', 'fóo')])[0] == {}


def test_process_typos_multi_format():
    pairs = [('tezt', 'test'), ('tezt', 'tent'), ('teht', 'the'), ('tost', 'test')]
    counts, pairs_count = typostats.process_typos(pairs)
    assert counts[('s', 'z')] == 1
    assert counts[('n', 'z')] == 1
    assert counts[('he', 'eh')] == 0
    assert counts[('e', 'o')] == 1
    assert pairs_count == 4

    # Non-ASCII filters
    assert typostats.process_typos([('tést', 'test')])[0] == {}
    assert typostats.process_typos([('test', 'tést')])[0] == {}


def test_generate_report_formats(capsys, tmp_path):
    counts = {('s', 'z'): 3, ('e', 'a'): 1}
    typostats.generate_report(counts, output_format='arrow', quiet=True)
    assert 'z    │ s' in capsys.readouterr().out
    typostats.generate_report(counts, output_format='json')
    assert len(json.loads(capsys.readouterr().out)["replacements"]) == 2
    typostats.generate_report(counts, output_format='csv')
    assert "typo,correction,count" in capsys.readouterr().out
    typostats.generate_report(counts, output_format='yaml')
    assert "  s:" in capsys.readouterr().out
    typostats.generate_report(counts, output_format='table')
    out_table = capsys.readouterr().out
    assert 'z = "s"' in out_table
    assert 'a = "e"' in out_table
    typostats.generate_report(counts, output_format='markdown')
    out_md = capsys.readouterr().out
    assert "| Typo | Correction | Count |" in out_md
    assert "| z | s | 3 |" in out_md
    out_file = tmp_path / "report.txt"
    typostats.generate_report(counts, output_file=str(out_file))
    assert "ANALYSIS SUMMARY" in out_file.read_text()


def test_generate_report_formats_extra():
    counts = {('q', 'w'): 1}
    # JSON with keyboard
    with patch('sys.stdout', new=io.StringIO()) as out:
        typostats.generate_report(counts, output_format='json', keyboard=True)
        data = json.loads(out.getvalue())
        assert data["replacements"][0]["is_adjacent"] is True

    # CSV explicit
    with patch('sys.stdout', new=io.StringIO()) as out:
        typostats.generate_report(counts, output_format='csv')
        assert "w,q,1" in out.getvalue()

    # Generic YAML fallback
    with patch('sys.stdout', new=io.StringIO()) as out:
        typostats.generate_report(counts, output_format='other')
        assert "  q:" in out.getvalue()


def test_generate_report_sorting_and_filtering(capsys):
    counts = {('b', 'z'): 1, ('a', 'y'): 2, ('a', 'x'): 3}
    typostats.generate_report(counts, sort_by='count', output_format='arrow', quiet=True)
    lines = [line for line in capsys.readouterr().out.splitlines() if '│' in line and 'TYPO' not in line and '─' not in line]
    assert 'x' in lines[0] and 'y' in lines[1] and 'z' in lines[2]
    typostats.generate_report(counts, sort_by='typo', output_format='arrow', quiet=True)
    lines = [line for line in capsys.readouterr().out.splitlines() if '│' in line and 'TYPO' not in line and '─' not in line]
    assert 'x' in lines[0] and 'y' in lines[1] and 'z' in lines[2]
    typostats.generate_report(counts, sort_by='correct', output_format='arrow', quiet=True)
    lines = [line for line in capsys.readouterr().out.splitlines() if '│' in line and 'TYPO' not in line and '─' not in line]
    assert '│ a' in lines[0] and '│ b' in lines[2]
    typostats.generate_report(counts, min_occurrences=2, output_format='arrow', quiet=True)
    assert len([line for line in capsys.readouterr().out.splitlines() if '│' in line and 'TYPO' not in line and '─' not in line]) == 2


def test_generate_report_summaries(capsys):
    typostats.generate_report({('he', 'eh'): 1}, allow_transposition=True, quiet=False)
    assert "Transpositions [T]:" in capsys.readouterr().out
    typostats.generate_report({('a', 'b'): 1}, min_occurrences=2, quiet=False)
    assert "Patterns matching criteria:" in capsys.readouterr().out
    typostats.generate_report({('a', 'b'): 2, ('c', 'd'): 1}, limit=1, quiet=False)
    assert "Showing patterns:" in capsys.readouterr().out
    counts = {('a', 'aa'): 1, ('bb', 'b'): 1, ('m', 'rn'): 1, ('ph', 'f'): 1}
    typostats.generate_report(counts, include_deletions=True, allow_1to2=True, allow_2to1=True, quiet=False)
    out = capsys.readouterr().out
    assert "Insertions [Ins]:" in out and "Deletions [Del]:" in out
    assert "1-to-2 replacements [1:2]" in out and "2-to-1 replacements [2:1]" in out


def test_generate_report_keyboard(capsys):
    counts = {('q', 'w'): 5, ('q', 'p'): 1}
    typostats.generate_report(counts, keyboard=True, quiet=False)
    captured = capsys.readouterr()
    assert "Keyboard Adjacency" in captured.out
    assert "[K]" in captured.out
    assert "[K]" not in captured.out.splitlines()[-1]


def test_generate_report_markers(capsys):
    counts = {('a', 'ab'): 1, ('bc', 'b'): 1, ('m', 'rn'): 1, ('ph', 'f'): 1, ('he', 'eh'): 1}
    typostats.generate_report(counts, all=True, quiet=True)
    out = capsys.readouterr().out
    assert all(m in out for m in ["[Ins]", "[Del]", "[1:2]", "[2:1]", "[T]"])


def test_generate_report_markers_extra():
    counts = {
        ('a', 'abc'): 1, # [Ins]
        ('abc', 'a'): 1, # [Del]
        ('a', 'bc'): 1,  # [1:2]
        ('bc', 'a'): 1,  # [2:1]
    }
    with patch('sys.stdout', new=io.StringIO()) as out:
        typostats.generate_report(counts, all=True)
        val = out.getvalue()
        assert all(m in val for m in ["[Ins]", "[Del]", "[1:2]", "[2:1]"])


def test_generate_report_edge_cases():
    # Empty filtering result
    with patch('sys.stdout', new=io.StringIO()) as out:
        typostats.generate_report({}, quiet=False)
        assert "No patterns passed the filtering criteria" in out.getvalue()

    # File write failure
    with patch("builtins.open", side_effect=Exception("Write error")):
        with patch('logging.error') as mock_log:
            typostats.generate_report({('a', 'b'): 1}, output_file="fail.txt")
            mock_log.assert_called()

    # Explicit no results
    with patch('sys.stdout', new=io.StringIO()) as out:
        typostats.generate_report({}, quiet=False)
        assert "No replacements found matching the criteria" in out.getvalue()


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


def test_read_file_lines_robust_variants(tmp_path):
    from unittest.mock import MagicMock
    # Reset STDIN cache
    typostats._STDIN_CACHE = None

    # Mock sys.stdin
    mock_stdin = MagicMock()
    mock_stdin.buffer.read.return_value = b"line1\n"
    with patch('typostats.sys.stdin', mock_stdin):
        assert typostats._read_file_lines_robust('-') == ["line1\n"]

    # Test nonexistent file
    with pytest.raises(SystemExit):
        typostats._read_file_lines_robust(str(tmp_path / "nonexistent"))

    # Test directory
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()
    assert typostats._read_file_lines_robust(str(dir_path)) == []

    # Test encoding fallback
    mock_files = {'dummy.txt': b'\xff'}
    def mocked_open_func(file, mode='r', encoding=None, **kwargs):
        if 'b' in mode:
            return io.BytesIO(mock_files['dummy.txt'])
        if encoding == 'utf-8':
            raise UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')
        if encoding == 'latin-1':
            return io.StringIO("\xff")
        raise UnicodeDecodeError('other', b'', 0, 1, 'invalid')

    with patch('builtins.open', side_effect=mocked_open_func), \
         patch('typostats.detect_encoding', return_value=None), \
         patch('os.path.exists', return_value=True), \
         patch('os.path.isdir', return_value=False):
        assert typostats._read_file_lines_robust('dummy.txt') == ["\xff"]


def test_read_file_lines_robust_encoding_failures():
    # Detected encoding failure
    with patch("builtins.open") as mocked_open:
        def side_effect(file, mode='r', encoding=None, **kwargs):
            if 'b' in mode:
                return io.BytesIO(b"data")
            if mode == 'r' and encoding == 'utf-8':
                raise UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')
            if mode == 'r' and encoding == 'detected':
                raise UnicodeDecodeError('detected', b'', 0, 1, 'invalid')
            if mode == 'r' and encoding == 'latin-1':
                return io.StringIO("latin-1")
            return io.StringIO("default")
        mocked_open.side_effect = side_effect
        with patch("typostats.detect_encoding", return_value="detected"), \
             patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=False):
            assert typostats._read_file_lines_robust("dummy.txt") == ["latin-1"]

    # Detect encoding returns None
    with patch("builtins.open") as mocked_open:
        def side_effect_none(file, mode='r', encoding=None, **kwargs):
            if 'b' in mode:
                return io.BytesIO(b"data")
            if mode == 'r' and encoding == 'utf-8':
                raise UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')
            if mode == 'r' and encoding == 'latin-1':
                return io.StringIO("latin-1_fallback")
            return io.StringIO("default")
        mocked_open.side_effect = side_effect_none
        with patch("typostats.detect_encoding", return_value=None), \
             patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=False):
            assert typostats._read_file_lines_robust("dummy_none.txt") == ["latin-1_fallback"]


def test_main_cli_input_flag(tmp_path):
    f1 = tmp_path / "typos1.txt"
    f1.write_text("teh -> the\n")
    f2 = tmp_path / "typos2.txt"
    f2.write_text("recived -> received\n")

    with patch('sys.argv', ['typostats.py', '-i', str(f1), str(f2), '-q']), \
         patch('typostats.generate_report') as mock_report:
        typostats.main()
        assert mock_report.call_args[1]['total_pairs'] == 2


def test_main_cli_functionality():
    with patch('sys.argv', ['typostats.py', '--help']):
        with pytest.raises(SystemExit):
            typostats.main()
    with patch('sys.argv', ['typostats.py', 'input.txt', '-a', '-q']), \
         patch('typostats._extract_pairs', return_value=[("teh", "the")]), \
         patch('typostats.generate_report') as mock_report:
        typostats.main()
        assert mock_report.call_args[1]['keyboard'] is True and mock_report.call_args[1]['quiet'] is True
    with patch('sys.argv', ['typostats.py', 'input.txt', '--allow-two-char']), \
         patch('typostats._extract_pairs', return_value=[("m", "rn")]), \
         patch('typostats.process_typos', return_value=({}, 0)) as mock_process, \
         patch('typostats.generate_report'):
        typostats.main()
        assert mock_process.call_args[1]['allow_1to2'] is True and mock_process.call_args[1]['allow_2to1'] is True


def test_main_cli_args_extra():
    # args.all = True if no flags
    with patch('sys.argv', ['typostats.py', 'input.txt']), \
         patch('typostats._extract_pairs', return_value=[]), \
         patch('typostats.generate_report') as mock_report:
        typostats.main()
        assert mock_report.call_args[1]['keyboard'] is True

    # input_files = ['-']
    with patch('sys.argv', ['typostats.py']), \
         patch('typostats._extract_pairs', return_value=[]) as mock_extract, \
         patch('typostats.generate_report'):
        typostats.main()
        mock_extract.assert_called_with(['-'], quiet=False)

    # empty result
    with patch('sys.argv', ['typostats.py', 'empty.txt']), \
         patch('typostats._extract_pairs', return_value=[]), \
         patch('typostats.generate_report') as mock_report:
        typostats.main()
        assert mock_report.call_args[1]['total_pairs'] == 0


def test_typostats_subprocess_all(tmp_path):
    typos_file = tmp_path / "typos.txt"
    typos_file.write_text("teh -> the\nrecieve -> receive\nm -> rn\nph -> f\nor -> o\na -> aa\n", encoding="utf-8")
    result = subprocess.run([sys.executable, "typostats.py", str(typos_file), "-a"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Enabled features:" in result.stdout
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


def test_minimal_formatter_color_full():
    formatter = typostats.MinimalFormatter()
    record = logging.LogRecord('n', logging.WARNING, 'p', 1, 'msg', None, None)
    with patch('typostats.sys.stderr.isatty', return_value=True), \
         patch.dict(formatter.LEVEL_COLORS, {logging.WARNING: "\033[1;33m"}), \
         patch('typostats.RESET', "\033[0m"):
        res = formatter.format(record)
        assert "\033[1;33mWARNING\033[0m: msg" in res

    # Test with level not in LEVEL_COLORS
    record_unknown = logging.LogRecord('n', logging.CRITICAL + 1, 'p', 1, 'msg', None, None)
    with patch('typostats.sys.stderr.isatty', return_value=True):
        res = formatter.format(record_unknown)
        assert "Level 51: msg" in res

    record_no_name = logging.LogRecord('n', logging.WARNING, 'p', 1, 'msg', None, None)
    record_no_name.levelname = None
    assert formatter.format(record_no_name) == "None: msg"


def test_format_analysis_summary_branches():
    # Retention bar branches
    report = "\n".join(typostats._format_analysis_summary(10, ["a"] * 10))
    assert "100.0%" in report and "████████████████████" in report

    report = "\n".join(typostats._format_analysis_summary(10, []))
    assert "0.0%" in report and "No items passed" in report

    # Non-hashable unique items
    report = "\n".join(typostats._format_analysis_summary(2, [["a"], ["a"]], item_label="list"))
    assert "Unique items:" in report or "Unique lists:" in report

    # Shortest/Longest
    report = "\n".join(typostats._format_analysis_summary(2, [("a", "bc"), ("def", "g")]))
    assert "Shortest item:" in report or "Shortest replacement:" in report
    assert "Longest item:" in report or "Longest replacement:" in report


def test_format_analysis_summary_extra_full():
    report = typostats._format_analysis_summary(
        10, ["a"] * 5,
        extra_metrics={"Extra": "Value"},
        total_input_items=100,
        start_time=0.0
    )
    report_text = "\n".join(report)
    assert "Total word pairs analyzed:" in report_text
    assert "Extra:" in report_text
    assert "Processing time:" in report_text


def test_format_analysis_summary_edge_cases():
    # Bad item causing TypeError in str()
    class ReallyBadItem:
        def __str__(self): raise TypeError("Really Bad")
    report = typostats._format_analysis_summary(10, [ReallyBadItem()])
    assert report

    # Bad tuple for distances
    report = typostats._format_analysis_summary(10, [("a", "b"), ("c",)])
    assert report


def test_get_adjacent_keys_no_diagonals():
    adj = typostats.get_adjacent_keys(include_diagonals=False)
    assert 'w' in adj['q']
    assert 'a' in adj['q']
    assert 's' not in adj['q']


def test_is_one_letter_replacement_logic_fix():
    # Test that include_deletions=True works independently of allow_1to2/allow_2to1
    # Deletion: 'receive' -> 'receve' (actually multiple potential deletions, but let's test one)
    # is_one_letter_replacement returns a list of potential replacements

    # 1. Deletion case (typo is shorter)
    # 'a' in 'aa' -> deletion of 'a'
    res = typostats.is_one_letter_replacement('a', 'aa', include_deletions=True)
    assert ('aa', 'a') in res

    res = typostats.is_one_letter_replacement('a', 'aa', include_deletions=False)
    assert ('aa', 'a') not in res

    # 2. Insertion case (typo is longer)
    # 'a' in 'aa' -> insertion of 'a'
    res = typostats.is_one_letter_replacement('aa', 'a', include_deletions=True)
    assert ('a', 'aa') in res

    res = typostats.is_one_letter_replacement('aa', 'a', include_deletions=False)
    assert ('a', 'aa') not in res

    # 3. 1-to-2 replacement (not an insertion)
    res = typostats.is_one_letter_replacement('rn', 'm', allow_1to2=True)
    assert ('m', 'rn') in res

    res = typostats.is_one_letter_replacement('rn', 'm', allow_1to2=False)
    assert ('m', 'rn') not in res

    # 4. 2-to-1 replacement (not a deletion)
    res = typostats.is_one_letter_replacement('f', 'ph', allow_2to1=True)
    assert ('ph', 'f') in res

    res = typostats.is_one_letter_replacement('f', 'ph', allow_2to1=False)
    assert ('ph', 'f') not in res

def test_process_typos_logic_fix():
    # Test process_typos with the new logic
    pairs = [("receve", "receive")]
    counts, _ = typostats.process_typos(pairs, include_deletions=True)
    # With early return, only the first possible match is counted
    assert ('ei', 'e') in counts

    counts, _ = typostats.process_typos(pairs, include_deletions=False)
    assert not counts

    pairs = [("aa", "a")]
    counts, _ = typostats.process_typos(pairs, include_deletions=True)
    assert ('a', 'aa') in counts

    counts, _ = typostats.process_typos(pairs, include_deletions=False)
    assert not counts

def test_parse_markdown_table_row_edge_cases():
    # Generic row with no vertical bars
    assert typostats._parse_markdown_table_row("no bars") is None
    # Empty parts row
    assert typostats._parse_markdown_table_row("| | |") == ["", ""]
    # Rows with empty edge parts to test edge trimming branches
    assert typostats._parse_markdown_table_row("|a|b|") == ["a", "b"]
    # Header skip
    assert typostats._parse_markdown_table_row("| Typo | Correction |") is None
    # Header skip with other variants
    assert typostats._parse_markdown_table_row("| item | count |") is None
    # Partial header (should NOT skip)
    assert typostats._parse_markdown_table_row("| Typo | SomethingElse |") == ["Typo", "SomethingElse"]
    # Divider skip
    assert typostats._parse_markdown_table_row("| --- | --- |") is None
    # Too few parts
    assert typostats._parse_markdown_table_row("| only_one |") is None
    # Valid row
    assert typostats._parse_markdown_table_row("| teh | the |") == ["teh", "the"]


def test_extract_pairs_json_variations(tmp_path):
    # JSON list of dicts with 'typo' and 'correct'
    # Also test item that is not a dict, or missing typo, or missing correct
    f = tmp_path / "test1.json"
    f.write_text(json.dumps([
        {"typo": "teh", "correct": "the"},
        "not a dict",
        {"no typo": "here"},
        {"typo": "missing_correct", "correct": None}
    ]), encoding="utf-8")
    assert list(typostats._extract_pairs([str(f)])) == [("teh", "the")]

    # JSON dict with 'replacements' list
    # Also test item missing typo or missing correct
    f = tmp_path / "test2.json"
    f.write_text(json.dumps({"replacements": [
        {"typo": "teh", "correction": "the"},
        {"no typo": "here"},
        {"typo": "missing_correct", "correct": None}
    ]}), encoding="utf-8")
    assert list(typostats._extract_pairs([str(f)])) == [("teh", "the")]

    # JSON that is just a string (neither dict nor list)
    f = tmp_path / "string.json"
    f.write_text(json.dumps("just a string"), encoding="utf-8")
    assert list(typostats._extract_pairs([str(f)])) == []

    # Empty JSON
    f = tmp_path / "empty.json"
    f.write_text("   ", encoding="utf-8")
    assert list(typostats._extract_pairs([str(f)])) == []

    # JSON flat dict
    f = tmp_path / "test3.json"
    f.write_text(json.dumps({"teh": "the"}), encoding="utf-8")
    assert list(typostats._extract_pairs([str(f)])) == [("teh", "the")]

    # JSON with 'replacements' but not a list
    f = tmp_path / "test5.json"
    f.write_text(json.dumps({"replacements": "not a list", "a": "b"}), encoding="utf-8")
    assert list(typostats._extract_pairs([str(f)])) == [("replacements", "not a list"), ("a", "b")]

    # JSON invalid/empty
    f = tmp_path / "test4.json"
    f.write_text("invalid", encoding="utf-8")
    with patch('logging.error') as mock_log:
        assert list(typostats._extract_pairs([str(f)])) == []
        mock_log.assert_called()


def test_extract_pairs_yaml_variations(tmp_path):
    if not typostats._YAML_AVAILABLE:
        pytest.skip("PyYAML not available")

    # YAML list of dicts
    # Also test item with 'typo' but missing correct, and item not a dict
    f = tmp_path / "test1.yaml"
    f.write_text("""
- typo: teh
  correct: the
- typo: only_typo
- not_a_dict: value
- k: v
""", encoding="utf-8")
    # item with only 'typo' will yield ('typo', 'only_typo') because it falls back to k,v iteration
    assert list(typostats._extract_pairs([str(f)])) == [("teh", "the"), ("typo", "only_typo"), ("not_a_dict", "value"), ("k", "v")]

    # YAML that is just a string
    f = tmp_path / "string.yaml"
    f.write_text("just a string", encoding="utf-8")
    assert list(typostats._extract_pairs([str(f)])) == []

    # YAML flat dict
    f = tmp_path / "test2.yaml"
    f.write_text("teh: the", encoding="utf-8")
    assert list(typostats._extract_pairs([str(f)])) == [("teh", "the")]

    # YAML with nested list (should be ignored by inner loop if not a dict)
    f = tmp_path / "test_nested.yaml"
    f.write_text("- - nested list", encoding="utf-8")
    assert list(typostats._extract_pairs([str(f)])) == []

    # YAML multiple docs
    f = tmp_path / "test3.yaml"
    f.write_text("a: b\n---\nc: d", encoding="utf-8")
    assert list(typostats._extract_pairs([str(f)])) == [("a", "b"), ("c", "d")]

    # YAML invalid
    f = tmp_path / "test4.yaml"
    f.write_text("!!invalid", encoding="utf-8")
    with patch('logging.error') as mock_log:
        assert list(typostats._extract_pairs([str(f)])) == []
        mock_log.assert_called()


def test_extract_pairs_text_variations(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("""
# Comment
teh -> the
m = "rn"
key: value
csv1,csv2
* bullet -> fix
| Typo | Correction |
| --- | --- |
| table_typo | table_fix |
""", encoding="utf-8")
    pairs = list(typostats._extract_pairs([str(f)], quiet=True))
    assert ("teh", "the") in pairs
    assert ("m", "rn") in pairs
    assert ("key", "value") in pairs
    assert ("csv1", "csv2") in pairs
    assert ("bullet", "fix") in pairs
    assert ("table_typo", "table_fix") in pairs


def test_extract_pairs_csv_error(tmp_path):
    f = tmp_path / "test.txt"
    # A line that might cause csv.Error - actually csv.reader is very robust,
    # but we can try to trigger StopIteration by providing an empty row if possible
    with patch('csv.reader', side_effect=csv.Error("test error")):
        f.write_text("a,b", encoding="utf-8")
        assert list(typostats._extract_pairs([str(f)], quiet=True)) == []


def test_generate_report_with_file_output(tmp_path):
    counts = {('s', 'z'): 1}
    out_file = tmp_path / "report.txt"
    # Test with arrow format to file
    typostats.generate_report(counts, output_file=str(out_file), output_format='arrow')
    content = out_file.read_text()
    assert "LETTER REPLACEMENTS" in content
    assert "z" in content and "s" in content

    # Test with empty results to file
    typostats.generate_report({}, output_file=str(out_file))
    assert "No replacements found" in out_file.read_text()


def test_main_with_stdin(tmp_path):
    # Reset STDIN cache
    typostats._STDIN_CACHE = None
    mock_stdin = io.StringIO("teh -> the\n")
    with patch('sys.stdin', mock_stdin), \
         patch('sys.argv', ['typostats.py']), \
         patch('typostats.generate_report') as mock_report:
        typostats.main()
        # total_lines should be 1
        assert mock_report.call_args[1]['total_lines'] == 1


def test_main_with_multiple_files(tmp_path):
    f1 = tmp_path / "f1.txt"
    f1.write_text("a -> b\n")
    f2 = tmp_path / "f2.txt"
    f2.write_text("c -> d\n")

    with patch('sys.argv', ['typostats.py', str(f1), str(f2)]), \
         patch('typostats.generate_report') as mock_report:
        typostats.main()
        assert mock_report.call_args[1]['total_pairs'] == 2


def test_main_no_yaml(caplog):
    with patch('typostats._YAML_AVAILABLE', False), \
         patch('sys.argv', ['typostats.py', 'test.yaml']):
        typostats.main()
        assert "PyYAML not installed" in caplog.text

def test_levenshtein_distance_optimization():
    assert typostats.levenshtein_distance('a', 'abc') == 2


def test_extract_pairs_json_variants(tmp_path):
    f1 = tmp_path / "simple.json"
    f1.write_text(json.dumps({"typo1": "correct1"}), encoding="utf-8")
    assert list(typostats._extract_pairs([str(f1)])) == [("typo1", "correct1")]

    f2 = tmp_path / "list.json"
    f2.write_text(json.dumps([{"typo": "typo2", "correction": "correct2"}]), encoding="utf-8")
    assert list(typostats._extract_pairs([str(f2)])) == [("typo2", "correct2")]


def test_extract_pairs_yaml_variants(tmp_path):
    if not typostats._YAML_AVAILABLE:
        pytest.skip("PyYAML not installed")
    f1 = tmp_path / "test.yaml"
    f1.write_text("- typo: t1\n  correct: c1\n- key2: val2", encoding="utf-8")
    assert list(typostats._extract_pairs([str(f1)])) == [("t1", "c1"), ("key2", "val2")]


def test_extract_pairs_text_formats(tmp_path):
    f1 = tmp_path / "text.txt"
    f1.write_text('  - bullet -> bullet_fix\nkey = "value"\ncolon: fix', encoding="utf-8")
    pairs = list(typostats._extract_pairs([str(f1)]))
    assert ("bullet", "bullet_fix") in pairs
    assert ("key", "value") in pairs
    assert ("colon", "fix") in pairs


def test_read_file_lines_robust_stdin_cache_explicit():
    typostats._STDIN_CACHE = ["line1\n"]
    assert typostats._read_file_lines_robust("-") == ["line1\n"]


def test_generate_report_with_file_variants(tmp_path, capsys):
    out_file = tmp_path / "report.txt"
    counts = {("correct", "typo"): 1}
    # When quiet=True, summary and headers are omitted even when writing to a file
    typostats.generate_report(counts, output_file=str(out_file), quiet=False)
    assert out_file.exists()
    assert "Correction" in out_file.read_text()

    out_file_empty = tmp_path / "empty.txt"
    typostats.generate_report({}, output_file=str(out_file_empty), quiet=True)
    assert "No replacements found" in out_file_empty.read_text()


def test_generate_report_extra_metrics_full(capsys):
    # correct, typo pairs
    counts = {
        ("te", "t"): 1,     # Deletion typo is 't', correct is 'te'
        ("t", "te"): 1,     # Insertion typo is 'te', correct is 't'
        ("m", "rn"): 1,     # 1:2
        ("ph", "f"): 1,     # 2:1
        ("he", "eh"): 1,    # Transposition
        ("q", "w"): 1,      # [K]
    }
    typostats.generate_report(counts, include_deletions=True, allow_1to2=True, allow_2to1=True, allow_transposition=True, quiet=False, keyboard=True, total_lines=10)
    out = capsys.readouterr().out
    assert "Total lines processed:" in out
    assert "Transpositions [T]:" in out
    assert "Keyboard Adjacency [K]:" in out
    assert "Insertions [Ins]:" in out
    assert "Deletions [Del]:" in out
    assert "1-to-2 replacements [1:2]:" in out
    assert "2-to-1 replacements [2:1]:" in out


def test_generate_report_json_keyboard_branches():
    counts = {("he", "eh"): 1, ("q", "p"): 1}
    with patch('sys.stdout', new=io.StringIO()) as out:
        typostats.generate_report(counts, output_format='json', keyboard=True)
        data = json.loads(out.getvalue())
        for item in data["replacements"]:
            assert item["is_adjacent"] is False


def test_main_stdin_path_explicit(capsys):
    with patch('sys.stdin', io.StringIO("typo -> correct\n")),          patch('sys.argv', ['typostats.py', '-a']),          patch('typostats._STDIN_CACHE', None):
        typostats.main()
        out = capsys.readouterr().out
        assert "Total word pairs analyzed:" in out


def test_generate_report_neutral_bar_color(capsys):
    import os
    counts = {("a", "b"): 1}
    with patch('sys.stdout.isatty', return_value=True), \
         patch.dict(os.environ, {}, clear=True):
        typostats.generate_report(counts, output_format='arrow', quiet=False)
        out = capsys.readouterr().out
        assert "\033[1;36m" in out


def test_should_enable_color_via_env_vars(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert typostats._should_enable_color(sys.stdout) is False
    monkeypatch.delenv("NO_COLOR")

    monkeypatch.setenv("FORCE_COLOR", "1")
    assert typostats._should_enable_color(sys.stdout) is True
    monkeypatch.delenv("FORCE_COLOR")

    mock_stream = MagicMock()
    mock_stream.isatty.return_value = True
    assert typostats._should_enable_color(mock_stream) is True

    mock_stream.isatty.return_value = False
    assert typostats._should_enable_color(mock_stream) is False


def test_detect_format_from_extension_resolution():
    allowed = ["json", "csv", "yaml", "arrow", "table", "markdown", "md"]
    assert typostats._detect_format_from_extension("test.json", allowed, "arrow") == "json"
    assert typostats._detect_format_from_extension("test.csv", allowed, "arrow") == "csv"
    assert typostats._detect_format_from_extension("test.yaml", allowed, "arrow") == "yaml"
    assert typostats._detect_format_from_extension("test.yml", allowed, "arrow") == "yaml"
    assert typostats._detect_format_from_extension("test.arrow", allowed, "arrow") == "arrow"
    assert typostats._detect_format_from_extension("test.txt", allowed, "arrow") == "arrow"
    assert typostats._detect_format_from_extension("test.toml", allowed, "arrow") == "table"
    assert typostats._detect_format_from_extension("test.table", allowed, "arrow") == "table"
    assert typostats._detect_format_from_extension("test.md", allowed, "arrow") == "markdown"
    assert typostats._detect_format_from_extension("test.markdown", allowed, "arrow") == "markdown"
    assert typostats._detect_format_from_extension("test.unknown", allowed, "default") == "default"
    assert typostats._detect_format_from_extension("testfile", allowed, "default") == "default"
    assert typostats._detect_format_from_extension("-", allowed, "default") == "default"
    assert typostats._detect_format_from_extension("", allowed, "default") == "default"


def test_markdown_export_cli_integration(tmp_path):
    input_file = tmp_path / "typos.txt"
    input_file.write_text("teh -> the\n")
    output_md = tmp_path / "report.md"
    with patch("sys.argv", ["typostats.py", str(input_file), "-f", "md", "-o", str(output_md)]):
        typostats.main()
    assert output_md.exists()
    content = output_md.read_text()
    assert "| Typo | Correction | Count |" in content
    assert "| eh | he | 1 |" in content


def test_is_one_letter_replacement_disallowed_patterns_check():
    assert typostats.is_one_letter_replacement("bc", "a", allow_1to2=False, include_deletions=True) == []
    assert typostats.is_one_letter_replacement("c", "ab", allow_2to1=False, include_deletions=True) == []


def test_read_file_lines_robust_stdin_string_cache(monkeypatch):
    monkeypatch.setattr(typostats, "_STDIN_CACHE", None)
    mock_stdin = MagicMock()
    if hasattr(mock_stdin, 'buffer'):
        del mock_stdin.buffer
    mock_stdin.read.return_value = "line1\nline2\n"
    with patch("sys.stdin", mock_stdin):
        assert typostats._read_file_lines_robust("-") == ["line1\n", "line2\n"]


def test_read_file_lines_robust_stdin_binary_fallback_handling(monkeypatch):
    monkeypatch.setattr(typostats, "_STDIN_CACHE", None)
    mock_stdin = MagicMock()
    mock_buffer = MagicMock()
    mock_buffer.read.return_value = b"\xe9\n"
    mock_stdin.buffer = mock_buffer
    with patch("sys.stdin", mock_stdin):
        assert typostats._read_file_lines_robust("-") == ["\xe9\n"]


def test_read_file_lines_robust_with_directory_path(tmp_path):
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()
    assert typostats._read_file_lines_robust(str(dir_path)) == []


def test_read_file_lines_robust_file_encoding_fallback_cases(tmp_path):
    file_path = tmp_path / "latin1.txt"
    with open(file_path, "wb") as f:
        f.write(b"\xe9\n")
    with patch("typostats.detect_encoding", return_value="latin-1"):
        assert typostats._read_file_lines_robust(str(file_path)) == ["\xe9\n"]
    with patch("typostats.detect_encoding", return_value=None):
        assert typostats._read_file_lines_robust(str(file_path)) == ["\xe9\n"]


def test_read_file_lines_robust_file_detect_encoding_fails_midway_handling(tmp_path):
    file_path = tmp_path / "latin1_v2.txt"
    with open(file_path, "wb") as f:
        f.write(b"\xe9\n")
    with patch("typostats.detect_encoding", return_value="utf-8"):
        assert typostats._read_file_lines_robust(str(file_path)) == ["\xe9\n"]


def test_typostats_main_basic_invocation(tmp_path):
    input_file = tmp_path / "input.csv"
    input_file.write_text("typo,correction\nteh,the")
    with patch("sys.argv", ["typostats.py", str(input_file), "--format", "json"]):
        try:
            typostats.main()
        except SystemExit:
            pass


def test_typostats_recursive_directory_scanning_integration(tmp_path, monkeypatch):
    root = tmp_path / "typostats_test_root"
    root.mkdir()
    subdir = root / "subdir"
    subdir.mkdir()
    ignored_dir = root / "node_modules"
    ignored_dir.mkdir()
    file1 = root / "file1.txt"
    file1.write_text("teh -> the\n")
    file2 = subdir / "file2.csv"
    file2.write_text("recived,received\n")
    file3 = subdir / "file3.json"
    file3.write_text('{"replacements": [{"typo": "seperate", "correct": "separate"}]}')
    ignored_file = ignored_dir / "ignored.txt"
    ignored_file.write_text("ignoredtypo -> correction\n")
    unsupported_file = root / "script.py"
    unsupported_file.write_text("pytypo -> pycorrect\n")
    output_file = tmp_path / "report.json"
    monkeypatch.setattr("typostats._STDIN_CACHE", None)
    with patch("sys.argv", ["typostats.py", str(root), "--format", "json", "--output", str(output_file), "--all"]):
        typostats.main()
    assert output_file.exists()
    content = output_file.read_text()
    assert "ignoredtypo" not in content
    assert "pytypo" not in content
    assert '"typo": "e"' in content
    assert '"correct": "a"' in content


def test_typostats_interactive_terminal_scans_cwd_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file1 = tmp_path / "typos.txt"
    file1.write_text("teh -> the\n")
    monkeypatch.setattr("typostats._STDIN_CACHE", None)
    with patch("sys.stdin.isatty", return_value=True), \
         patch("sys.argv", ["typostats.py", "--quiet", "--output", str(tmp_path / "report.json"), "--all"]):
        typostats.main()
    report_file = tmp_path / "report.json"
    assert report_file.exists()
    content = report_file.read_text()
    assert '"typo": "eh"' in content or "eh" in content


def test_typostats_non_interactive_terminal_reads_stdin_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr("typostats._STDIN_CACHE", None)
    mock_stdin = io.StringIO("recived -> received\n")
    with patch("sys.stdin.isatty", return_value=False), \
         patch("sys.stdin", mock_stdin), \
         patch("sys.argv", ["typostats.py", "--quiet", "--output", str(tmp_path / "report.json"), "--all"]):
        typostats.main()
    report_file = tmp_path / "report.json"
    assert report_file.exists()
    content = report_file.read_text()
    assert '"typo": "i"' in content or "i" in content


def test_typostats_main_run_as_script(tmp_path):
    f1 = tmp_path / "left.txt"
    f1.write_text("apple\n")
    f2 = tmp_path / "right.txt"
    f2.write_text("aple\n")
    import runpy
    with patch.object(sys, 'argv', ["typostats.py", str(f1), str(f2), "--quiet"]):
        runpy.run_path("typostats.py", run_name="__main__")


def test_generate_report_table_format(tmp_path):
    out_file = tmp_path / "report.toml"
    counts = {("he", "eh"): 2, ("the", "teh"): 1, ("do not", "don't"): 1}
    typostats.generate_report(
        counts,
        output_file=str(out_file),
        output_format="table",
    )
    assert out_file.exists()
    content = out_file.read_text()
    assert 'eh = "he"' in content
    assert 'teh = "the"' in content
    assert '"don\'t" = "do not"' in content


def test_detect_format_from_extension_table_and_toml():
    allowed = ['arrow', 'yaml', 'json', 'csv', 'table', 'markdown', 'md']
    assert typostats._detect_format_from_extension("output.table", allowed, "arrow") == "table"
    assert typostats._detect_format_from_extension("output.toml", allowed, "arrow") == "table"


def test_typostats_cli_table_format(tmp_path):
    typo_file = tmp_path / "typos.txt"
    typo_file.write_text("teh -> the\n")
    out_file = tmp_path / "typos.toml"
    with patch("sys.argv", ["typostats.py", str(typo_file), "-o", str(out_file), "-f", "table"]):
        typostats.main()
    assert out_file.exists()
    content = out_file.read_text()
    assert 'eh = "he"' in content


def test_should_enable_color_gap(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert typostats._should_enable_color(sys.stdout) is False
    monkeypatch.delenv("NO_COLOR")

    monkeypatch.setenv("FORCE_COLOR", "1")
    assert typostats._should_enable_color(sys.stdout) is True
    monkeypatch.delenv("FORCE_COLOR")

    mock_stream = MagicMock()
    mock_stream.isatty.return_value = True
    assert typostats._should_enable_color(mock_stream) is True

    mock_stream.isatty.return_value = False
    assert typostats._should_enable_color(mock_stream) is False


def test_detect_format_from_extension_main_gap(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("teh -> the\n")

    extensions_to_expected = {
        "test.json": "json",
        "test.csv": "csv",
        "test.yaml": "yaml",
        "test.yml": "yaml",
        "test.arrow": "arrow",
        "test.txt": "arrow",
        "test.toml": "table",
        "test.md": "markdown",
        "test.unknown": "arrow",
        "testfile": "arrow",
        "-": "arrow",
        "": "arrow",
    }

    for filename, expected_format in extensions_to_expected.items():
        out_path = str(tmp_path / filename) if filename not in ("-", "") else filename
        argv_args = ["typostats.py", str(input_file), "--quiet"]
        if out_path:
            argv_args.extend(["-o", out_path])

        with patch("sys.argv", argv_args), patch("typostats.generate_report") as mock_report:
            try:
                typostats.main()
            except SystemExit:
                pass
            assert mock_report.called, f"generate_report not called for output {filename}"
            kwargs = mock_report.call_args[1]
            assert kwargs["output_format"] == expected_format, f"Expected {expected_format} for output {filename}, but got {kwargs['output_format']}"


def test_is_one_letter_replacement_disallowed_patterns_gap():
    assert typostats.is_one_letter_replacement("bc", "a", allow_1to2=False, include_deletions=True) == []
    assert typostats.is_one_letter_replacement("c", "ab", allow_2to1=False, include_deletions=True) == []


def test_read_file_lines_robust_stdin_string_gap(monkeypatch):
    monkeypatch.setattr(typostats, "_STDIN_CACHE", None)

    mock_stdin = MagicMock()
    if hasattr(mock_stdin, 'buffer'):
        del mock_stdin.buffer
    mock_stdin.read.return_value = "line1\nline2\n"

    with patch("sys.stdin", mock_stdin):
        lines = typostats._read_file_lines_robust("-")
        assert lines == ["line1\n", "line2\n"]


def test_read_file_lines_robust_stdin_binary_fallback_gap(monkeypatch):
    monkeypatch.setattr(typostats, "_STDIN_CACHE", None)

    mock_stdin = MagicMock()
    mock_buffer = MagicMock()
    mock_buffer.read.return_value = b"\xe9\n"
    mock_stdin.buffer = mock_buffer

    with patch("sys.stdin", mock_stdin):
        lines = typostats._read_file_lines_robust("-")
        assert lines == ["\xe9\n"]


def test_read_file_lines_robust_directory_gap(tmp_path):
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()

    lines = typostats._read_file_lines_robust(str(dir_path))
    assert lines == []


def test_read_file_lines_robust_file_encoding_fallback_gap(tmp_path):
    file_path = tmp_path / "latin1.txt"
    with open(file_path, "wb") as f:
        f.write(b"\xe9\n")

    with patch("typostats.detect_encoding", return_value="latin-1"):
        lines = typostats._read_file_lines_robust(str(file_path))
        assert lines == ["\xe9\n"]

    with patch("typostats.detect_encoding", return_value=None):
        lines = typostats._read_file_lines_robust(str(file_path))
        assert lines == ["\xe9\n"]


def test_read_file_lines_robust_file_detect_encoding_fails_midway_gap(tmp_path):
    file_path = tmp_path / "latin1_v2.txt"
    with open(file_path, "wb") as f:
        f.write(b"\xe9\n")

    with patch("typostats.detect_encoding", return_value="utf-8"):
        lines = typostats._read_file_lines_robust(str(file_path))
        assert lines == ["\xe9\n"]


def test_typostats_main_basic_gap(tmp_path):
    input_file = tmp_path / "input.csv"
    input_file.write_text("typo,correction\nteh,the")

    with patch("sys.argv", ["typostats.py", str(input_file), "--format", "json"]):
        try:
            typostats.main()
        except SystemExit:
            pass


def test_format_analysis_summary_levenshtein_exception():
    with patch("typostats.levenshtein_distance", side_effect=ValueError("Test Exception")):
        items = [("the", "teh")]
        report = typostats._format_analysis_summary(
            raw_count=1,
            filtered_items=items,
            item_label="pattern",
            use_color=False,
        )
        assert len(report) > 0
        assert not any("Min/Max/Avg changes:" in line for line in report)


def test_detect_encoding_mock_chardet_branches(tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_bytes(b"sample content")

    mock_chardet = MagicMock()
    mock_chardet.detect.return_value = {'encoding': 'utf-8', 'confidence': 0.95}

    with patch('typostats._CHARDET_AVAILABLE', True), \
         patch('typostats.chardet', mock_chardet):
        assert typostats.detect_encoding(str(test_file)) == 'utf-8'

    mock_chardet.detect.return_value = {'encoding': 'ascii', 'confidence': 0.30}
    with patch('typostats._CHARDET_AVAILABLE', True), \
         patch('typostats.chardet', mock_chardet):
        assert typostats.detect_encoding(str(test_file)) is None

    mock_chardet.detect.return_value = {'encoding': None, 'confidence': 0.0}
    with patch('typostats._CHARDET_AVAILABLE', True), \
         patch('typostats.chardet', mock_chardet):
        assert typostats.detect_encoding(str(test_file)) is None


def test_format_analysis_summary_type_error_len():
    class BadStrItem:
        def __str__(self):
            raise TypeError("Bad str item")

    report = typostats._format_analysis_summary(
        raw_count=1,
        filtered_items=[BadStrItem()],
        item_label="item",
        use_color=False,
    )
    assert len(report) > 0


def test_extract_pairs_tqdm_progress_branch(tmp_path):
    input_file = tmp_path / "tqdm_test.txt"
    input_file.write_text("teh -> the\n")
    mock_tqdm = MagicMock(side_effect=lambda iterable, **kwargs: iterable)
    with patch("typostats._TQDM_AVAILABLE", True), \
         patch("typostats.tqdm", mock_tqdm):
        pairs = list(typostats._extract_pairs([str(input_file)], quiet=False))
        assert ("teh", "the") in pairs


def test_main_dry_run_mock_arg(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("teh -> the\n")
    mock_args = MagicMock()
    mock_args.input_files = [str(input_file)]
    mock_args.input_files_flag = None
    mock_args.output = "-"
    mock_args.min = 1
    mock_args.sort = "count"
    mock_args.format = "arrow"
    mock_args.allow_1to2 = False
    mock_args.allow_2to1 = False
    mock_args.include_deletions = False
    mock_args.transposition = False
    mock_args.keyboard = False
    mock_args.all = True
    mock_args.allow_two_char = False
    mock_args.limit = None
    mock_args.quiet = True
    mock_args.exclude = None
    mock_args.dry_run = MagicMock()

    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        typostats.main()


def test_main_line_counting_oserror_branch(tmp_path):
    input_file = tmp_path / "valid.txt"
    input_file.write_text("teh -> the\n")

    orig_open = open
    def mock_open_func(*args, **kwargs):
        if args and str(args[0]) == str(input_file):
            mode = args[1] if len(args) > 1 else kwargs.get('mode', 'r')
            if mode == 'rb':
                raise OSError("Access error during line count")
        return orig_open(*args, **kwargs)

    with patch("builtins.open", side_effect=mock_open_func), \
         patch("sys.argv", ["typostats.py", str(input_file), "--quiet"]):
        typostats.main()
