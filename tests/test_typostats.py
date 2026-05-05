import json
import csv
import sys
import logging
import io
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
    lines = [l for l in capsys.readouterr().out.splitlines() if '│' in l and 'TYPO' not in l and '─' not in l]
    assert 'x' in lines[0] and 'y' in lines[1] and 'z' in lines[2]
    typostats.generate_report(counts, sort_by='typo', output_format='arrow', quiet=True)
    lines = [l for l in capsys.readouterr().out.splitlines() if '│' in l and 'TYPO' not in l and '─' not in l]
    assert 'x' in lines[0] and 'y' in lines[1] and 'z' in lines[2]
    typostats.generate_report(counts, sort_by='correct', output_format='arrow', quiet=True)
    lines = [l for l in capsys.readouterr().out.splitlines() if '│' in l and 'TYPO' not in l and '─' not in l]
    assert '│ a' in lines[0] and '│ b' in lines[2]
    typostats.generate_report(counts, min_occurrences=2, output_format='arrow', quiet=True)
    assert len([l for l in capsys.readouterr().out.splitlines() if '│' in l and 'TYPO' not in l and '─' not in l]) == 2


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
    with patch('sys.stderr', new=io.StringIO()) as err:
        typostats.generate_report({}, quiet=False)
        assert "No patterns passed the filtering criteria" in err.getvalue()

    # File write failure
    with patch("builtins.open", side_effect=Exception("Write error")):
        with patch('logging.error') as mock_log:
            typostats.generate_report({('a', 'b'): 1}, output_file="fail.txt")
            mock_log.assert_called()

    # Explicit no results
    with patch('sys.stderr', new=io.StringIO()) as err:
        typostats.generate_report({}, quiet=False)
        assert "No replacements found matching the criteria" in err.getvalue()


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
        if 'b' in mode: return io.BytesIO(mock_files['dummy.txt'])
        if encoding == 'utf-8': raise UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')
        if encoding == 'latin-1': return io.StringIO("\xff")
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
            if 'b' in mode: return io.BytesIO(b"data")
            if mode == 'r' and encoding == 'utf-8': raise UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')
            if mode == 'r' and encoding == 'detected': raise UnicodeDecodeError('detected', b'', 0, 1, 'invalid')
            if mode == 'r' and encoding == 'latin-1': return io.StringIO("latin-1")
            return io.StringIO("default")
        mocked_open.side_effect = side_effect
        with patch("typostats.detect_encoding", return_value="detected"), \
             patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=False):
            assert typostats._read_file_lines_robust("dummy.txt") == ["latin-1"]

    # Detect encoding returns None
    with patch("builtins.open") as mocked_open:
        def side_effect_none(file, mode='r', encoding=None, **kwargs):
            if 'b' in mode: return io.BytesIO(b"data")
            if mode == 'r' and encoding == 'utf-8': raise UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')
            if mode == 'r' and encoding == 'latin-1': return io.StringIO("latin-1_fallback")
            return io.StringIO("default")
        mocked_open.side_effect = side_effect_none
        with patch("typostats.detect_encoding", return_value=None), \
             patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=False):
            assert typostats._read_file_lines_robust("dummy_none.txt") == ["latin-1_fallback"]


def test_main_cli_functionality():
    with patch('sys.argv', ['typostats.py', '--help']):
        with pytest.raises(SystemExit): typostats.main()
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
    assert ('ei', 'e') in counts
    assert ('iv', 'v') in counts

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
    typostats.generate_report(counts, output_file=str(out_file), quiet=True)
    assert out_file.exists()
    assert "CORRECT" in out_file.read_text()

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
    err = capsys.readouterr().err
    assert "Total lines processed:" in err
    assert "Transpositions [T]:" in err
    assert "Keyboard Adjacency [K]:" in err
    assert "Insertions [Ins]:" in err
    assert "Deletions [Del]:" in err
    assert "1-to-2 replacements [1:2]:" in err
    assert "2-to-1 replacements [2:1]:" in err


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
        err = capsys.readouterr().err
        assert "Total word pairs analyzed:" in err
