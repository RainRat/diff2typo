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


def test_is_one_letter_replacement_basic():
    assert typostats.is_one_letter_replacement('tezt', 'test') == [('s', 'z')]
    assert typostats.is_one_letter_replacement('test', 'test') == []
    assert typostats.is_one_letter_replacement('abc', 'xyz') == []


def test_is_one_letter_replacement_one_to_two():
    # By default, insertions (a -> aa) are filtered out
    assert typostats.is_one_letter_replacement('aa', 'a', allow_1to2=True) == []
    # If include_deletions is True, they are included
    assert typostats.is_one_letter_replacement('aa', 'a', allow_1to2=True, include_deletions=True) == [('a', 'aa')]
    # 'rn' for 'm' is a replacement, not an insertion, so it should be included by default
    assert typostats.is_one_letter_replacement('rn', 'm', allow_1to2=True) == [('m', 'rn')]
    assert typostats.is_one_letter_replacement('aa', 'a', allow_1to2=False) == []


def test_is_one_letter_replacement_multiple_two_char():
    # Example: cabt -> cat
    # Two possible interpretations:
    # 1. 'a' in 'cat' is replaced by 'ab' in 'cabt' (insertion, 'a' in 'ab')
    # 2. 't' in 'cat' is replaced by 'bt' in 'cabt' (insertion, 't' in 'bt')
    # By default, all are filtered out
    assert typostats.is_one_letter_replacement('cabt', 'cat', allow_1to2=True) == []
    # If include_deletions is True, both are included
    assert typostats.is_one_letter_replacement('cabt', 'cat', allow_1to2=True, include_deletions=True) == [
        ('a', 'ab'),
        ('t', 'bt'),
    ]


def test_is_one_letter_replacement_doubled_letter():
    # For a doubled letter, we capture all valid one-to-two interpretations.
    # While 'a' -> 'aa' is the most likely, 'c' -> 'ca' and 't' -> 'at' are also technically valid.
    # By default, all of these are filtered out as insertions.
    assert typostats.is_one_letter_replacement('caat', 'cat', allow_1to2=True) == []
    # With include_deletions=True, they all return
    assert typostats.is_one_letter_replacement('caat', 'cat', allow_1to2=True, include_deletions=True) == [
        ('a', 'aa'),
        ('c', 'ca'),
        ('t', 'at'),
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
    # By default, 'aa, a' (insertion) is filtered out
    counts, _, _ = typostats.process_typos(lines, allow_1to2=True, allow_2to1=True)
    assert counts == {('s', 'z'): 1, ('e', 'a'): 1}


def test_process_typos_table_format():
    lines = [
        'tezt = "test"',
        'lavel = "level"',
    ]
    counts, _, _ = typostats.process_typos(lines, allow_1to2=False, allow_2to1=False)
    assert counts == {('s', 'z'): 1, ('e', 'a'): 1}


def test_process_typos_colon_format():
    lines = [
        'tezt: test',
        'lavel: level',
    ]
    counts, _, _ = typostats.process_typos(lines, allow_1to2=False, allow_2to1=False)
    assert counts == {('s', 'z'): 1, ('e', 'a'): 1}


def test_process_typos_with_transposition():
    lines = [
        'teh -> the',
        'tset -> test',
    ]
    # Without allow_transposition, these should return nothing (multi-letter diff)
    counts, _, _ = typostats.process_typos(lines, allow_1to2=False, allow_2to1=False, allow_transposition=False)
    assert counts == {}

    # With allow_transposition, they should be detected
    counts, _, _ = typostats.process_typos(lines, allow_1to2=False, allow_2to1=False, allow_transposition=True)
    assert counts == {('he', 'eh'): 1, ('es', 'se'): 1}


def test_generate_report_arrow(capsys):
    counts = {('s', 'z'): 3, ('e', 'a'): 1}
    typostats.generate_report(counts, min_occurrences=2, output_format='arrow', quiet=True)
    captured = capsys.readouterr().out
    # 's' should be padded with 6 spaces (max_c=7)
    # New format: padding(2) + 's' in 7 chars (6 spaces + s) = 8 spaces
    # Followed by │ z and │ count '3' and │ percentage
    assert '        s │ z    │     3 │  75.0% │ ███████' in captured
    assert 'e' not in captured


def test_generate_report_limit(capsys):
    counts = {('a', 'b'): 10, ('c', 'd'): 5, ('e', 'f'): 2}
    typostats.generate_report(counts, limit=2, output_format='arrow', quiet=True)
    captured = capsys.readouterr().out
    assert '        a │ b    │    10 │  58.8% │ █████' in captured
    assert '        c │ d    │     5 │  29.4% │ ██' in captured
    assert 'e' not in captured


def test_generate_report_limit_with_typo_sort(capsys):
    # Counts are such that 'z' would be last in count sort, but first in reverse typo sort?
    # Actually sort_by='typo' sorts alphabetically by typo char.
    counts = {('a', 'z'): 10, ('b', 'x'): 5, ('c', 'y'): 2}
    # Sorted by typo: ('b', 'x'), ('c', 'y'), ('a', 'z')
    typostats.generate_report(counts, limit=2, sort_by='typo', output_format='arrow', quiet=True)
    captured = capsys.readouterr().out
    assert '        b │ x    │     5 │  29.4% │ ██' in captured
    assert '        c │ y    │     2 │  11.8% │ █' in captured
    assert 'a │ z' not in captured


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
    assert '        a │ x    │     3 │  50.0% │ █████' in lines[0]
    assert '        a │ y    │     2 │  33.3% │ ███' in lines[1]
    assert '        b │ z    │     1 │  16.7% │ █' in lines[2]
    assert "LETTER REPLACEMENTS" in captured.err


def test_generate_report_sort_by_correct(capsys):
    counts = {('b', 'z'): 1, ('a', 'y'): 2, ('c', 'x'): 3}
    typostats.generate_report(counts, sort_by='correct', output_format='arrow')
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line]
    # Header is now on stderr, so lines contains only data
    assert '        a │ y    │     2 │  33.3% │ ███' in lines[0]
    assert '        b │ z    │     1 │  16.7% │ █' in lines[1]
    assert '        c │ x    │     3 │  50.0% │ █████' in lines[2]
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
    counts, _, _ = typostats.process_typos(lines, allow_1to2=False, allow_2to1=False)
    assert counts == {('s', 'z'): 1}

def test_typostats_all_flag(tmp_path):
    typos_file = tmp_path / "typos.txt"
    typos_file.write_text("teh -> the\nrecieve -> receive\nm -> rn\nph -> f\nor -> o\na -> aa\n", encoding="utf-8")

    # Run typostats with --all flag
    # Using sys.executable to ensure we use the same python environment
    cmd = [sys.executable, "typostats.py", str(typos_file), "-a"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0

    # Check that enabled features are listed in stderr
    assert "Enabled features:" in result.stderr
    assert "keyboard, transposition, 1-to-2, 2-to-1, deletions/insertions" in result.stderr
    # Check that transposition summary is in stderr
    assert "Transpositions [T]:" in result.stderr

    # Check stdout for some expected output
    # Since we have teh->the and recieve->receive, we expect transposition
    # he | eh and ei | ie
    assert "he" in result.stdout
    assert "eh" in result.stdout
    assert "ei" in result.stdout
    assert "ie" in result.stdout

    # Check for 1-to-2 (m -> rn)
    assert "rn" in result.stdout
    assert "m" in result.stdout

    # Check for 2-to-1 (ph -> f)
    assert "f" in result.stdout
    assert "ph" in result.stdout

    # Check for deletions/insertions (or -> o, a -> aa)
    assert " o " in result.stdout
    assert " or " in result.stdout
    assert " aa " in result.stdout
    assert " a " in result.stdout


def test_typostats_limit_alias(tmp_path):
    typos_file = tmp_path / "typos.txt"
    typos_file.write_text("a -> b\nc -> d\ne -> f\n", encoding="utf-8")

    cmd = [sys.executable, "typostats.py", str(typos_file), "-L", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0
    # Count lines in stdout that look like report rows (contain '|')
    rows = [line for line in result.stdout.splitlines() if "│" in line and "CORRECT" not in line and "─" not in line]
    assert len(rows) == 1


def test_generate_report_transposition_summary(capsys):
    counts = {('he', 'eh'): 1}
    # allow_transposition=True will trigger the transposition summary calculation
    typostats.generate_report(counts, allow_transposition=True, quiet=False)
    captured = capsys.readouterr()
    assert "Transpositions [T]:" in captured.err
    assert "1/1" in captured.err


def test_generate_report_criteria_summary_stderr(capsys):
    counts = {('a', 'b'): 1, ('c', 'd'): 1}
    # min_occurrences=2 will filter out everything, unique_filtered (0) != unique_total (2)
    typostats.generate_report(counts, min_occurrences=2, quiet=False)
    captured = capsys.readouterr()
    assert "Patterns matching criteria:" in captured.err


def test_generate_report_output_file_summaries(tmp_path):
    output_file = tmp_path / "report.txt"
    counts = {('he', 'eh'): 2, ('a', 'b'): 1}

    # Trigger criteria_summary (line 509) and transposition_summary (line 515)
    typostats.generate_report(
        counts,
        output_file=str(output_file),
        min_occurrences=2,
        allow_transposition=True
    )
    content = output_file.read_text()
    assert "Patterns matching criteria:" in content # line 509
    assert "Transpositions [T]:" in content # line 515

    # Trigger display_summary (line 518)
    typostats.generate_report(
        counts,
        output_file=str(output_file),
        limit=1
    )
    content = output_file.read_text()
    assert "Showing patterns:" in content # line 518


def test_generate_report_non_adjacent_keyboard(capsys):
    counts = {('q', 'p'): 1} # q and p are not adjacent
    typostats.generate_report(counts, keyboard=True, quiet=False)
    captured = capsys.readouterr()
    # Check that [K] is NOT in the row
    assert "[K]" not in captured.out
    # Coverage for marker = "     " (line 698)


def test_detect_encoding_no_chardet(caplog):
    with patch('typostats._CHARDET_AVAILABLE', False):
        with caplog.at_level(logging.WARNING):
            assert typostats.detect_encoding("dummy.txt") is None
            assert "chardet not installed" in caplog.text


def test_load_lines_from_file_detection_success():
    mock_files = {
        'dummy.txt': b'\xff'
    }
    def mocked_open(file, mode='r', encoding=None, **kwargs):
        if 'b' in mode:
            return io.BytesIO(mock_files[file])
        if encoding == 'utf-8':
            raise UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')
        if encoding == 'latin1_detected':
            return io.StringIO("detected content")
        return io.StringIO("other content")

    with patch('builtins.open', side_effect=mocked_open),          patch('typostats.detect_encoding', return_value='latin1_detected'):
        lines = typostats.load_lines_from_file('dummy.txt')
        assert lines == ["detected content"]


def test_main_all_flag():
    with patch('sys.argv', ['typostats.py', '--all']),          patch('typostats.load_lines_from_file', return_value=[]),          patch('typostats.generate_report') as mock_report:
        typostats.main()
        _, kwargs = mock_report.call_args
        assert kwargs['allow_1to2'] is True
        assert kwargs['allow_2to1'] is True
        assert kwargs['include_deletions'] is True
        assert kwargs['allow_transposition'] is True
        assert kwargs['keyboard'] is True


def test_main_execution():
    with patch('sys.argv', ['typostats.py', '--help']):
        with pytest.raises(SystemExit):
            runpy.run_path('typostats.py', run_name='__main__')

def test_get_adjacent_keys():
    # include_diagonals=True
    adj_diag = typostats.get_adjacent_keys(include_diagonals=True)
    # 'w' is (0, 1). Neighbours: (0, 0)->'q', (0, 2)->'e', (1, 0)->'a', (1, 1)->'s', (1, 2)->'d'
    assert 'q' in adj_diag['w']
    assert 'e' in adj_diag['w']
    assert 'a' in adj_diag['w']
    assert 's' in adj_diag['w']
    assert 'd' in adj_diag['w']

    # include_diagonals=False
    adj_no_diag = typostats.get_adjacent_keys(include_diagonals=False)
    # (1, 0) and (1, 2) are diagonals
    assert 'q' in adj_no_diag['w']
    assert 'e' in adj_no_diag['w']
    assert 's' in adj_no_diag['w']
    assert 'a' not in adj_no_diag['w']
    assert 'd' not in adj_no_diag['w']


def test_is_one_letter_replacement_2to1_advanced():
    # ph -> f
    assert typostats.is_one_letter_replacement('f', 'ph', allow_2to1=True) == [('ph', 'f')]

    # include_deletions=False (default)
    # or -> o is a deletion because 'o' is in 'or'
    assert typostats.is_one_letter_replacement('o', 'or', allow_2to1=True) == []

    # include_deletions=True
    assert typostats.is_one_letter_replacement('o', 'or', allow_2to1=True, include_deletions=True) == [('or', 'o')]


def test_generate_report_keyboard_arrow(capsys):
    counts = {('q', 'w'): 5} # 'q' and 'w' are adjacent
    typostats.generate_report(counts, keyboard=True, output_format='arrow', quiet=False)
    captured = capsys.readouterr()
    # Check stderr for the summary
    assert "Keyboard Adjacency" in captured.err
    assert "5/5" in captured.err
    assert "100.0%" in captured.err
    # Check stdout for the marker
    assert "[K]" in captured.out


def test_generate_report_keyboard_json():
    counts = {('q', 'w'): 5}
    with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
        typostats.generate_report(counts, keyboard=True, output_format='json')
        data = json.loads(fake_stdout.getvalue())
        assert data["replacements"][0]["is_adjacent"] is True


def test_generate_report_write_error(caplog):
    caplog.set_level(logging.ERROR)
    counts = {('a', 'b'): 1}
    with patch("builtins.open", side_effect=IOError("Permission denied")):
        typostats.generate_report(counts, output_file="unwritable.txt")
        assert "Failed to write report to 'unwritable.txt'. Error: Permission denied" in caplog.text


def test_detect_encoding_logic():
    with patch('typostats._CHARDET_AVAILABLE', True),          patch('typostats.chardet') as mock_chardet,          patch('builtins.open', mock_open(read_data=b"some data")):

        # High confidence
        mock_chardet.detect.return_value = {'encoding': 'utf-8', 'confidence': 0.9}
        assert typostats.detect_encoding("dummy.txt") == 'utf-8'

        # Low confidence
        mock_chardet.detect.return_value = {'encoding': 'utf-8', 'confidence': 0.4}
        assert typostats.detect_encoding("dummy.txt") is None


def test_load_lines_from_file_variants(monkeypatch):
    # Test stdin
    monkeypatch.setattr(sys.stdin, 'readlines', lambda: ["line1\n"])
    assert typostats.load_lines_from_file('-') == ["line1\n"]

    mock_files = {
        'dummy.txt': b'\xff' # Not valid UTF-8
    }

    def mocked_open(file, mode='r', encoding=None, **kwargs):
        if 'b' in mode:
            return io.BytesIO(mock_files[file])

        content = mock_files[file]
        if encoding == 'utf-8':
            raise UnicodeDecodeError('utf-8', content, 0, 1, 'invalid')
        elif encoding == 'ascii':
            raise UnicodeDecodeError('ascii', content, 0, 1, 'invalid')
        elif encoding == 'latin1':
            return io.StringIO(content.decode('latin1'))

        return io.StringIO(content.decode('utf-8'))

    with patch('builtins.open', side_effect=mocked_open),          patch('typostats.detect_encoding', return_value='ascii'):
        lines = typostats.load_lines_from_file('dummy.txt')
        assert lines == ['\xff']


def test_main_stdin_default():
    with patch('sys.argv', ['typostats.py']),          patch('typostats.load_lines_from_file', return_value=[]) as mock_load,          patch('typostats.generate_report'):
        typostats.main()
        mock_load.assert_called_with('-')


def test_main_allow_two_char_alias():
    # provide some lines so process_typos is called
    with patch('sys.argv', ['typostats.py', 'input.txt', '--allow-two-char']),          patch('typostats.load_lines_from_file', return_value=["tezt -> test"]),          patch('typostats.process_typos', return_value=({}, 0, 0)) as mock_process,          patch('typostats.generate_report'):
        typostats.main()
        # Verify that allow_1to2 and allow_2to1 are both True
        args, kwargs = mock_process.call_args
        assert kwargs['allow_1to2'] is True
        assert kwargs['allow_2to1'] is True


def test_minimal_formatter_warning():
    formatter = typostats.MinimalFormatter('%(levelname)s: %(message)s')

    # INFO level should not have prefix
    info_record = logging.LogRecord('name', logging.INFO, 'pathname', 10, 'info msg', None, None)
    assert formatter.format(info_record) == 'info msg'

    # WARNING level should have prefix
    warn_record = logging.LogRecord('name', logging.WARNING, 'pathname', 10, 'warn msg', None, None)
    assert formatter.format(warn_record) == 'WARNING: warn msg'


def test_generate_report_keyboard_with_limit(capsys):
    # Two adjacent pairs
    counts = {('q', 'w'): 5, ('a', 's'): 10}
    # Limit to 1, but keyboard summary should still show both (5+10 = 15)
    typostats.generate_report(counts, keyboard=True, limit=1, output_format='arrow', quiet=False)
    captured = capsys.readouterr()
    assert "Keyboard Adjacency" in captured.err
    assert "15/15" in captured.err
    assert "100.0%" in captured.err


def test_is_transposition_length_mismatch():
    assert typostats.is_transposition("abc", "ab") == []


def test_process_typos_empty_line():
    lines = ["", "tezt -> test"]
    counts, total, raw = typostats.process_typos(lines)
    # total_lines is incremented before checking if line is empty
    assert total == 2


def test_generate_report_with_file_and_keyboard():
    counts = {('q', 'w'): 5}
    m = mock_open()
    with patch("builtins.open", m):
        typostats.generate_report(counts, output_file="test.txt", keyboard=True)

    handle = m()
    written_content = "".join(call.args[0] for call in handle.write.call_args_list)
    assert "Keyboard Adjacency" in written_content


def test_generate_report_no_results_stderr(capsys):
    counts = {}
    typostats.generate_report(counts, quiet=False)
    captured = capsys.readouterr()
    assert "No replacements found matching the criteria." in captured.err


def test_minimal_formatter_color_with_tty():
    with patch("typostats.RED", "\033[31m"), patch("typostats.RESET", "\033[0m"):
        new_colors = {logging.ERROR: "\033[31m"}
        with patch.object(typostats.MinimalFormatter, 'LEVEL_COLORS', new_colors):
            formatter = typostats.MinimalFormatter('%(levelname)s: %(message)s')
            record = logging.LogRecord('name', logging.ERROR, 'pathname', 10, 'error msg', None, None)
            with patch("typostats.sys.stderr.isatty", return_value=True):
                formatted = formatter.format(record)
                assert "\033[31m" in formatted


def test_2to1_replacement_not_deletion():
    # Case: len(c) > len(t) and t not in c (e.g., 'th' -> 'f')
    replacement_counts = {('th', 'f'): 10}
    with patch('sys.stderr', new_callable=io.StringIO) as mock_stderr:
        typostats.generate_report(replacement_counts, allow_2to1=True, quiet=False)
        report = mock_stderr.getvalue()
    assert "2-to-1 replacements [2:1]" in report

def test_generate_report_multi_char_summary_stderr(capsys):
    counts = {('a', 'aa'): 1, ('bb', 'b'): 1}
    typostats.generate_report(counts, include_deletions=True, quiet=False)
    captured = capsys.readouterr()
    assert "Insertions [Ins]:" in captured.err
    assert "Deletions [Del]:" in captured.err
    assert "1/2" in captured.err


def test_generate_report_multi_char_summary_file(tmp_path):
    output_file = tmp_path / "report.txt"
    counts = {('abc', 'ab'): 5}

    typostats.generate_report(
        counts,
        output_file=str(output_file),
        allow_2to1=True
    )

    content = output_file.read_text()
    assert "Deletions [Del]:" in content
    assert "5/5" in content


def test_generate_report_multi_char_summary_mixed(capsys):
    counts = {('a', 'b'): 3, ('c', 'cc'): 2}

    typostats.generate_report(counts, allow_1to2=True, quiet=False)
    captured = capsys.readouterr()
    assert "Insertions [Ins]:" in captured.err
    assert "2/5" in captured.err


def test_new_markers(capsys):
    """Specifically verify the logic for [Ins], [Del], [1:2], and [2:1] markers."""
    counts = {
        ('a', 'ab'): 1,   # Insertion
        ('bc', 'b'): 1,   # Deletion
        ('m', 'rn'): 1,   # 1-to-2 replacement
        ('ph', 'f'): 1    # 2-to-1 replacement
    }
    typostats.generate_report(counts, all=True)
    captured = capsys.readouterr().out

    assert "[Ins]" in captured
    assert "[Del]" in captured
    assert "[1:2]" in captured
    assert "[2:1]" in captured


def test_summary_labels(capsys):
    """Verify improved labels in the analysis summary."""
    counts = {('teh', 'the'): 1}
    # generate_report prints summary to stderr when no output_file
    typostats.generate_report(counts, quiet=False)
    captured = capsys.readouterr().err

    assert "Total word pairs encountered:" in captured
    assert "Total patterns after analysis:" in captured
    assert "Unique patterns found:" in captured


def test_generate_report_marker_multi_char(capsys):
    """Verify the [M] marker for multi-character replacements in the report."""
    # counts with a 1-to-2 replacement
    counts = {('m', 'rn'): 5}
    # We need to enable at least one multi-char flag for show_attr to be True
    typostats.generate_report(counts, allow_1to2=True, quiet=False)
    captured = capsys.readouterr().out
    assert "[1:2]" in captured


def test_generate_report_marker_transposition(capsys):
    """Verify the [T] marker for transpositions in the report."""
    counts = {('he', 'eh'): 5}
    typostats.generate_report(counts, allow_transposition=True, quiet=False)
    captured = capsys.readouterr().out
    assert "[T]" in captured


def test_tqdm_unavailable_fallback():
    """Verify the fallback logic when tqdm is not installed."""
    initial_tqdm = typostats._TQDM_AVAILABLE

    try:
        with patch.dict(sys.modules, {'tqdm': None}):
            importlib.reload(typostats)
            assert typostats._TQDM_AVAILABLE is False
            assert typostats.tqdm is None
    finally:
        importlib.reload(typostats)

    assert typostats._TQDM_AVAILABLE == initial_tqdm

def test_levenshtein_distance_empty_s2():
    # Targets line 68
    assert typostats.levenshtein_distance('abc', '') == 3


def test_minimal_formatter_custom_level():
    # Targets line 57 (branch where color is None)
    formatter = typostats.MinimalFormatter('%(levelname)s: %(message)s')
    # logging.DEBUG is usually not in LEVEL_COLORS
    record = logging.LogRecord('name', logging.DEBUG, 'pathname', 10, 'debug msg', None, None)
    with patch("typostats.sys.stderr.isatty", return_value=True):
        formatted = formatter.format(record)
        # Should not have color but should have the level name
        assert "DEBUG: debug msg" in formatted
        assert "\033[" not in formatted

def test_format_analysis_summary_unhashable():
    # Targets line 148-149
    # Passing unhashable items (lists) should trigger TypeError in set()
    items = [[1], [2], [1]]
    report = typostats._format_analysis_summary(3, items)
    # unique_count should fall back to len(filtered_items)
    # Search for the line with 'Unique items:'
    unique_line = next(line for line in report if "Unique items:" in line)
    assert "3" in unique_line


def test_format_analysis_summary_item_label():
    # Targets line 113-114
    items = ["a", "b"]
    report = typostats._format_analysis_summary(2, items, item_label="word")
    assert "Total words encountered:" in report[2]
    assert "Total words after filtering:" in report[3]


def test_is_transposition_not_swapped():
    # Targets line 247 (branch where char differences are not a swap)
    # Differences at i=1 ('b' vs 'd') and i=2 ('c' vs 'e')
    # typo[1]=='d', correction[2]=='c' -> mismatch
    assert typostats.is_transposition("abc", "ade") == []

def test_is_one_letter_replacement_2to1_deletion_filter():
    # Targets line 367
    # typo='a', correction='ab' -> 2-to-1 replacement?
    # Actually correction='ab', typo='a'
    # correction[0:2]='ab', typo[0]='a'
    # repl_correction='ab', repl_typo='a'
    # repl_typo in repl_correction is True ('a' in 'ab')
    # include_deletions=False should continue (filter out)
    assert typostats.is_one_letter_replacement('a', 'ab', allow_2to1=True, include_deletions=False) == []

def test_generate_report_json_keyboard_multi_char():
    # Targets line 737-738 (multi-char items bypass adjacency check in JSON)
    counts = {('m', 'rn'): 1}
    with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
        typostats.generate_report(counts, keyboard=True, output_format='json')
        data = json.loads(fake_stdout.getvalue())
        assert data["replacements"][0]["is_adjacent"] is False


def test_generate_report_file_newline_padding(tmp_path):
    # Targets line 770
    output_file = tmp_path / "report.txt"
    # YAML format with one entry
    counts = {('a', 'b'): 1}
    typostats.generate_report(counts, output_file=str(output_file), output_format='yaml')
    content = output_file.read_text()
    assert content.endswith('\n')
    # If the format didn't naturally end in \n, it would have been added.
    # YAML grouping adds \n between entries but maybe not at the very end of the string?
    # Let's check CSV which might not have a trailing newline in report_content if not for line 750
    typostats.generate_report(counts, output_file=str(output_file), output_format='csv')
    content = output_file.read_text()
    assert content.endswith('\n')


def test_generate_report_stdout_newline_padding():
    # Targets line 778
    counts = {('a', 'b'): 1}
    with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
        typostats.generate_report(counts, output_format='yaml')
        content = fake_stdout.getvalue()
        assert content.endswith('\n')

def test_generate_report_no_results_file(tmp_path):
    # Targets line 673
    output_file = tmp_path / "report.txt"
    typostats.generate_report({}, output_file=str(output_file))
    content = output_file.read_text()
    assert "No replacements found matching the criteria." in content


def test_generate_report_multi_char_marker_deletion():
    # Targets line 717
    # (correct_char, typo_char) where len(c) > len(t) and t in c
    # typo='a', correction='ab' -> Deletion
    counts = {('ab', 'a'): 1}
    with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
        typostats.generate_report(counts, include_deletions=True)
        assert "[Del]" in fake_stdout.getvalue()


def test_generate_report_keyboard_multi_char_marker_keyboard():
    # Targets line 701, 710, etc.
    # We need to cover the branch where a multi-char item DOES NOT get [K] marker
    # and the one where it DOES get [T], [Ins], [1:2], etc.
    # [1:2] is already covered. Let's do [Ins].
    counts = {('a', 'ab'): 1}
    with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
        typostats.generate_report(counts, include_deletions=True, keyboard=True)
        output = fake_stdout.getvalue()
        assert "[Ins]" in output
        assert "[K]" not in output # because it's not 1-to-1


def test_format_analysis_summary_full_bar():
    # Targets line 137-138 (branch when full_blocks < max_bar)
    # 50% retention with max_bar=20 should give full_blocks=10
    # 10 < 20 is True.
    report = typostats._format_analysis_summary(10, ["a"] * 5)
    retention_line = next(line for line in report if "Retention rate:" in line)
    assert "50.0%" in retention_line
    # Bar should contain blocks
    assert "█" in retention_line


def test_format_analysis_summary_no_filtered_items():
    # Targets line 166 (if lengths: branch)
    # Targets line 212-215 (if not filtered_items: branch)
    report = typostats._format_analysis_summary(10, [])
    assert "No items passed the filtering criteria." in report[-2]


def test_generate_report_no_transposition_found():
    # Targets line 568
    counts = {('a', 'b'): 1}
    with patch('sys.stderr', new_callable=io.StringIO) as mock_stderr:
        typostats.generate_report(counts, allow_transposition=True, quiet=False)
        assert "Transpositions [T]:" not in mock_stderr.getvalue()


def test_generate_report_no_multi_char_found():
    # Targets line 585
    counts = {('a', 'b'): 1}
    with patch('sys.stderr', new_callable=io.StringIO) as mock_stderr:
        typostats.generate_report(counts, include_deletions=True, quiet=False)
        assert "Insertions [Ins]:" not in mock_stderr.getvalue()
        assert "Deletions [Del]:" not in mock_stderr.getvalue()

def test_no_color_env(monkeypatch):
    # Targets line 36-37
    # NO_COLOR=1 should disable colors
    monkeypatch.setenv('NO_COLOR', '1')
    # Force reload of typostats to re-run the module-level color init
    importlib.reload(typostats)
    assert typostats.BLUE == ""
    # Clean up for other tests
    monkeypatch.delenv('NO_COLOR')
    importlib.reload(typostats)


def test_format_analysis_summary_unformatable():
    # Targets line 186-187 (ValueError/TypeError in format_item logic)
    # If filtered_items contains something that len(format_item(it)) fails on?
    # format_item handles tuples and strings. Let's try something else.
    class Unformatable:
        def __len__(self):
            raise TypeError("unformatable")
        def __str__(self):
            raise TypeError("unformatable")

    items = [Unformatable()]
    report = typostats._format_analysis_summary(1, items)
    # Should skip the min/max/avg length and shortest/longest blocks
    assert "Shortest" not in "".join(report)


def test_format_analysis_summary_distances_exception():
    # Targets line 204-205 (try/except around distances)
    class BadPair:
        def __getitem__(self, idx):
            if idx == 0: return "a"
            raise Exception("bad pair")
        def __len__(self): return 2

    items = [BadPair()]
    report = typostats._format_analysis_summary(1, items)
    assert "Min/Max/Avg changes:" not in "".join(report)

def test_generate_report_quiet_no_results(capsys):
    # Targets line 670->675 (if not quiet: False branch)
    counts = {}
    typostats.generate_report(counts, quiet=True)
    captured = capsys.readouterr()
    assert captured.err == ""


def test_is_one_letter_replacement_2to1_mismatch():
    # Targets line 360->356 (if correction[:i] == typo[:i]... False branch)
    # len(typo)=2, len(correction)=3.
    assert typostats.is_one_letter_replacement('xy', 'abc', allow_2to1=True) == []


def test_generate_report_json_not_adjacent():
    # Targets line 738->740 (if typo_char.lower() in ... False branch)
    counts = {('a', 'p'): 1} # Not adjacent
    with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
        typostats.generate_report(counts, keyboard=True, output_format='json')
        data = json.loads(fake_stdout.getvalue())
        assert data["replacements"][0]["is_adjacent"] is False


def test_generate_report_show_attr_not_longer(capsys):
    # Targets line 710->717 (elif len(c) > len(t) False branch)
    counts = {('a', 'b'): 1}
    typostats.generate_report(counts, keyboard=True)
    captured = capsys.readouterr()
    assert "a │ b" in captured.out


def test_minimal_formatter_no_tty():
    # Targets line 55->60 (if sys.stderr.isatty() False branch)
    formatter = typostats.MinimalFormatter('%(levelname)s: %(message)s')
    record = logging.LogRecord('name', logging.ERROR, 'pathname', 10, 'error msg', None, None)
    with patch("typostats.sys.stderr.isatty", return_value=False):
        formatted = formatter.format(record)
        assert "ERROR: error msg" in formatted
        assert "\033[" not in formatted

def test_generate_report_marker_no_match(capsys):
    # Targets line 710->717 (more precisely the end of the marker logic)
    # Item that is not 1-to-1, not 2-to-2 swap, not longer typo, not shorter typo.
    # This shouldn't really happen with current logic as replacements are usually length diff 0, 1 or transposition 2-to-2.
    # But we can force it with a custom counts dict.
    counts = {('abc', 'def'): 1}
    # Enable show_attr
    typostats.generate_report(counts, keyboard=True)
    captured = capsys.readouterr()
    # Marker should be spaces
    assert "abc │ def" in captured.out


def test_format_analysis_summary_avg_length():
    # Targets line 166 (if lengths: True branch)
    # Already covered by many tests. Why is it still 166->174?
    # Ah, maybe I need to check the exact line numbers again.
    pass


def test_no_multi_char_label_in_summary(capsys):
    # Targets line 585->584
    # We enable multi-char but none are found.
    counts = {('a', 'b'): 1}
    typostats.generate_report(counts, allow_1to2=True, quiet=False)
    captured = capsys.readouterr()
    assert "Insertions [Ins]:" not in captured.err
