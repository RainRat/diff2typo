import io
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import diff2typo


def test_filter_to_letters():
    assert diff2typo.filter_to_letters('Hello, World!123') == 'helloworld'


def test_extract_backticks():
    text = 'use `code` and `more` but `x` is ignored'
    assert diff2typo.extract_backticks(text) == ['code', 'more']


def test_read_allowed_words(tmp_path):
    allowed_file = tmp_path / 'allowed.csv'
    allowed_file.write_text('Foo\nBar\n')
    assert diff2typo.read_allowed_words(str(allowed_file)) == {'foo', 'bar'}


def test_split_into_subwords():
    assert diff2typo.split_into_subwords('camelCase_word') == ['camel', 'Case', 'word']


def test_read_words_mapping(tmp_path):
    mapping_file = tmp_path / 'words.csv'
    mapping_file.write_text('typo,correct1,correct2\nvalid\n')
    mapping = diff2typo.read_words_mapping(str(mapping_file))
    assert mapping == {'typo': {'correct1', 'correct2'}, 'valid': set()}


def test_find_typos():
    diff_text = ('--- a/f.txt\n+++ b/f.txt\n@@\n-This line has an eror in it\n'
                 '+This line has an error in it\n unchanged\n')
    assert diff2typo.find_typos(diff_text) == ['eror -> error']


def test_find_typos_multiline_reflow():
    diff_text = (
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@\n"
        "-This is a long eror line.\n"
        "+This is a long\n"
        "+error line.\n"
    )
    assert diff2typo.find_typos(diff_text) == ['eror -> error']


def test_find_typos_skips_unpaired_changes():
    diff_text = (
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@\n"
        "-Removed typo line\n"
        " context\n"
        "@@\n"
        "+Added typo line\n"
        " context\n"
    )
    assert diff2typo.find_typos(diff_text) == []


def test_find_typos_large_input():
    diff_lines = ["--- a/f.txt", "+++ b/f.txt", "@@"]
    for index in range(200):
        diff_lines.append(f"-Repeated eror line {index}")
        diff_lines.append(f"+Repeated error line {index}")
    diff_lines.append(" context")
    diff_text = "\n".join(diff_lines)
    result = diff2typo.find_typos(diff_text)
    assert len(result) == 200
    assert result[0] == 'eror -> error'


def test_validate_adjacent_context():
    before = ['a', 'eror', 'line']
    after = ['a', 'error', 'line']
    assert diff2typo._validate_adjacent_context(before, after, 1)
    after_mismatch = ['a', 'error', 'change']
    assert not diff2typo._validate_adjacent_context(before, after_mismatch, 1)


def test_compare_word_lists():
    before_words = ['This', 'eror', 'line']
    after_words = ['This', 'error', 'line']
    assert diff2typo._compare_word_lists(before_words, after_words, 2) == ['eror -> error']
    assert diff2typo._compare_word_lists(['foo'], ['foo', 'bar'], 2) == []


def test_lowercase_sort_dedup():
    items = ['Banana', 'apple', 'banana']
    assert diff2typo.lowercase_sort_dedup(items) == ['apple', 'banana']


def test_format_typos():
    typos = ['teh -> the']
    assert diff2typo.format_typos(typos, 'arrow') == ['teh -> the']
    assert diff2typo.format_typos(typos, 'csv') == ['teh,the']
    assert diff2typo.format_typos(typos, 'table') == ['teh = "the"']
    assert diff2typo.format_typos(typos, 'list') == ['teh']


def test_filter_known_typos(monkeypatch, tmp_path):
    candidates = ['eror -> error', 'typo -> type']

    # Mock subprocess.run to simulate the behavior of the typos tool
    def mock_run(*args, **kwargs):
        return SimpleNamespace(stdout='Found `eror` typo.')

    monkeypatch.setattr('subprocess.run', mock_run)

    # Create a dummy typos tool executable for path checking
    typos_tool = tmp_path / 'typos'
    typos_tool.touch()

    result = diff2typo.filter_known_typos(candidates, str(typos_tool))
    assert result == ['typo -> type']

def test_filter_allowed_words():
    candidates = ['teh -> the', 'mispell -> misspell']
    allowed_words = {'teh'}
    result = diff2typo.filter_allowed_words(candidates, allowed_words, quiet=True)
    assert result == ['mispell -> misspell']

def test_filter_dictionary_words():
    candidates = ['fluro -> fluoro', 'wierd -> weird']
    valid_words = {'wierd'}
    result = diff2typo.filter_dictionary_words(candidates, valid_words, quiet=True)
    assert result == ['fluro -> fluoro']

def test_process_new_typos(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    allowed = tmp_path / 'allowed.csv'
    allowed.write_text('teh\n')
    args = SimpleNamespace(typos_tool_path='nonexistent', allowed_file=str(allowed), output_format='arrow', quiet=True)
    candidates = ['mispell -> misspell', 'teh -> the', 'recieve -> receive', 'recieve -> receive']
    result = diff2typo.process_new_typos(candidates, args, {'mispell'})
    assert result == ['recieve -> receive']


def test_process_new_corrections():
    words_mapping = {'teh': {'the'}, 'mispell': {'misspell'}}
    candidates = ['teh -> the', 'teh -> thee', 'recieve -> receive']
    result = diff2typo.process_new_corrections(candidates, words_mapping, quiet=True)
    assert result == ['teh -> thee']


def test_process_new_corrections_dedup_and_sort():
    words_mapping = {'teh': {'the'}}
    candidates = ['teh -> thee', 'Teh -> THEE', 'teh -> thea']
    result = diff2typo.process_new_corrections(candidates, words_mapping, quiet=True)
    assert result == ['teh -> thea', 'teh -> thee']


def test_temp_typo_file_cleanup(tmp_path):
    with diff2typo.TempTypoFile() as temp_path:
        path_obj = Path(temp_path)
        path_obj.write_text('data')
        assert path_obj.exists()
    assert not Path(temp_path).exists()


def test_read_words_mapping_file_not_found(tmp_path):
    with pytest.raises(SystemExit):
        diff2typo.read_words_mapping(str(tmp_path / 'missing.csv'))


def test_read_allowed_words_logs_warning(tmp_path, caplog):
    missing_file = tmp_path / 'missing_allowed.csv'

    with caplog.at_level(logging.WARNING):
        allowed = diff2typo.read_allowed_words(str(missing_file))

    assert allowed == set()
    assert any('Allowed words file' in message for message in caplog.messages)


def test_filter_known_typos_tool_not_found(tmp_path, caplog):
    candidates = ['eror -> error']

    with caplog.at_level(logging.WARNING):
        result = diff2typo.filter_known_typos(candidates, str(tmp_path / 'missing_tool'))

    assert result == candidates
    assert any('Typos tool' in message for message in caplog.messages)


def test_main_integration_success(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    diff_file = tmp_path / 'diff.txt'
    diff_file.write_text('--- a/file\n+++ b/file\n@@\n-teh word\n+the word\n')

    dictionary_file = tmp_path / 'words.csv'
    dictionary_file.write_text('valid\n')

    allowed_file = tmp_path / 'allowed.csv'
    allowed_file.write_text('')

    output_file = tmp_path / 'output.txt'

    typos_tool = tmp_path / 'typos'
    typos_tool.write_text('#!/usr/bin/env python3\nimport sys\nprint("")\n')
    typos_tool.chmod(0o755)

    def fake_run(*_, **__):
        return SimpleNamespace(stdout='')

    monkeypatch.setattr(diff2typo.subprocess, 'run', fake_run)

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'diff2typo.py',
            '--input_file',
            str(diff_file),
            '--output_file',
            str(output_file),
            '--dictionary_file',
            str(dictionary_file),
            '--allowed_file',
            str(allowed_file),
            '--typos_tool_path',
            str(typos_tool),
            '--output_format',
            'arrow',
            '--quiet',
        ],
    )

    diff2typo.main()

    assert output_file.read_text().strip().splitlines() == ['teh -> the']


def test_main_input_file_not_found(monkeypatch, tmp_path):
    dictionary_file = tmp_path / 'words.csv'
    dictionary_file.write_text('valid\n')

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'diff2typo.py',
            '--input_file',
            str(tmp_path / 'missing.diff'),
            '--dictionary_file',
            str(dictionary_file),
        ],
    )

    with pytest.raises(SystemExit):
        diff2typo.main()


def test_main_reads_stdin(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    dictionary_file = tmp_path / 'words.csv'
    dictionary_file.write_text('valid\n')

    allowed_file = tmp_path / 'allowed.csv'
    allowed_file.write_text('')

    output_file = tmp_path / 'output.txt'

    diff_text = '--- a/file\n+++ b/file\n@@\n-teh value\n+the value\n'

    monkeypatch.setattr(sys, 'stdin', io.StringIO(diff_text))
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'diff2typo.py',
            '--input_file',
            '-',
            '--output_file',
            str(output_file),
            '--dictionary_file',
            str(dictionary_file),
            '--allowed_file',
            str(allowed_file),
            '--typos_tool_path',
            str(tmp_path / 'missing_tool'),
            '--output_format',
            'arrow',
            '--quiet',
        ],
    )

    diff2typo.main()

    assert output_file.read_text().strip().splitlines() == ['teh -> the']
