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


def test_lowercase_sort_dedup():
    items = ['Banana', 'apple', 'banana']
    assert diff2typo.lowercase_sort_dedup(items) == ['apple', 'banana']


def test_format_typos():
    typos = ['teh -> the']
    assert diff2typo.format_typos(typos, 'arrow') == ['teh -> the']
    assert diff2typo.format_typos(typos, 'csv') == ['teh,the']
    assert diff2typo.format_typos(typos, 'table') == ['teh = "the"']
    assert diff2typo.format_typos(typos, 'list') == ['teh']


def test_process_new_typos(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    allowed = tmp_path / 'allowed.csv'
    allowed.write_text('teh\n')
    args = SimpleNamespace(typos_tool_path='nonexistent', allowed_file=str(allowed), output_format='arrow')
    candidates = ['mispell -> misspell', 'teh -> the', 'recieve -> receive', 'recieve -> receive']
    result = diff2typo.process_new_typos(candidates, args, {'mispell'})
    assert result == ['recieve -> receive']


def test_process_new_corrections():
    words_mapping = {'teh': {'the'}, 'mispell': {'misspell'}}
    candidates = ['teh -> the', 'teh -> thee', 'recieve -> receive']
    result = diff2typo.process_new_corrections(candidates, SimpleNamespace(), words_mapping, 'arrow')
    assert result == ['teh -> thee']
