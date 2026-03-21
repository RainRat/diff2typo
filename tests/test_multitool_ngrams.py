import pytest
from multitool import _extract_ngram_items, ngrams_mode
import io
import os

def test_extract_ngram_items_basic(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("the quick brown fox", encoding="utf-8")

    # Bigrams
    results = list(_extract_ngram_items(str(f), n=2, clean_items=False))
    assert results == ["the quick", "quick brown", "brown fox"]

    # Trigrams
    results = list(_extract_ngram_items(str(f), n=3, clean_items=False))
    assert results == ["the quick brown", "quick brown fox"]

def test_extract_ngram_items_cleaning(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("The quick, brown fox!", encoding="utf-8")

    # With cleaning (default)
    results = list(_extract_ngram_items(str(f), n=2, clean_items=True))
    assert results == ["the quick", "quick brown", "brown fox"]

def test_extract_ngram_items_cross_lines(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("the quick\nbrown fox", encoding="utf-8")

    results = list(_extract_ngram_items(str(f), n=2, clean_items=False))
    assert results == ["the quick", "quick brown", "brown fox"]

def test_ngrams_mode_output(tmp_path, capsys):
    f = tmp_path / "test.txt"
    f.write_text("one two three four", encoding="utf-8")
    output = tmp_path / "output.txt"

    ngrams_mode(
        input_files=[str(f)],
        output_file=str(output),
        min_length=1,
        max_length=100,
        process_output=False,
        n=2,
        clean_items=False
    )

    assert output.read_text(encoding="utf-8").strip().split('\n') == ["one two", "two three", "three four"]

def test_extract_ngram_items_smart_split(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("CamelCase snake_case", encoding="utf-8")

    results = list(_extract_ngram_items(str(f), n=2, smart=True, clean_items=True))
    # CamelCase -> camel case
    # snake_case -> snake case
    # stream: camel, case, snake, case
    assert results == ["camel case", "case snake", "snake case"]
