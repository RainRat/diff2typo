"""
Tests for batch loading allowed words from multiple files, directories, and glob patterns in diff2typo.py.
"""

import os
import pytest
import diff2typo


def test_read_allowed_words_multiple_files(tmp_path):
    file1 = tmp_path / "allowed1.csv"
    file2 = tmp_path / "allowed2.csv"

    file1.write_text("word1\nword2\n", encoding="utf-8")
    file2.write_text("word2\nword3\n", encoding="utf-8")

    allowed = diff2typo.read_allowed_words([str(file1), str(file2)])
    assert allowed == {"word1", "word2", "word3"}


def test_read_allowed_words_directory_and_glob(tmp_path):
    dir_path = tmp_path / "allowed_dir"
    dir_path.mkdir()

    f1 = dir_path / "a.csv"
    f2 = dir_path / "b.csv"
    f1.write_text("alpha\nbeta\n", encoding="utf-8")
    f2.write_text("gamma\n", encoding="utf-8")

    glob_file = tmp_path / "extra_allowed.csv"
    glob_file.write_text("delta\n", encoding="utf-8")

    glob_pattern = str(tmp_path / "extra_*.csv")

    allowed = diff2typo.read_allowed_words([str(dir_path), glob_pattern])
    assert allowed == {"alpha", "beta", "gamma", "delta"}


def test_read_allowed_words_missing_and_empty(tmp_path):
    missing_file = tmp_path / "missing.csv"
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")

    allowed = diff2typo.read_allowed_words([str(missing_file), str(empty_file)])
    assert allowed == set()
