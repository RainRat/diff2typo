import sys
import os
import pytest
import runpy
import yaml
from unittest.mock import patch
import cmdrunner
from diff2typo import _compare_word_lists

def test_cmdrunner_main_folder_type_check():
    config = {
        'main_folder': 123,
        'command_to_run': 'echo test'
    }
    with patch("builtins.open", create=True):
        with patch("yaml.safe_load", return_value=config):
            with pytest.raises(cmdrunner.ConfigError, match="'main_folder' must be a string"):
                cmdrunner.load_config("fake_path.yaml")

def test_diff2typo_identical_words_in_replace():
    before = ['the', 'teh']
    after = ['the', 'the']
    result = _compare_word_lists(before, after, min_length=2)
    assert "teh -> the" in result

def test_diff2typo_short_words_in_unequal_replace():
    before = ['a', 'house']
    after = ['the', 'big', 'house']
    result = _compare_word_lists(before, after, min_length=4)
    assert len(result) == 0

def test_main_entry_points():
    with patch("sys.argv", ["diff2typo.py", "--help"]):
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("diff2typo", run_name="__main__")
        assert excinfo.value.code == 0

    with patch("sys.argv", ["cmdrunner.py", "--help"]):
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("cmdrunner", run_name="__main__")
        assert excinfo.value.code == 0
