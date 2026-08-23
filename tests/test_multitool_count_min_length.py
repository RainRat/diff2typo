import contextlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool


def _get_mode_kwargs(cli_args, target_mode="multitool.count_mode"):
    with patch("sys.argv", cli_args), patch(target_mode) as mock_func:
        with contextlib.suppress(SystemExit):
            multitool.main()
        _, kwargs = mock_func.call_args
        return kwargs


@pytest.mark.parametrize(
    "cli_args, expected_min_length, target_mode",
    [
        (["multitool.py", "count", "dummy_input"], 3, "multitool.count_mode"),
        (["multitool.py", "count", "dummy_input", "--pairs"], 1, "multitool.count_mode"),
        (["multitool.py", "count", "dummy_input", "--chars"], 1, "multitool.count_mode"),
        (["multitool.py", "count", "dummy_input", "--lines"], 1, "multitool.count_mode"),
        (["multitool.py", "count", "dummy_input", "--add", "a:b"], 1, "multitool.count_mode"),
        (["multitool.py", "count", "dummy_input", "--mapping", "dummy_mapping"], 1, "multitool.count_mode"),
        (["multitool.py", "words", "dummy_input"], 3, "multitool.words_mode"),
        (["multitool.py", "search", "dummy_input", "--query", "abc"], 1, "multitool.search_mode"),
        (["multitool.py", "ngrams", "dummy_input"], 3, "multitool.ngrams_mode"),
        (["multitool.py", "stats", "dummy_input"], 3, "multitool.stats_mode"),
        (["multitool.py", "count", "dummy_input", "--min-length", "5"], 5, "multitool.count_mode"),
        (["multitool.py", "count", "dummy_input", "--chars", "--min-length", "3"], 3, "multitool.count_mode"),
    ],
)
def test_min_length_defaults_in_count_mode(tmp_path, cli_args, expected_min_length, target_mode):
    dummy_input = tmp_path / "input.txt"
    dummy_input.write_text("a b c d e")
    dummy_mapping = tmp_path / "mapping.csv"
    dummy_mapping.write_text("a,b")

    formatted_args = [
        arg if arg != "dummy_input" else str(dummy_input)
        for arg in cli_args
    ]
    formatted_args = [
        arg if arg != "dummy_mapping" else str(dummy_mapping)
        for arg in formatted_args
    ]

    kwargs = _get_mode_kwargs(formatted_args, target_mode)
    assert kwargs["min_length"] == expected_min_length
