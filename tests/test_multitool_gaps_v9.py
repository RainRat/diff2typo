import os
import sys
import argparse
import logging
from unittest.mock import MagicMock, patch
from pathlib import Path
import multitool

def test_unzip_mode(tmp_path):
    test_file = tmp_path / "pairs.txt"
    test_file.write_text("a -> b\nc -> d\n")
    output_file = tmp_path / "output.txt"

    # Test left side (default)
    multitool.unzip_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        right_side=False
    )
    assert output_file.read_text().strip().splitlines() == ["a", "c"]

    # Test right side
    multitool.unzip_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        right_side=True
    )
    assert output_file.read_text().strip().splitlines() == ["b", "d"]

def test_format_size_extra():
    # Line 117: size_bytes < 0
    assert multitool._format_size(-1) == "-1"

    # Line 119: size_bytes == 0
    assert multitool._format_size(0) == "0 B"

    # Line 123: integer result after division
    assert multitool._format_size(2048) == "2 KB"

    # Line 124: float result (size_bytes != int(size_bytes))
    # 1.5 KB = 1.5 * 1024 = 1536
    assert multitool._format_size(1536) == "1.5 KB"

    # Line 126: EB scale
    eb_size = 1024**6 * 2.5
    assert "EB" in multitool._format_size(int(eb_size))

def test_count_paragraphs_coverage(tmp_path, monkeypatch):
    test_file = tmp_path / "paragraphs.txt"
    test_file.write_text("Para one.\n\nPara two.\n")
    output_file = tmp_path / "out.txt"

    # Test Line 3531 (paragraph label) and 9827 (count mode paragraphs default min_length 20)
    test_args = ["multitool.py", "count", str(test_file), "--paragraphs", "-o", str(output_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch("sys.stdout.isatty", return_value=False):
        multitool.main()

    content = output_file.read_text()
    assert "Para one." not in content
    assert "Para two." not in content

def test_min_length_defaults(tmp_path, monkeypatch):
    # Test 9817 (words -> 3), 9819 (sentences -> 10), 9821 (paragraphs -> 20)
    # Also 9825 (count sentences -> 10), 9832 (count words -> 3), 9835 (other -> 1)

    scenarios = [
        ('words', "hi", 3, False), # hi is 2, should be skipped
        ('sentences', "Short.", 10, False), # "Short." is 6, should be skipped
        ('paragraphs', "Short paragraph.", 20, False), # 16 chars, should be skipped
        ('count', "hi", 3, False), # default count is words
        ('line', "a", 1, True), # default line is 1, so "a" should be kept
    ]

    for mode, text, expected_min, should_keep in scenarios:
        test_file = tmp_path / f"{mode}.txt"
        test_file.write_text(text + "\n")
        output_file = tmp_path / f"{mode}_out.txt"

        test_args = ["multitool.py", mode, str(test_file), "-o", str(output_file)]
        monkeypatch.setattr(sys, "argv", test_args)

        with patch("sys.stdout.isatty", return_value=False):
            multitool.main()

        content = output_file.read_text().strip()
        if should_keep:
            assert text in content, f"Mode {mode} should have kept {text}"
        else:
            assert text not in content, f"Mode {mode} should have skipped {text} (min_length should be {expected_min})"

def test_count_sentences_min_length_default(tmp_path, monkeypatch):
    # Test 9825 (count --sentences -> 10)
    test_file = tmp_path / "sent.txt"
    test_file.write_text("Short.\n")
    output_file = tmp_path / "sent_out.txt"
    test_args = ["multitool.py", "count", "--sentences", str(test_file), "-o", str(output_file)]
    monkeypatch.setattr(sys, "argv", test_args)
    with patch("sys.stdout.isatty", return_value=False):
        multitool.main()
    assert "Short." not in output_file.read_text()

def test_show_mode_help_comprehensive_metavar():
    # Line 8045-8052
    mock_parser = MagicMock(spec=argparse.ArgumentParser)
    mock_parser.prog = "multitool.py"
    mock_subparser = MagicMock(spec=argparse.ArgumentParser)

    # Action 1: Tuple metavar (8045)
    action_tuple = MagicMock()
    action_tuple.option_strings = ["--tuple"]
    action_tuple.metavar = ("VAL1", "VAL2")
    action_tuple.help = "Tuple help"
    action_tuple.nargs = 2

    # Action 2: Single metavar (8047)
    action_single = MagicMock()
    action_single.option_strings = ["--single"]
    action_single.metavar = "VAL"
    action_single.help = "Single help"
    action_single.nargs = 1

    # Action 3: Choices (8049)
    action_choices = MagicMock()
    action_choices.option_strings = ["--choices"]
    action_choices.metavar = None
    action_choices.choices = ["A", "B"]
    action_choices.help = "Choices help"
    action_choices.nargs = 1

    # Action 4: Default metavar (8052)
    action_default = MagicMock()
    action_default.option_strings = ["--default"]
    action_default.metavar = None
    action_default.choices = None
    action_default.dest = "test_dest"
    action_default.help = "Default help"
    action_default.nargs = 1

    mock_group = MagicMock()
    mock_group.title = "TEST OPTIONS"
    mock_group._group_actions = [action_tuple, action_single, action_choices, action_default]

    mock_subparser._action_groups = [mock_group]

    mock_action = MagicMock(spec=argparse._SubParsersAction)
    mock_action.choices = {"testmode": mock_subparser}
    mock_parser._actions = [mock_action]

    with patch("multitool.MODE_DETAILS", {"testmode": {"summary": "test", "description": "desc", "example": "ex"}}), \
         patch("multitool._should_enable_color", return_value=False):
        try:
            multitool.show_mode_help("testmode", mock_parser)
        except SystemExit:
            pass
