import sys
from pathlib import Path
from unittest.mock import patch
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

def test_min_length_defaults_in_count_mode(tmp_path, monkeypatch):
    """
    Verify that --min-length correctly defaults to 1 or 3 in count mode
    depending on the other flags.
    """
    dummy_input = tmp_path / "input.txt"
    dummy_input.write_text("a b c d e")
    dummy_mapping = tmp_path / "mapping.csv"
    dummy_mapping.write_text("a,b")

    # 1. Default (word extraction) should be 3
    with patch("sys.argv", ["multitool.py", "count", str(dummy_input)]), \
         patch("multitool.count_mode") as mock_count:
        try:
            multitool.main()
        except SystemExit:
            pass
        _, kwargs = mock_count.call_args
        assert kwargs["min_length"] == 3

    # 2. With --pairs should be 1
    with patch("sys.argv", ["multitool.py", "count", str(dummy_input), "--pairs"]), \
         patch("multitool.count_mode") as mock_count:
        try:
            multitool.main()
        except SystemExit:
            pass
        _, kwargs = mock_count.call_args
        assert kwargs["min_length"] == 1

    # 3. With --chars should be 1
    with patch("sys.argv", ["multitool.py", "count", str(dummy_input), "--chars"]), \
         patch("multitool.count_mode") as mock_count:
        try:
            multitool.main()
        except SystemExit:
            pass
        _, kwargs = mock_count.call_args
        assert kwargs["min_length"] == 1

    # 3b. With --lines should be 1
    with patch("sys.argv", ["multitool.py", "count", str(dummy_input), "--lines"]), \
         patch("multitool.count_mode") as mock_count:
        try:
            multitool.main()
        except SystemExit:
            pass
        _, kwargs = mock_count.call_args
        assert kwargs["min_length"] == 1

    # 4. With --add (ad_hoc) should be 1
    with patch("sys.argv", ["multitool.py", "count", str(dummy_input), "--add", "a:b"]), \
         patch("multitool.count_mode") as mock_count:
        try:
            multitool.main()
        except SystemExit:
            pass
        _, kwargs = mock_count.call_args
        assert kwargs["min_length"] == 1

    # 5. With --mapping should be 1 (BUG: Currently it's 3 because of wrong attribute name)
    with patch("sys.argv", ["multitool.py", "count", str(dummy_input), "--mapping", str(dummy_mapping)]), \
         patch("multitool.count_mode") as mock_count:
        try:
            multitool.main()
        except SystemExit:
            pass
        _, kwargs = mock_count.call_args
        # This is expected to FAIL before the fix
        assert kwargs["min_length"] == 1

    # 6. Words mode should be 3
    with patch("sys.argv", ["multitool.py", "words", str(dummy_input)]), \
         patch("multitool.words_mode") as mock_mode:
        try:
            multitool.main()
        except SystemExit:
            pass
        _, kwargs = mock_mode.call_args
        assert kwargs["min_length"] == 3

    # 7. Search mode should be 1
    with patch("sys.argv", ["multitool.py", "search", str(dummy_input), "--query", "abc"]), \
         patch("multitool.search_mode") as mock_mode:
        try:
            multitool.main()
        except SystemExit:
            pass
        _, kwargs = mock_mode.call_args
        assert kwargs["min_length"] == 1

    # 8. ngrams mode should be 3
    with patch("sys.argv", ["multitool.py", "ngrams", str(dummy_input)]), \
         patch("multitool.ngrams_mode") as mock_mode:
        try:
            multitool.main()
        except SystemExit:
            pass
        _, kwargs = mock_mode.call_args
        assert kwargs["min_length"] == 3

    # 9. stats mode should be 3
    with patch("sys.argv", ["multitool.py", "stats", str(dummy_input)]), \
         patch("multitool.stats_mode") as mock_mode:
        try:
            multitool.main()
        except SystemExit:
            pass
        _, kwargs = mock_mode.call_args
        assert kwargs["min_length"] == 3

    # 10. Explicit min_length should be preserved
    with patch("sys.argv", ["multitool.py", "count", str(dummy_input), "--min-length", "5"]), \
         patch("multitool.count_mode") as mock_count:
        try:
            multitool.main()
        except SystemExit:
            pass
        _, kwargs = mock_count.call_args
        assert kwargs["min_length"] == 5

    # 11. Explicit min_length for --chars should be preserved (NOT overridden by 1)
    with patch("sys.argv", ["multitool.py", "count", str(dummy_input), "--chars", "--min-length", "3"]), \
         patch("multitool.count_mode") as mock_count:
        try:
            multitool.main()
        except SystemExit:
            pass
        _, kwargs = mock_count.call_args
        assert kwargs["min_length"] == 3
