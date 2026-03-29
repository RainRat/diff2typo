import sys
import os
from unittest.mock import patch

# Add the repository root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multitool import search_mode

def test_search_safety_check(caplog):
    # Search for punctuation only should trigger warning and return early
    with caplog.at_level("WARNING"):
        search_mode(
            input_files=["test_search.txt"],
            query="!!!",
            output_file="-",
            min_length=1,
            max_length=100,
            process_output=False,
            clean_items=True
        )
    assert "contains no alphanumeric characters" in caplog.text

def test_search_fuzzy_highlighting(tmp_path):
    input_file = tmp_path / "fuzzy.txt"
    input_file.write_text("This has accomodation", encoding='utf-8')
    output_file = tmp_path / "out.txt"

    # We need to simulate a terminal or FORCE_COLOR to get highlighting
    with patch.dict(os.environ, {"FORCE_COLOR": "1"}):
        # We need to re-import or somehow ensure YELLOW is not empty.
        # In search_mode, use_color = bool(YELLOW).
        # Since YELLOW is a global, it's determined at import time.

        # Let's try to call it and see if we can get it to use color.
        # If the test environment already has colors disabled, YELLOW will be "".
        import multitool
        if not multitool.YELLOW:
            multitool.YELLOW = "\033[1;33m"
            multitool.RESET = "\033[0m"

        search_mode(
            input_files=[str(input_file)],
            query="accommodation",
            output_file=str(output_file),
            min_length=1,
            max_length=100,
            process_output=False,
            max_dist=1,
            clean_items=True
        )

    content = output_file.read_text(encoding='utf-8')
    assert "\033[1;33m" in content
    assert "accomodation" in content

def test_search_multi_word_highlighting(tmp_path):
    input_file = tmp_path / "multi.txt"
    input_file.write_text("The quick brown fox", encoding='utf-8')
    output_file = tmp_path / "out.txt"

    import multitool
    if not multitool.YELLOW:
        multitool.YELLOW = "\033[1;33m"
        multitool.RESET = "\033[0m"

    with patch.dict(os.environ, {"FORCE_COLOR": "1"}):
        search_mode(
            input_files=[str(input_file)],
            query="quick brown",
            output_file=str(output_file),
            min_length=1,
            max_length=100,
            process_output=False
        )

    content = output_file.read_text(encoding='utf-8')
    assert "\033[1;33mquick brown\033[0m" in content
