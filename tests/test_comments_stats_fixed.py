import logging
import pytest
import re
from multitool import comments_mode

def test_comments_mode_retention_stats(tmp_path, caplog):
    """Verify that comments_mode correctly counts lines within multi-line comments for its stats."""
    f = tmp_path / "test.py"
    f.write_text('"""\nLine 1\nLine 2\nLine 3\n"""\n', encoding='utf-8')
    out = tmp_path / "out.txt"

    with caplog.at_level(logging.INFO):
        comments_mode(
            input_files=[str(f)],
            output_file=str(out),
            min_length=1,
            max_length=1000,
            process_output=False,
            clean_items=True
        )

    # Check the analysis summary in the logs
    # We expect 3 items analyzed (the 3 lines inside the triple quotes)
    assert "Total comments analyzed:            3" in caplog.text
    assert "Total comments after filtering:     3" in caplog.text
    assert "Retention rate:                     100.0%" in caplog.text

def test_comments_mode_retention_stats_with_filtering(tmp_path, caplog):
    """Verify stats when some lines in a multi-line comment are filtered out."""
    f = tmp_path / "test.py"
    # 'a' will be filtered by min_length=2
    f.write_text('"""\na\nlong line\n"""\n', encoding='utf-8')
    out = tmp_path / "out.txt"

    with caplog.at_level(logging.INFO):
        comments_mode(
            input_files=[str(f)],
            output_file=str(out),
            min_length=2,
            max_length=1000,
            process_output=False,
            clean_items=True
        )

    # We expect 2 items analyzed ('a' and 'long line')
    # and 1 item after filtering ('long line' -> 'longline')
    assert "Total comments analyzed:            2" in caplog.text
    assert "Total comments after filtering:     1" in caplog.text
    # Use re.search to handle multiple spaces
    assert re.search(r"Retention rate:\s+50\.0%", caplog.text)
