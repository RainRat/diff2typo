"""
Unit tests for --max-length CLI option in diff2typo.py
"""

import sys
import unittest.mock
from diff2typo import find_typos, main


SAMPLE_DIFF = """
diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
-teh short word and supercalifragilisticexpialidocious
+the short word and supercalifragilisticexpialidociou
"""


def test_find_typos_with_max_length():
    # Without max_length
    typos_all = find_typos(SAMPLE_DIFF, min_length=2, max_length=None)
    assert "teh -> the" in typos_all
    assert "supercalifragilisticexpialidocious -> supercalifragilisticexpialidociou" in typos_all

    # With max_length filtering out the long word (34 chars)
    typos_filtered = find_typos(SAMPLE_DIFF, min_length=2, max_length=10)
    assert "teh -> the" in typos_filtered
    assert "supercalifragilisticexpialidocious -> supercalifragilisticexpialidociou" not in typos_filtered


def test_diff2typo_cli_max_length(tmp_path, monkeypatch, capsys):
    diff_file = tmp_path / "test.diff"
    diff_file.write_text(SAMPLE_DIFF, encoding="utf-8")
    out_file = tmp_path / "out.txt"

    test_args = [
        "diff2typo.py",
        str(diff_file),
        "--output",
        str(out_file),
        "--max-length",
        "10",
        "--dictionary",
        "nonexistent.csv",
        "--allowed",
        "nonexistent.csv",
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with unittest.mock.patch("diff2typo.filter_known_typos", side_effect=lambda c, **kwargs: c):
        main()

    content = out_file.read_text(encoding="utf-8")
    assert "teh -> the" in content
    assert "supercalifragilisticexpialidocious" not in content
