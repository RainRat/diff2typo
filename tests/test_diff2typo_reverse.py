import os
import tempfile
import sys
from unittest.mock import patch

import diff2typo


SAMPLE_DIFF = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
-    banana = "test"
-    apple = "test"
-    apple = "test"
-    cherry = "test"
+    bananas = "test"
+    apples = "test"
+    apples = "test"
+    cherrys = "test"
"""


def test_diff2typo_reverse_alpha():
    with tempfile.TemporaryDirectory() as tmpdir:
        diff_file = os.path.join(tmpdir, "test.diff")
        out_file = os.path.join(tmpdir, "out.txt")
        words_file = os.path.join(tmpdir, "words.csv")
        allowed_file = os.path.join(tmpdir, "allowed.csv")

        with open(diff_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_DIFF)
        with open(words_file, "w", encoding="utf-8") as f:
            f.write("correct\n")
        with open(allowed_file, "w", encoding="utf-8") as f:
            f.write("allowed\n")

        test_args = [
            "diff2typo.py",
            diff_file,
            "-o", out_file,
            "-s", "alpha",
            "-r",
            "-d", words_file,
            "-a", allowed_file,
            "-f", "list",
            "--quiet",
        ]

        with patch.object(sys, "argv", test_args):
            diff2typo.main()

        with open(out_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Alpha descending: cherry -> banana -> apple
        assert lines == ["cherry", "banana", "apple"]


def test_diff2typo_reverse_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        diff_file = os.path.join(tmpdir, "test.diff")
        out_file = os.path.join(tmpdir, "out.txt")
        words_file = os.path.join(tmpdir, "words.csv")
        allowed_file = os.path.join(tmpdir, "allowed.csv")

        with open(diff_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_DIFF)
        with open(words_file, "w", encoding="utf-8") as f:
            f.write("correct\n")
        with open(allowed_file, "w", encoding="utf-8") as f:
            f.write("allowed\n")

        test_args = [
            "diff2typo.py",
            diff_file,
            "-o", out_file,
            "-s", "count",
            "-r",
            "-d", words_file,
            "-a", allowed_file,
            "-f", "list",
            "--quiet",
        ]

        with patch.object(sys, "argv", test_args):
            diff2typo.main()

        with open(out_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Count ascending (least frequent first): banana (1) / cherry (1), then apple (2)
        # For frequency 1: banana and cherry
        assert lines[-1] == "apple"
        assert set(lines[:2]) == {"banana", "cherry"}


def test_diff2typo_dry_run_reverse(caplog):
    with tempfile.TemporaryDirectory() as tmpdir:
        diff_file = os.path.join(tmpdir, "test.diff")
        words_file = os.path.join(tmpdir, "words.csv")
        allowed_file = os.path.join(tmpdir, "allowed.csv")

        with open(diff_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_DIFF)
        with open(words_file, "w", encoding="utf-8") as f:
            f.write("correct\n")
        with open(allowed_file, "w", encoding="utf-8") as f:
            f.write("allowed\n")

        test_args = [
            "diff2typo.py",
            diff_file,
            "-n",
            "-r",
            "-d", words_file,
            "-a", allowed_file,
        ]

        with patch.object(sys, "argv", test_args):
            with caplog.at_level("INFO"):
                diff2typo.main()

        assert "Reverse: True" in caplog.text
