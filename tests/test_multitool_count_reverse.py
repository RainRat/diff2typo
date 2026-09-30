import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from multitool import count_mode, main


class TestMultitoolCountReverse(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_count_mode_reverse_words(self):
        # Create input file with word frequencies:
        # apple (3), banana (2), cherry (1)
        input_path = os.path.join(self.temp_dir.name, "words.txt")
        output_default = os.path.join(self.temp_dir.name, "out_default.json")
        output_reverse = os.path.join(self.temp_dir.name, "out_reverse.json")

        with open(input_path, "w", encoding="utf-8") as f:
            f.write("apple\napple\napple\nbanana\nbanana\ncherry\n")

        # Run count_mode with default sorting (reverse=False)
        count_mode(
            input_files=[input_path],
            output_file=output_default,
            min_length=1,
            max_length=100,
            process_output=False,
            output_format="json",
            reverse=False,
            quiet=True,
        )

        with open(output_default, "r", encoding="utf-8") as f:
            data_default = json.load(f)

        self.assertEqual(len(data_default), 3)
        self.assertEqual(data_default[0]["item"], "apple")
        self.assertEqual(data_default[0]["count"], 3)
        self.assertEqual(data_default[1]["item"], "banana")
        self.assertEqual(data_default[1]["count"], 2)
        self.assertEqual(data_default[2]["item"], "cherry")
        self.assertEqual(data_default[2]["count"], 1)

        # Run count_mode with reverse sorting (reverse=True)
        count_mode(
            input_files=[input_path],
            output_file=output_reverse,
            min_length=1,
            max_length=100,
            process_output=False,
            output_format="json",
            reverse=True,
            quiet=True,
        )

        with open(output_reverse, "r", encoding="utf-8") as f:
            data_reverse = json.load(f)

        self.assertEqual(len(data_reverse), 3)
        self.assertEqual(data_reverse[0]["item"], "cherry")
        self.assertEqual(data_reverse[0]["count"], 1)
        self.assertEqual(data_reverse[1]["item"], "banana")
        self.assertEqual(data_reverse[1]["count"], 2)
        self.assertEqual(data_reverse[2]["item"], "apple")
        self.assertEqual(data_reverse[2]["count"], 3)

    def test_count_mode_reverse_pairs(self):
        input_path = os.path.join(self.temp_dir.name, "pairs.txt")
        output_path = os.path.join(self.temp_dir.name, "pairs_rev.json")

        # Create pairs: teh -> the (3 times), wrod -> word (1 time)
        with open(input_path, "w", encoding="utf-8") as f:
            f.write("teh -> the\nteh -> the\nteh -> the\nwrod -> word\n")

        count_mode(
            input_files=[input_path],
            output_file=output_path,
            min_length=1,
            max_length=100,
            process_output=False,
            pairs=True,
            output_format="json",
            reverse=True,
            quiet=True,
        )

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["typo"], "wrod")
        self.assertEqual(data[0]["count"], 1)
        self.assertEqual(data[1]["typo"], "teh")
        self.assertEqual(data[1]["count"], 3)

    def test_count_cli_reverse_flag(self):
        input_path = os.path.join(self.temp_dir.name, "cli_words.txt")
        output_path = os.path.join(self.temp_dir.name, "cli_out.json")

        with open(input_path, "w", encoding="utf-8") as f:
            f.write("one\ntwo\ntwo\nthree\nthree\nthree\n")

        test_args = [
            "multitool.py",
            "count",
            input_path,
            "-o",
            output_path,
            "-f",
            "json",
            "-r",
            "-q",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data), 3)
        self.assertEqual(data[0]["item"], "one")
        self.assertEqual(data[0]["count"], 1)
        self.assertEqual(data[1]["item"], "two")
        self.assertEqual(data[1]["count"], 2)
        self.assertEqual(data[2]["item"], "three")
        self.assertEqual(data[2]["count"], 3)


if __name__ == "__main__":
    unittest.main()
