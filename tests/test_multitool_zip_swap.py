import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from multitool import zip_mode, main


class TestMultitoolZipSwap(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file1 = os.path.join(self.temp_dir.name, "left.txt")
        self.file2 = os.path.join(self.temp_dir.name, "right.txt")
        self.out_file = os.path.join(self.temp_dir.name, "output.json")

        with open(self.file1, "w", encoding="utf-8") as f:
            f.write("apple\nbanana\ncherry\n")

        with open(self.file2, "w", encoding="utf-8") as f:
            f.write("fruit\nyellow\nred\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_zip_mode_without_swap(self):
        zip_mode(
            input_files=[self.file1],
            file2=self.file2,
            output_file=self.out_file,
            min_length=1,
            max_length=1000,
            process_output=False,
            output_format="json",
            quiet=True,
            swap=False,
        )

        with open(self.out_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data, {"apple": "fruit", "banana": "yellow", "cherry": "red"})

    def test_zip_mode_with_swap(self):
        zip_mode(
            input_files=[self.file1],
            file2=self.file2,
            output_file=self.out_file,
            min_length=1,
            max_length=1000,
            process_output=False,
            output_format="json",
            quiet=True,
            swap=True,
        )

        with open(self.out_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data, {"fruit": "apple", "yellow": "banana", "red": "cherry"})

    def test_cli_zip_swap_flag(self):
        test_args = [
            "multitool.py",
            "zip",
            self.file1,
            self.file2,
            "--swap",
            "-o",
            self.out_file,
            "-f",
            "json",
            "-q",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        with open(self.out_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data, {"fruit": "apple", "yellow": "banana", "red": "cherry"})

    def test_cli_zip_reverse_pairs_alias(self):
        test_args = [
            "multitool.py",
            "zip",
            self.file1,
            self.file2,
            "--reverse-pairs",
            "-o",
            self.out_file,
            "-f",
            "json",
            "-q",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        with open(self.out_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data, {"fruit": "apple", "yellow": "banana", "red": "cherry"})


if __name__ == "__main__":
    unittest.main()
