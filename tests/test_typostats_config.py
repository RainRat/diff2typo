import os
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml

import typostats


class TestTypostatsConfig(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)

        # Create a sample input file with typos
        self.input_file = os.path.join(self.temp_dir.name, "sample_typos.txt")
        with open(self.input_file, "w", encoding="utf-8") as f:
            f.write("teh -> the\n")
            f.write("teh -> the\n")
            f.write("wrod -> word\n")

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.temp_dir.cleanup()

    def test_typostats_config_custom_file(self):
        config_file = os.path.join(self.temp_dir.name, "custom_typostats.yaml")
        config_data = {
            "input_files": [self.input_file],
            "output": "custom_output.txt",
            "format": "json",
            "min_count": 2,
            "sort": "typo",
            "limit": 5,
            "quiet": True,
        }
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        test_args = ["typostats.py", "-C", config_file]
        with patch.object(sys, "argv", test_args):
            typostats.main()

        out_path = os.path.join(self.temp_dir.name, "custom_output.txt")
        self.assertTrue(os.path.exists(out_path))

    def test_typostats_config_auto_load(self):
        config_file = os.path.join(self.temp_dir.name, "typostats.yaml")
        config_data = {
            "input_files": [self.input_file],
            "output": "auto_output.csv",
            "format": "csv",
            "quiet": True,
        }
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        test_args = ["typostats.py"]
        with patch.object(sys, "argv", test_args):
            typostats.main()

        out_path = os.path.join(self.temp_dir.name, "auto_output.csv")
        self.assertTrue(os.path.exists(out_path))

    def test_typostats_config_cli_overrides(self):
        config_file = os.path.join(self.temp_dir.name, "typostats.yaml")
        config_data = {
            "input_files": [self.input_file],
            "output": "config_output.csv",
            "format": "csv",
            "quiet": True,
        }
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        override_out = os.path.join(self.temp_dir.name, "override_output.json")
        test_args = ["typostats.py", "-o", override_out, "-f", "json"]
        with patch.object(sys, "argv", test_args):
            typostats.main()

        self.assertFalse(os.path.exists(os.path.join(self.temp_dir.name, "config_output.csv")))
        self.assertTrue(os.path.exists(override_out))

    def test_typostats_config_file_not_found(self):
        test_args = ["typostats.py", "-C", "non_existent_config.yaml"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit) as cm:
                typostats.main()
            self.assertEqual(cm.exception.code, 1)

    def test_typostats_config_missing_pyyaml(self):
        config_file = os.path.join(self.temp_dir.name, "typostats.yaml")
        with open(config_file, "w", encoding="utf-8") as f:
            f.write("quiet: true\n")

        test_args = ["typostats.py"]
        with patch("typostats._YAML_AVAILABLE", False):
            with patch.object(sys, "argv", test_args):
                with self.assertRaises(SystemExit) as cm:
                    typostats.main()
                self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
