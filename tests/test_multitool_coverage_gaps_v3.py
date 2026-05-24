import os
import unittest
from unittest.mock import patch, mock_open, MagicMock
from collections import Counter
import multitool
from multitool import (
    _detect_format_from_extension,
    classify_typo,
    _get_total_line_count,
    _write_paired_output,
    _extract_xml_items,
    _extract_toml_items,
    discovery_mode,
    similarity_mode,
    near_duplicates_mode,
    fuzzymatch_mode,
    count_mode,
    _format_search_line,
    sort_mode,
    standardize_mode,
    main
)

class TestMultitoolCoverageGapsV3(unittest.TestCase):

    def test_detect_format_extension_none_and_unknown(self):
        # Line 152
        self.assertEqual(_detect_format_from_extension("file", ["json"], "default"), "default")
        # Line 172
        self.assertEqual(_detect_format_from_extension("file.unknown", ["json"], "default"), "default")

    def test_classify_typo_1to2_and_fallback(self):
        # Line 284
        self.assertEqual(classify_typo("abcde", "axde", {}), "[1:2]")

        # Line 294
        self.assertEqual(classify_typo("abc", "xyz", {}), "[M]")

    def test_get_total_line_count_oserror(self):
        # Line 351-352
        with patch("builtins.open", side_effect=OSError):
            self.assertEqual(_get_total_line_count(["fake.txt"]), 0)

    def test_write_paired_output_semantic_colors(self):
        # Lines 1052-1057 (scrub_mode logic)
        from io import StringIO
        out = StringIO()

        with patch("multitool.RED", "COLOR_RED"), \
             patch("multitool.GREEN", "COLOR_GREEN"), \
             patch("multitool.MAGENTA", "COLOR_MAGENTA"), \
             patch("multitool.YELLOW", "COLOR_YELLOW"), \
             patch("multitool.CYAN", "COLOR_CYAN"), \
             patch("multitool.RESET", "COLOR_RESET"), \
             patch("multitool.BOLD", "COLOR_BOLD"), \
             patch("multitool.BLUE", "COLOR_BLUE"):

            formatted_pairs = [
                ("the", "teh", "[T]"),     # Magenta
                ("the", "th", "[Del]"),    # Red
                ("the", "th2", "[2:1]"),   # Red
                ("the", "thee", "[Ins]"),  # Green
                ("the", "the2", "[1:2]"),  # Green
                ("the", "tha", "[R]"),     # Yellow
                ("the", "xxx", "[M]"),     # Yellow
                ("the", "the", "[K]")      # Cyan (default)
            ]

            with patch("multitool._should_enable_color", return_value=True), \
                 patch("sys.stdout", out):
                _write_paired_output(formatted_pairs, '-', output_format='arrow', mode_label='Count')
            output = out.getvalue()
            self.assertIn("COLOR_MAGENTA", output)
            self.assertIn("COLOR_RED", output)
            self.assertIn("COLOR_GREEN", output)
            self.assertIn("COLOR_YELLOW", output)
            self.assertIn("COLOR_CYAN", output)

    def test_extract_xml_items_exception(self):
        # Line 1306-1307
        with patch("multitool._read_file_lines_robust", return_value=["<root>abc</root>"]), \
             patch("xml.etree.ElementTree.fromstring", side_effect=Exception("Boom")):
            items = list(_extract_xml_items("fake.xml", key_path=None))
            self.assertEqual(items, [])

    def test_extract_toml_items_empty_and_fallback(self):
        # Line 1323 (empty content)
        with patch("multitool._read_file_lines_robust", return_value=[]):
            items = list(_extract_toml_items("fake.toml", key_path=None))
            self.assertEqual(items, [])

        # Line 1328-1329 (fallback to toml package)
        with patch("multitool._TOMLLIB_AVAILABLE", False), \
             patch("multitool._TOML_AVAILABLE", True), \
             patch("multitool._read_file_lines_robust", return_value=["a = 1\n"]), \
             patch("toml.loads", return_value={"a": 1}):
            items = list(_extract_toml_items("fake.toml", key_path="a"))
            self.assertEqual([str(x) for x in items], ["1"])

        # Line 1331-1332 (Exception)
        with patch("multitool._read_file_lines_robust", return_value=["invalid"]), \
             patch("tomllib.loads" if multitool._TOMLLIB_AVAILABLE else "toml.loads", side_effect=Exception):
            items = list(_extract_toml_items("fake.toml", key_path=None))
            self.assertEqual(items, [])

        # Line 1314-1315 (TOML not available)
        with patch("multitool._TOMLLIB_AVAILABLE", False), \
             patch("multitool._TOML_AVAILABLE", False):
            with self.assertRaises(SystemExit):
                list(_extract_toml_items("fake.toml", key_path=None))

    def test_max_dist_adjustments(self):
        # near_duplicates_mode Line 2837
        with patch("multitool._load_and_clean_file", return_value=([], [], [])):
            near_duplicates_mode(["f"], "-", 1, 100, False, keyboard=True, max_dist=0)
            near_duplicates_mode(["f"], "-", 1, 100, False, transposition=True, max_dist=0)

        # similarity_mode Line 2719
        with patch("multitool._extract_pairs", return_value=[]):
            similarity_mode(["f1"], "-", 1, 100, False, keyboard=True, max_dist=0)

        # fuzzymatch_mode Line 2944
        with patch("multitool._load_and_clean_file", return_value=([], [], [])):
            fuzzymatch_mode(["f1"], "f2", "-", 1, 100, False, keyboard=True, max_dist=0)

        # discovery_mode Line 3279
        with patch("multitool._extract_words_items", return_value=iter([])):
            discovery_mode(["f"], "-", 1, 100, False, keyboard=True, max_dist=0)

    def test_format_search_line_no_parts(self):
        # Line 3450
        res = _format_search_line("filename", 0, "line", True, False, False, True)
        self.assertEqual(res, "line")

    def test_sort_mode_numeric_warning(self):
        # Line 3831
        with patch("multitool._load_and_clean_file", return_value=([], [], [])), \
             patch("logging.warning") as mock_warn:
            sort_mode(["f"], "-", 1, 100, False, by="numeric", clean_items=True)
            mock_warn.assert_called_with("Numeric sorting works best with the --raw (-R) flag. Default cleaning might remove digits.")

    def test_standardize_mode_fuzzy_adjustments_and_filter(self):
        # Line 4466, 4468, 4493, 4495, 4497
        # Pass 1: keyboard filter match
        with patch("multitool._read_file_lines_robust", return_value=["apple", "apple", "aple"]), \
             patch("multitool.classify_typo", return_value="[K]"), \
             patch("multitool._write_paired_output"):
            standardize_mode(["f"], "-", 1, 100, False, fuzzy=0, keyboard=True, threshold=0.1)

        # Pass 2: transposition filter match
        with patch("multitool._read_file_lines_robust", return_value=["apple", "apple", "aple"]), \
             patch("multitool.classify_typo", return_value="[T]"), \
             patch("multitool._write_paired_output"):
            standardize_mode(["f"], "-", 1, 100, False, fuzzy=0, transposition=True, threshold=0.1)

        # Pass 3: no match (continue)
        with patch("multitool._read_file_lines_robust", return_value=["apple", "apple", "aple"]), \
             patch("multitool.classify_typo", return_value="[R]"), \
             patch("multitool._write_paired_output"):
            standardize_mode(["f"], "-", 1, 100, False, fuzzy=0, keyboard=True, threshold=0.1)

    def test_help_choices_metavar(self):
        # Line 5852
        from io import StringIO
        with patch("sys.stdout", new=StringIO()) as mock_out, \
             patch("sys.stderr", new=StringIO()) as mock_err:
            try:
                with patch("sys.argv", ["multitool.py", "help", "sort"]):
                    main()
            except SystemExit:
                pass
            output = mock_out.getvalue() + mock_err.getvalue()
            self.assertIn("alpha,length,numeric", output)

    def test_count_mode_markdown_by_file(self):
        # Line 2077-2078 and visual report color branches 2221-2226
        from io import StringIO

        # 1. Arrow format for visual bar colors 2221-2226
        out = StringIO()
        with patch("multitool.RED", "COLOR_RED"), \
             patch("multitool.GREEN", "COLOR_GREEN"), \
             patch("multitool.MAGENTA", "COLOR_MAGENTA"), \
             patch("multitool.YELLOW", "COLOR_YELLOW"), \
             patch("multitool.CYAN", "COLOR_CYAN"), \
             patch("multitool.RESET", "COLOR_RESET"), \
             patch("multitool.BOLD", "COLOR_BOLD"), \
             patch("multitool.BLUE", "COLOR_BLUE"):

            with patch("multitool._extract_pairs", return_value=[("apple", "aple"), ("the", "teh")]), \
                 patch("multitool.classify_typo", side_effect=["[Del]", "[T]"]):
                with patch("sys.stdout", out), patch("multitool._should_enable_color", return_value=True):
                    count_mode(["f1", "f2"], "-", 1, 100, False, output_format="arrow", pairs=True, by_file=True)

                output = out.getvalue()
                self.assertIn("COLOR_MAGENTA", output)
                self.assertIn("COLOR_RED", output)

        # 2. Markdown format for 2077-2078
        out2 = StringIO()
        with patch("multitool._extract_pairs", return_value=[("apple", "aple")]), \
             patch("multitool.classify_typo", return_value="[T]"):
            with patch("sys.stdout", out2):
                count_mode(["f1", "f2"], "-", 1, 100, False, output_format="markdown", pairs=True, by_file=True)
        self.assertIn("apple -> aple", out2.getvalue())

    def test_write_paired_output_yaml_has_attr(self):
        # Line 960
        from io import StringIO
        out = StringIO()
        pairs = [("left", "right", "[Attr]")]
        with patch("multitool.smart_open_output", return_value=MagicMock(__enter__=lambda s: out, __exit__=lambda s, *a: None)):
             _write_paired_output(pairs, "fake.yaml", output_format="yaml", mode_label="Map")
        self.assertIn("left: right [Attr]", out.getvalue())

if __name__ == "__main__":
    unittest.main()
