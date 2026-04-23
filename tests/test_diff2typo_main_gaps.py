import unittest
from unittest.mock import patch, MagicMock
import diff2typo
import io
import os
import logging

class TestDiff2TypoMainGaps(unittest.TestCase):
    def test_main_both_mode_no_typos(self):
        # Target branch 793->797: if typos_result is empty
        with patch('diff2typo.argparse.ArgumentParser.parse_args') as mock_args, \
             patch('diff2typo._read_diff_sources', return_value=""), \
             patch('diff2typo.read_words_mapping', return_value={}), \
             patch('diff2typo.read_allowed_words', return_value=set()), \
             patch('diff2typo.process_typos_mode', return_value=[]), \
             patch('diff2typo.process_corrections_mode', return_value=[]), \
             patch('diff2typo.format_typos', return_value=['correction1']), \
             patch('diff2typo.smart_open_output') as mock_open:

            mock_args.return_value = MagicMock(
                mode='both',
                output_file='-',
                output_format='text',
                quiet=True,
                git=None,
                diff_files=[],
                dictionary_file='words.csv',
                allowed_file='allowed.txt',
                min_length=2,
                max_dist=2,
                typos_tool_path='typos'
            )

            f = io.StringIO()
            mock_open.return_value.__enter__.return_value = f

            diff2typo.main()

            output = f.getvalue()
            self.assertIn("=== Corrections ===", output)
            self.assertNotIn("=== Typos ===", output)

    def test_main_both_mode_no_corrections(self):
        # Target branch 797->808: if corrections_result is empty
        with patch('diff2typo.argparse.ArgumentParser.parse_args') as mock_args, \
             patch('diff2typo._read_diff_sources', return_value=""), \
             patch('diff2typo.read_words_mapping', return_value={}), \
             patch('diff2typo.read_allowed_words', return_value=set()), \
             patch('diff2typo.process_typos_mode', return_value=['typo1']), \
             patch('diff2typo.process_corrections_mode', return_value=[]), \
             patch('diff2typo.format_typos', return_value=[]), \
             patch('diff2typo.smart_open_output') as mock_open:

            mock_args.return_value = MagicMock(
                mode='both',
                output_file='-',
                output_format='text',
                quiet=True,
                git=None,
                diff_files=[],
                dictionary_file='words.csv',
                allowed_file='allowed.txt',
                min_length=2,
                max_dist=2,
                typos_tool_path='typos'
            )

            f = io.StringIO()
            mock_open.return_value.__enter__.return_value = f

            diff2typo.main()

            output = f.getvalue()
            self.assertIn("=== Typos ===", output)
            self.assertNotIn("=== Corrections ===", output)

    def test_main_typos_mode(self):
        # Target coverage for typos mode branch and 804->808 jump
        with patch('diff2typo.argparse.ArgumentParser.parse_args') as mock_args, \
             patch('diff2typo._read_diff_sources', return_value=""), \
             patch('diff2typo.read_words_mapping', return_value={}), \
             patch('diff2typo.read_allowed_words', return_value=set()), \
             patch('diff2typo.process_typos_mode', return_value=['typo1']), \
             patch('diff2typo.smart_open_output') as mock_open:

            mock_args.return_value = MagicMock(
                mode='typos',
                output_file='-',
                output_format='text',
                quiet=True,
                git=None,
                diff_files=[],
                dictionary_file='words.csv',
                allowed_file='allowed.txt',
                min_length=2,
                max_dist=2,
                typos_tool_path='typos'
            )

            f = io.StringIO()
            mock_open.return_value.__enter__.return_value = f

            diff2typo.main()
            self.assertEqual(f.getvalue(), "typo1\n")

    def test_main_corrections_mode(self):
        # Target coverage for corrections mode branch and 804->808 jump
        with patch('diff2typo.argparse.ArgumentParser.parse_args') as mock_args, \
             patch('diff2typo._read_diff_sources', return_value=""), \
             patch('diff2typo.read_words_mapping', return_value={}), \
             patch('diff2typo.read_allowed_words', return_value=set()), \
             patch('diff2typo.process_corrections_mode', return_value=[]), \
             patch('diff2typo.format_typos', return_value=['correction1']), \
             patch('diff2typo.smart_open_output') as mock_open:

            mock_args.return_value = MagicMock(
                mode='corrections',
                output_file='-',
                output_format='text',
                quiet=True,
                git=None,
                diff_files=[],
                dictionary_file='words.csv',
                allowed_file='allowed.txt',
                min_length=2,
                max_dist=2
            )

            f = io.StringIO()
            mock_open.return_value.__enter__.return_value = f

            diff2typo.main()
            self.assertEqual(f.getvalue(), "correction1\n")

    def test_main_both_mode_none(self):
        # Target both branches being empty
        with patch('diff2typo.argparse.ArgumentParser.parse_args') as mock_args, \
             patch('diff2typo._read_diff_sources', return_value=""), \
             patch('diff2typo.read_words_mapping', return_value={}), \
             patch('diff2typo.read_allowed_words', return_value=set()), \
             patch('diff2typo.process_typos_mode', return_value=[]), \
             patch('diff2typo.process_corrections_mode', return_value=[]), \
             patch('diff2typo.format_typos', return_value=[]), \
             patch('diff2typo.smart_open_output') as mock_open:

            mock_args.return_value = MagicMock(
                mode='both',
                output_file='-',
                output_format='text',
                quiet=True,
                git=None,
                diff_files=[],
                dictionary_file='words.csv',
                allowed_file='allowed.txt',
                min_length=2,
                max_dist=2,
                typos_tool_path='typos'
            )

            f = io.StringIO()
            mock_open.return_value.__enter__.return_value = f

            diff2typo.main()

            output = f.getvalue()
            self.assertEqual(output, "")

    def test_main_audit_mode_success(self):
        with patch('diff2typo.argparse.ArgumentParser.parse_args') as mock_args, \
             patch('diff2typo._read_diff_sources', return_value=""), \
             patch('diff2typo.read_words_mapping', return_value={}), \
             patch('diff2typo.read_allowed_words', return_value=set()), \
             patch('diff2typo.process_audit_typos', return_value=['audit1']), \
             patch('diff2typo.smart_open_output') as mock_open:

            mock_args.return_value = MagicMock(
                mode='audit',
                output_file='-',
                output_format='text',
                quiet=True,
                git=None,
                diff_files=[],
                dictionary_file='words.csv',
                allowed_file='allowed.txt',
                min_length=2,
                max_dist=2
            )

            f = io.StringIO()
            mock_open.return_value.__enter__.return_value = f

            diff2typo.main()

            output = f.getvalue()
            self.assertEqual(output, "audit1\n")

    def test_main_invalid_mode(self):
        # Target 804->808: args.mode is not 'audit' (and not 'both', 'typos', 'corrections')
        with patch('diff2typo.argparse.ArgumentParser.parse_args') as mock_args, \
             patch('diff2typo._read_diff_sources', return_value=""), \
             patch('diff2typo.read_words_mapping', return_value={}), \
             patch('diff2typo.read_allowed_words', return_value=set()), \
             patch('diff2typo.smart_open_output') as mock_open:

            mock_args.return_value = MagicMock(
                mode='invalid',
                output_file='-',
                output_format='text',
                quiet=True,
                git=None,
                diff_files=[],
                dictionary_file='words.csv',
                allowed_file='allowed.txt',
                min_length=2,
                max_dist=2
            )

            f = io.StringIO()
            mock_open.return_value.__enter__.return_value = f

            diff2typo.main()
            self.assertEqual(f.getvalue(), "")

    def test_process_corrections_mode_known_correction(self):
        # Target 551->550 and 555 missing (after IS in words_mapping[before])
        candidates = ["teh -> the", "no-arrow"]
        words_mapping = {"teh": {"the"}}
        # should return empty because 'the' is already a known correction for 'teh'
        # and 'no-arrow' is skipped by 551->550
        result = diff2typo.process_corrections_mode(candidates, words_mapping, quiet=True)
        self.assertEqual(result, [])

    def test_process_audit_typos_not_arrow(self):
        # Target 571->570: candidate does not have ' -> '
        candidates = ["teh-the"]
        large_dictionary = {"teh"}
        allowed_words = set()
        args = MagicMock(output_format='text')
        result = diff2typo.process_audit_typos(candidates, args, large_dictionary, allowed_words)
        self.assertEqual(result, [])

    def test_format_typos_list_format(self):
        # Target 335->326 (list format)
        typos = ["teh -> the"]
        result = diff2typo.format_typos(typos, 'list')
        self.assertEqual(result, ["teh"])

    def test_format_typos_invalid_format(self):
        # Target 335->326: output_format is not 'list' (and not 'arrow', 'csv', 'table')
        typos = ["teh -> the"]
        result = diff2typo.format_typos(typos, 'invalid')
        self.assertEqual(result, [])

    def test_compare_word_lists_min_length_filter(self):
        # Target 220->210: words shorter than min_length
        # Target 221->210: distance > max_dist
        before = ["a", "house"]
        after = ["b", "mouse"]
        # 'a' -> 'b' is too short
        # 'house' -> 'mouse' is dist 1, so max_dist=0 should filter it
        result = diff2typo._compare_word_lists(before, after, min_length=2, max_dist=0)
        self.assertEqual(result, [])

    def test_read_words_mapping_empty_row(self):
        # Target 176->175: empty row
        with patch('diff2typo._read_csv_rows', return_value=[[], ["teh", "the"]]):
            result = diff2typo.read_words_mapping("fake.csv")
            self.assertEqual(result, {"teh": {"the"}})

    def test_minimal_formatter_no_color(self):
        # Target 80->83: color is None
        # We need to mock record.levelno to something NOT in LEVEL_COLORS or mock it but return None
        formatter = diff2typo.MinimalFormatter()
        record = logging.LogRecord("name", logging.DEBUG, "pathname", 10, "msg", None, None)
        # DEBUG is not in LEVEL_COLORS
        with patch('diff2typo.sys.stderr.isatty', return_value=True):
            result = formatter.format(record)
            self.assertEqual(result, "DEBUG: msg")

    def test_color_initialization(self):
        # Target 59->63: os.environ.get('NO_COLOR') or not sys.stdout.isatty()
        # This is hard because it runs at module load.
        # But we can reload the module.
        import importlib
        with patch.dict(os.environ, {"NO_COLOR": "1"}), patch('diff2typo.sys.stdout.isatty', return_value=True):
            importlib.reload(diff2typo)
            self.assertEqual(diff2typo.BLUE, "")

        # Reset it back
        with patch.dict(os.environ, {}), patch('diff2typo.sys.stdout.isatty', return_value=True):
            importlib.reload(diff2typo)
            self.assertNotEqual(diff2typo.BLUE, "")
