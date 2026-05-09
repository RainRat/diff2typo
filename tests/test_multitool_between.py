import unittest
from unittest.mock import patch

# Import the functions to be tested from multitool
from multitool import _extract_between_items, between_mode

class TestBetweenMode(unittest.TestCase):
    def test_extract_between_items_single_line(self):
        content = "The value is {{secret}} and {{another_one}} here."
        with patch('multitool._read_file_lines_robust', return_value=[content]):
            items = list(_extract_between_items('test.txt', start='{{', end='}}', quiet=True))
            self.assertEqual(items, ['secret', 'another_one'])

    def test_extract_between_items_multi_line(self):
        content = ["Start here <!--\n", "This is a comment\n", "spanning multiple lines\n", "--> End here"]
        # When multi_line is True, it joins the lines.
        with patch('multitool._read_file_lines_robust', return_value=content):
            items = list(_extract_between_items('test.txt', start='<!--', end='-->', multi_line=True, quiet=True))
            self.assertEqual(items, ['\nThis is a comment\nspanning multiple lines\n'])

    def test_extract_between_items_no_match(self):
        content = "No markers here."
        with patch('multitool._read_file_lines_robust', return_value=[content]):
            items = list(_extract_between_items('test.txt', start='[[', end=']]', quiet=True))
            self.assertEqual(items, [])

    def test_extract_between_items_regex_escaping(self):
        # Markers that contain regex special characters like dots, stars, parens
        content = "Matching (start) content (end) and [more] markers."
        with patch('multitool._read_file_lines_robust', return_value=[content]):
            items = list(_extract_between_items('test.txt', start='(start)', end='(end)', quiet=True))
            self.assertEqual(items, [' content '])

    def test_between_mode_execution(self):
        # Mocking _process_items to ensure between_mode calls it correctly
        with patch('multitool._process_items') as mock_process:
            between_mode(
                input_files=['input.txt'],
                output_file='output.txt',
                min_length=1,
                max_length=1000,
                process_output=False,
                start='[[',
                end=']]',
                multi_line=False,
                output_format='line',
                quiet=True,
                clean_items=True,
                limit=None
            )
            # Verify that the extractor passed to _process_items calls _extract_between_items correctly
            args, kwargs = mock_process.call_args
            extractor = args[0]

            with patch('multitool._extract_between_items') as mock_extract:
                extractor('some_file', quiet=True)
                mock_extract.assert_called_once_with('some_file', '[[', ']]', multi_line=False, quiet=True)

            self.assertEqual(args[1], ['input.txt'])
            self.assertEqual(args[2], 'output.txt')
            self.assertEqual(args[6], 'Between')
            self.assertEqual(args[7], 'Successfully got strings.')

if __name__ == '__main__':
    unittest.main()
