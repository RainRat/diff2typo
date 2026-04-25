import unittest
from unittest.mock import patch

# Import the functions to be tested from multitool
from multitool import _extract_quoted_items, quoted_mode

class TestQuotedMode(unittest.TestCase):
    def test_extract_quoted_items_double_quotes(self):
        # Create a temporary file with double-quoted strings
        content = 'This is a "double-quoted" string and another "one here".\n'
        with patch('multitool._read_file_lines_robust', return_value=[content]):
            items = list(_extract_quoted_items('test.txt', quiet=True))
            self.assertEqual(items, ['double-quoted', 'one here'])

    def test_extract_quoted_items_single_quotes(self):
        # Create a temporary file with single-quoted strings
        content = "This is a 'single-quoted' string and 'another' one.\n"
        with patch('multitool._read_file_lines_robust', return_value=[content]):
            items = list(_extract_quoted_items('test.txt', quiet=True))
            self.assertEqual(items, ['single-quoted', 'another'])

    def test_extract_quoted_items_mixed_quotes(self):
        content = 'Mixed "double" and \'single\' quotes on "the same" line.\n'
        with patch('multitool._read_file_lines_robust', return_value=[content]):
            items = list(_extract_quoted_items('test.txt', quiet=True))
            self.assertEqual(items, ['double', 'single', 'the same'])

    def test_extract_quoted_items_escaped_quotes(self):
        # Escaped double quotes inside double quotes
        content = 'Text with "escaped \\"quotes\\" inside" it.\n'
        with patch('multitool._read_file_lines_robust', return_value=[content]):
            items = list(_extract_quoted_items('test.txt', quiet=True))
            self.assertEqual(items, ['escaped \\"quotes\\" inside'])

        # Escaped single quotes inside single quotes
        content = "Text with 'escaped \\'quotes\\' inside' it.\n"
        with patch('multitool._read_file_lines_robust', return_value=[content]):
            items = list(_extract_quoted_items('test.txt', quiet=True))
            self.assertEqual(items, ["escaped \\'quotes\\' inside"])

    def test_extract_quoted_items_empty_quotes(self):
        content = 'Empty "" and \'\' quotes.\n'
        with patch('multitool._read_file_lines_robust', return_value=[content]):
            items = list(_extract_quoted_items('test.txt', quiet=True))
            self.assertEqual(items, ['', ''])

    def test_quoted_mode_execution(self):
        # Mocking _process_items to ensure quoted_mode calls it correctly
        with patch('multitool._process_items') as mock_process:
            quoted_mode(
                input_files=['input.txt'],
                output_file='output.txt',
                min_length=3,
                max_length=100,
                process_output=True,
                output_format='line',
                quiet=True,
                clean_items=True,
                limit=10
            )
            mock_process.assert_called_once_with(
                _extract_quoted_items,
                ['input.txt'],
                'output_file' if 'output_file' == 'output.txt' else 'output.txt', # Handle potential name mapping if any
                3,
                100,
                True,
                'Quoted',
                'Successfully got quoted strings.',
                'line',
                True,
                clean_items=True,
                limit=10
            )

if __name__ == '__main__':
    unittest.main()
