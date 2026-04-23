import unittest
from unittest.mock import patch
import io
from typostats import generate_report

class TestTypoStatsFinalGaps(unittest.TestCase):
    def test_2to1_replacement_not_deletion(self):
        # Case: len(c) > len(t) and t not in c (e.g., 'th' -> 'f')
        # This should trigger line 582 in typostats.py
        replacement_counts = {('th', 'f'): 10}
        with patch('sys.stderr', new_callable=io.StringIO) as mock_stderr:
            generate_report(replacement_counts, allow_2to1=True, show_color_out=False, show_color_err=False)
            report = mock_stderr.getvalue()
        self.assertIn("2-to-1 replacements [2:1]", report)
        self.assertIn("[2:1]", report)

if __name__ == '__main__':
    unittest.main()
