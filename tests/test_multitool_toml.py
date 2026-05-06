import os
import unittest
from unittest.mock import patch
import io
import multitool
import logging

class TestMultitoolToml(unittest.TestCase):
    def setUp(self):
        self.test_toml = "test.toml"
        with open(self.test_toml, "w", encoding="utf-8") as f:
            f.write("""
title = "TOML Example"

[owner]
name = "Tom Preston-Werner"
dob = 1979-05-27T07:32:00Z

[database]
enabled = true
ports = [ 8000, 8001, 8002 ]
data = [ ["delta", "phi"], [3.14] ]
temp_targets = { cpu = 79.5, case = 72.0 }

[[products]]
name = "Hammer"
sku = 738594937

[[products]]
name = "Nail"
sku = 284758393
""")

    def tearDown(self):
        if os.path.exists(self.test_toml):
            os.remove(self.test_toml)

    def test_toml_top_level_keys(self):
        with patch('sys.argv', ['multitool.py', '--raw', 'toml', self.test_toml]), \
             patch('sys.stdout', new=io.StringIO()) as fake_out:
            multitool.main()
            output = fake_out.getvalue().strip().split('\n')
            self.assertIn('title', output)
            self.assertIn('owner', output)
            self.assertIn('database', output)
            self.assertIn('products', output)

    def test_toml_nested_key(self):
        with patch('sys.argv', ['multitool.py', '--raw', 'toml', self.test_toml, '--key', 'owner.name']), \
             patch('sys.stdout', new=io.StringIO()) as fake_out:
            multitool.main()
            output = fake_out.getvalue().strip()
            self.assertEqual(output, "Tom Preston-Werner")

    def test_toml_list_of_tables(self):
        with patch('sys.argv', ['multitool.py', '--raw', 'toml', self.test_toml, '--key', 'products.name']), \
             patch('sys.stdout', new=io.StringIO()) as fake_out:
            multitool.main()
            output = fake_out.getvalue().strip().split('\n')
            self.assertEqual(output, ["Hammer", "Nail"])

    def test_toml_array(self):
        # Use --raw to keep numbers
        with patch('sys.argv', ['multitool.py', '--raw', 'toml', self.test_toml, '--key', 'database.ports']), \
             patch('sys.stdout', new=io.StringIO()) as fake_out:
            multitool.main()
            output = fake_out.getvalue().strip().split('\n')
            self.assertEqual(output, ["8000", "8001", "8002"])

    def test_toml_invalid(self):
        invalid_toml = "invalid.toml"
        with open(invalid_toml, "w") as f:
            f.write("this is not toml")
        try:
            # We use caplog or just check if it exits with 0 (it doesn't sys.exit(1) on parse error in _extract_toml_items, it just returns)
            # Actually _extract_toml_items logs error and returns.
            # But the handler might not exit.
            with patch('sys.argv', ['multitool.py', 'toml', invalid_toml]), \
                 patch('sys.stdout', new=io.StringIO()), \
                 patch('logging.Logger.error') as mock_log_error:
                multitool.main()
                # Check if logging.error was called with something containing "Failed to parse TOML"
                args, _ = mock_log_error.call_args
                self.assertIn("Failed to parse TOML", args[0])
        finally:
            if os.path.exists(invalid_toml):
                os.remove(invalid_toml)

    def test_toml_stdin(self):
        toml_content = 'key = "value"'
        with patch('sys.argv', ['multitool.py', '--raw', 'toml', '-', '--key', 'key']), \
             patch('multitool._read_file_lines_robust', return_value=[toml_content + '\n']), \
             patch('sys.stdout', new=io.StringIO()) as fake_out:
            multitool.main()
            self.assertEqual(fake_out.getvalue().strip(), "value")

    def test_extract_pairs_toml(self):
        with open(self.test_toml, "w") as f:
            f.write('typo = "correction"\n')
        pairs = list(multitool._extract_pairs([self.test_toml]))
        self.assertEqual(pairs, [('typo', 'correction')])

    def test_extract_pairs_toml_replacements(self):
        with open(self.test_toml, "w") as f:
            f.write("""
[[replacements]]
typo = "teh"
correct = "the"
""")
        pairs = list(multitool._extract_pairs([self.test_toml]))
        self.assertEqual(pairs, [('teh', 'the')])

if __name__ == "__main__":
    unittest.main()
