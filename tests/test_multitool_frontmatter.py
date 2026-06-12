import os
import unittest
from multitool import _extract_markdown_frontmatter, frontmatter_mode
from io import StringIO
from unittest.mock import patch

class TestMultitoolFrontmatter(unittest.TestCase):
    def test_extract_yaml_frontmatter(self):
        content = "---\ntitle: Hello\nauthor: World\n---\nBody content"
        with patch('multitool._read_file_lines_robust', return_value=content.splitlines(keepends=True)):
            result = list(_extract_markdown_frontmatter("test.md"))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], "title: Hello\nauthor: World\n")

    def test_extract_toml_frontmatter(self):
        content = "+++\ntitle = \"Hello\"\n+++\nBody content"
        with patch('multitool._read_file_lines_robust', return_value=content.splitlines(keepends=True)):
            result = list(_extract_markdown_frontmatter("test.md"))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], "title = \"Hello\"\n")

    def test_extract_body_with_frontmatter(self):
        content = "---\ntitle: Hello\n---\nBody content\nMore body"
        with patch('multitool._read_file_lines_robust', return_value=content.splitlines(keepends=True)):
            result = list(_extract_markdown_frontmatter("test.md", body=True))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], "Body content\nMore body")

    def test_extract_body_without_frontmatter(self):
        content = "Body content\nMore body"
        with patch('multitool._read_file_lines_robust', return_value=content.splitlines(keepends=True)):
            result = list(_extract_markdown_frontmatter("test.md", body=True))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], "Body content\nMore body")

    def test_no_frontmatter_returns_nothing_for_metadata(self):
        content = "Body content\nMore body"
        with patch('multitool._read_file_lines_robust', return_value=content.splitlines(keepends=True)):
            result = list(_extract_markdown_frontmatter("test.md", body=False))
            self.assertEqual(len(result), 0)

    def test_frontmatter_mode_integration(self):
        content = "---\ntitle: Hello\n---\nBody content"
        with patch('multitool._read_file_lines_robust', return_value=content.splitlines(keepends=True)):
            with patch('multitool.write_output') as mock_write:
                frontmatter_mode(
                    input_files=["test.md"],
                    output_file="-",
                    min_length=1,
                    max_length=1000,
                    process_output=False,
                    clean_items=False
                )
                mock_write.assert_called_once()
                items = mock_write.call_args[0][0]
                self.assertEqual(list(items), ["title: Hello\n"])

if __name__ == '__main__':
    unittest.main()
