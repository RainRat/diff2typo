import os
import tempfile
import unittest
from multitool import (
    write_output,
    _write_paired_output,
    _write_structured_data,
    _detect_format_from_extension,
    main,
)


class TestMultitoolHTML(unittest.TestCase):
    def test_detect_format_from_extension_html(self):
        allowed = ['line', 'json', 'csv', 'markdown', 'md-table', 'arrow', 'table', 'yaml', 'toml', 'xml', 'html', 'htm']
        self.assertEqual(_detect_format_from_extension('report.html', allowed, 'line'), 'html')
        self.assertEqual(_detect_format_from_extension('report.htm', allowed, 'line'), 'html')

    def test_write_output_html(self):
        items = ["alpha <one>", "beta & gamma", 'quote "test"']
        with tempfile.NamedTemporaryFile(suffix='.html', mode='w+', delete=False) as tf:
            filepath = tf.name

        try:
            write_output(items, filepath, output_format='html')
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("<table>", content)
            self.assertIn("alpha &lt;one&gt;", content)
            self.assertIn("beta &amp; gamma", content)
            self.assertIn("quote &quot;test&quot;", content)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_write_paired_output_html(self):
        pairs_2tuple = [("teh", "the"), ("foo", "bar")]
        with tempfile.NamedTemporaryFile(suffix='.html', mode='w+', delete=False) as tf:
            filepath = tf.name

        try:
            _write_paired_output(pairs_2tuple, filepath, output_format='html', mode_label='Pairs')
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("Pairs Mode Output", content)
            self.assertIn("<th>Typo</th><th>Correction</th>", content)
            self.assertIn("<td><code>teh</code></td><td><code>the</code></td>", content)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_write_paired_output_html_with_attr(self):
        pairs_3tuple = [("teh", "the", "[T]"), ("m", "rn", "[2:1]")]
        with tempfile.NamedTemporaryFile(suffix='.htm', mode='w+', delete=False) as tf:
            filepath = tf.name

        try:
            _write_paired_output(pairs_3tuple, filepath, output_format='htm', mode_label='Classify')
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("Classify Mode Output", content)
            self.assertIn("<th>Typo</th><th>Correction</th><th>Attr</th>", content)
            self.assertIn("<td><code>[T]</code></td>", content)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_write_structured_data_html(self):
        data = {"key": "value", "items": ["a", "b"]}
        with tempfile.NamedTemporaryFile(suffix='.html', mode='w+', delete=False) as tf:
            filepath = tf.name

        try:
            _write_structured_data(data, filepath, output_format='html')
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("<pre><code>", content)
            self.assertIn("&quot;key&quot;: &quot;value&quot;", content)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


if __name__ == '__main__':
    unittest.main()
