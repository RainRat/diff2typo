import os
import tempfile
from multitool import _extract_pairs, pairs_mode, swap_mode, count_mode, scrub_mode


def test_extract_pairs_html_table():
    html_content = """
    <!DOCTYPE html>
    <html>
    <body>
      <table>
        <thead>
          <tr><th>Left</th><th>Right</th></tr>
        </thead>
        <tbody>
          <tr><td>teh</td><td>the</td></tr>
          <tr><td>wrod</td><td>word</td></tr>
        </tbody>
      </table>
    </body>
    </html>
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(html_content)
        temp_path = f.name

    try:
        pairs = list(_extract_pairs([temp_path]))
        assert pairs == [("teh", "the"), ("wrod", "word")]
    finally:
        os.remove(temp_path)


def test_extract_pairs_html_entities_and_headers():
    html_content = """
    <table>
      <tr><th>Typo</th><th>Correction</th></tr>
      <tr><td>rock &amp; roll</td><td>rock &lt;roll&gt;</td></tr>
    </table>
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.htm', delete=False, encoding='utf-8') as f:
        f.write(html_content)
        temp_path = f.name

    try:
        pairs = list(_extract_pairs([temp_path]))
        assert pairs == [("rock & roll", "rock <roll>")]
    finally:
        os.remove(temp_path)


def test_extract_pairs_html_nested_tags():
    html_content = """
    <table>
      <tr><td><code>recieve</code></td><td><code>receive</code></td></tr>
    </table>
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(html_content)
        temp_path = f.name

    try:
        pairs = list(_extract_pairs([temp_path]))
        assert pairs == [("recieve", "receive")]
    finally:
        os.remove(temp_path)


def test_multitool_pairs_mode_html_input(capsys):
    html_content = """
    <table>
      <tr><th>Original</th><th>Fix</th></tr>
      <tr><td>teh</td><td>the</td></tr>
    </table>
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(html_content)
        input_path = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        output_path = f.name

    try:
        pairs_mode([input_path], output_path, min_length=1, max_length=100, process_output=False, output_format='csv')
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        assert "teh,the" in content
    finally:
        os.remove(input_path)
        os.remove(output_path)


def test_multitool_swap_mode_html_input():
    html_content = """
    <table>
      <tr><td>teh</td><td>the</td></tr>
    </table>
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(html_content)
        input_path = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        output_path = f.name

    try:
        swap_mode([input_path], output_path, min_length=1, max_length=100, process_output=False, output_format='csv')
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        assert "the,teh" in content
    finally:
        os.remove(input_path)
        os.remove(output_path)


def test_multitool_count_pairs_html_input():
    html_content = """
    <table>
      <tr><td>teh</td><td>the</td></tr>
      <tr><td>teh</td><td>the</td></tr>
    </table>
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(html_content)
        input_path = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        output_path = f.name

    try:
        count_mode([input_path], output_path, min_length=1, max_length=100, process_output=False, pairs=True, output_format='json')
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "teh" in content
        assert "the" in content
    finally:
        os.remove(input_path)
        os.remove(output_path)


def test_multitool_scrub_html_mapping():
    html_mapping = """
    <table>
      <tr><td>teh</td><td>the</td></tr>
    </table>
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(html_mapping)
        mapping_path = f.name

    target_text = "This is teh test."
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(target_text)
        target_path = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        output_path = f.name

    try:
        scrub_mode([target_path], mapping_path, output_path, min_length=1, max_length=100, process_output=False)
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        assert content == "This is the test."
    finally:
        os.remove(mapping_path)
        os.remove(target_path)
        os.remove(output_path)


def test_extract_pairs_html_no_table_fallback():
    html_content = "teh -> the\nwrod -> word\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(html_content)
        temp_path = f.name

    try:
        pairs = list(_extract_pairs([temp_path]))
        assert pairs == [("teh", "the"), ("wrod", "word")]
    finally:
        os.remove(temp_path)
