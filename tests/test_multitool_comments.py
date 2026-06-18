import pytest
import os
from multitool import comments_mode

def test_extract_comments_basic(tmp_path):
    d = tmp_path / "subdir"
    d.mkdir()
    f = d / "test.py"
    f.write_text("""
# This is a python comment
def hello():
    \"\"\"This is a docstring\"\"\"
    print("hello") # inline comment
    /* This is not a python comment but we might want to extract it if we are polyglot */
    # Another one
""", encoding='utf-8')

    out = tmp_path / "out.txt"
    # Testing python-style by default or all if not specified
    # We'll need to decide on the default behavior. Let's assume it extracts common ones.

    comments_mode(
        input_files=[str(f)],
        output_file=str(out),
        min_length=1,
        max_length=1000,
        process_output=False,
        clean_items=False
    )

    content = out.read_text(encoding='utf-8')
    lines = content.splitlines()
    assert "This is a python comment" in lines
    assert "This is a docstring" in lines
    assert "inline comment" in lines
    assert "This is not a python comment but we might want to extract it if we are polyglot" in lines
    assert "Another one" in lines

def test_extract_comments_c_style(tmp_path):
    f = tmp_path / "test.c"
    f.write_text("""
// C++ style comment
int main() {
    /* Multi-line
       C comment */
    return 0;
}
""", encoding='utf-8')
    out = tmp_path / "out.txt"
    comments_mode([str(f)], str(out), 1, 1000, False, clean_items=False)
    content = out.read_text(encoding='utf-8')
    assert "C++ style comment" in content
    assert "Multi-line" in content
    assert "C comment" in content

def test_extract_comments_xml_style(tmp_path):
    f = tmp_path / "test.xml"
    f.write_text("""
<!-- XML comment -->
<root>
    <!-- Another XML
         comment -->
</root>
""", encoding='utf-8')
    out = tmp_path / "out.txt"
    comments_mode([str(f)], str(out), 1, 1000, False, clean_items=False)
    content = out.read_text(encoding='utf-8')
    assert "XML comment" in content
    assert "Another XML" in content
    assert "comment" in content

def test_extract_comments_sql_style(tmp_path):
    f = tmp_path / "test.sql"
    f.write_text("""
-- SQL comment
SELECT * FROM table; -- another SQL comment
""", encoding='utf-8')
    out = tmp_path / "out.txt"
    comments_mode([str(f)], str(out), 1, 1000, False, clean_items=False)
    content = out.read_text(encoding='utf-8')
    assert "SQL comment" in content
    assert "another SQL comment" in content

def test_extract_comments_ignore_urls(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("""
Check out https://github.com/RainRat/diff2typo
Also http://example.com/some/path
This is a comment // but this is one
""", encoding='utf-8')
    out = tmp_path / "out.txt"
    comments_mode([str(f)], str(out), 1, 1000, False, clean_items=False)
    content = out.read_text(encoding='utf-8')
    lines = content.splitlines()
    assert "github.com/RainRat/diff2typo" not in lines
    assert "example.com/some/path" not in lines
    assert "but this is one" in lines

