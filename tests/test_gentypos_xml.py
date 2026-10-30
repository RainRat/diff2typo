import tempfile
import os
import gentypos


def test_format_typos_xml():
    typo_map = {"teh": "the", "wrod": "word"}
    lines = gentypos.format_typos(typo_map, "xml")
    content = "\n".join(lines)
    assert '<?xml version="1.0" encoding="UTF-8"?>' in content
    assert "<typos>" in content
    assert '<typo correct="the">teh</typo>' in content
    assert '<typo correct="word">wrod</typo>' in content
    assert "</typos>" in content


def test_detect_format_from_extension_xml():
    fmt = gentypos._detect_format_from_extension("output.xml", ["xml"], "arrow")
    assert fmt == "xml"


def test_load_substitutions_file_xml_item_format():
    xml_content = """<?xml version="1.0"?>
<root>
    <item>
        <typo>teh</typo>
        <correct>the</correct>
    </item>
    <item>
        <typo>wrod</typo>
        <correct>word</correct>
    </item>
</root>
"""
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_path = f.name

    try:
        subs = gentypos._load_substitutions_file(temp_path)
        assert "the" in subs
        assert "teh" in subs["the"]
        assert "word" in subs
        assert "wrod" in subs["word"]
    finally:
        os.remove(temp_path)


def test_load_substitutions_file_xml_attribute_format():
    xml_content = """<?xml version="1.0"?>
<typos>
    <typo correct="the">teh</typo>
    <typo correct="word">wrod</typo>
</typos>
"""
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_path = f.name

    try:
        subs = gentypos._load_substitutions_file(temp_path)
        assert "the" in subs
        assert "teh" in subs["the"]
        assert "word" in subs
        assert "wrod" in subs["word"]
    finally:
        os.remove(temp_path)


def test_load_substitutions_file_xml_tag_hierarchy_format():
    xml_content = """<?xml version="1.0"?>
<substitutions>
    <the>
        <item>teh</item>
    </the>
</substitutions>
"""
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_path = f.name

    try:
        subs = gentypos._load_substitutions_file(temp_path)
        assert "the" in subs
        assert "teh" in subs["the"]
    finally:
        os.remove(temp_path)
