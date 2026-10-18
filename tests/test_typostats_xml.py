import os
import tempfile
import xml.etree.ElementTree as ET
import typostats


def test_extract_pairs_xml_child_tags():
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<root>
    <pair>
        <left>teh</left>
        <right>the</right>
    </pair>
    <pair>
        <typo>receieve</typo>
        <correct>receive</correct>
    </pair>
</root>
"""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_path = f.name

    try:
        pairs = list(typostats._extract_pairs([temp_path]))
        assert ("teh", "the") in pairs
        assert ("receieve", "receive") in pairs
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_extract_pairs_xml_attributes():
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<root>
    <item typo="teh" correct="the" />
    <item left="adress" right="address" />
</root>
"""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_path = f.name

    try:
        pairs = list(typostats._extract_pairs([temp_path]))
        assert ("teh", "the") in pairs
        assert ("adress", "address") in pairs
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_generate_report_xml_format():
    counts = {("the", "teh"): 3, ("address", "adress"): 1}
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".xml", delete=False) as f:
        out_path = f.name

    try:
        typostats.generate_report(
            counts,
            output_file=out_path,
            output_format="xml",
            allow_transposition=True,
            keyboard=True,
            total_pairs=4,
        )

        tree = ET.parse(out_path)
        root = tree.getroot()
        assert root.tag == "typostats"

        summary = root.find("summary")
        assert summary is not None
        assert summary.find("word_pairs_analyzed").text == "4"

        replacements = root.find("replacements")
        assert replacements is not None
        items = replacements.findall("item")
        assert len(items) == 2

        typos = [item.find("typo").text for item in items]
        assert "teh" in typos
        assert "adress" in typos
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def test_detect_format_xml():
    fmt = typostats._detect_format_from_extension("report.xml", ["arrow", "xml", "json"], "arrow")
    assert fmt == "xml"
