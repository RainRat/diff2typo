import os
import xml.etree.ElementTree as ET
from pathlib import Path
import pytest
from multitool import _extract_xml_items, xml_mode, write_output, _write_paired_output, _detect_format_from_extension

def test_extract_xml_items_basic(tmp_path):
    xml_file = tmp_path / "test.xml"
    xml_file.write_text("<root><item>Hello</item><item>World</item></root>")

    # Extract all text (no key)
    items = list(_extract_xml_items(str(xml_file), ""))
    assert items == ["Hello", "World"]

    # Extract by tag
    items = list(_extract_xml_items(str(xml_file), "item"))
    assert items == ["Hello", "World"]

def test_extract_xml_items_xpath(tmp_path):
    xml_file = tmp_path / "test.xml"
    xml_file.write_text("""
    <root>
        <data>
            <item>Target 1</item>
        </data>
        <item>Ignored</item>
    </root>
    """)

    items = list(_extract_xml_items(str(xml_file), ".//data/item"))
    assert items == ["Target 1"]

def test_extract_xml_items_nested_text(tmp_path):
    xml_file = tmp_path / "test.xml"
    xml_file.write_text("<root><item>Hello <b>Beautiful</b> World</item></root>")

    items = list(_extract_xml_items(str(xml_file), "item"))
    assert items == ["Hello Beautiful World"]

def test_write_output_xml(tmp_path):
    output_file = tmp_path / "out.xml"
    items = ["alpha", "beta"]
    write_output(items, str(output_file), output_format='xml')

    content = output_file.read_text()
    assert "<items>" in content
    assert "<item>alpha</item>" in content
    assert "<item>beta</item>" in content

    # Verify it's valid XML
    root = ET.fromstring(content)
    assert root.tag == "items"
    assert len(root.findall("item")) == 2

def test_write_paired_output_xml(tmp_path):
    output_file = tmp_path / "pairs.xml"
    pairs = [("teh", "the"), ("recieved", "received", "[K]")]
    _write_paired_output(pairs, str(output_file), output_format='xml', mode_label="Test")

    content = output_file.read_text()
    assert "<pairs>" in content
    assert "<pair>" in content
    assert "<left>teh</left>" in content
    assert "<right>the</right>" in content
    assert "<attr>[K]</attr>" in content

    # Verify it's valid XML
    root = ET.fromstring(content)
    assert root.tag == "pairs"
    p_elems = root.findall("pair")
    assert len(p_elems) == 2
    assert p_elems[1].find("attr").text == "[K]"

def test_detect_format_xml():
    assert _detect_format_from_extension("data.xml", ["xml"], "line") == "xml"
    assert _detect_format_from_extension("data.XML", ["xml"], "line") == "xml"

def test_xml_mode_integration(tmp_path, caplog):
    xml_file = tmp_path / "input.xml"
    xml_file.write_text("<root><w>one</w><w>two</w></root>")
    out_file = tmp_path / "output.txt"

    xml_mode(
        input_files=[str(xml_file)],
        output_file=str(out_file),
        min_length=1,
        max_length=100,
        process_output=True,
        key="w",
        output_format='line',
        quiet=True
    )

    assert out_file.read_text().strip() == "one\ntwo"

def test_extract_xml_error_handling(tmp_path, caplog):
    # Empty file
    empty_file = tmp_path / "empty.xml"
    empty_file.write_text("")
    items = list(_extract_xml_items(str(empty_file), "key"))
    assert items == []

    # Invalid XML
    bad_xml = tmp_path / "bad.xml"
    bad_xml.write_text("<root>unclosed tag")
    with caplog.at_level("ERROR"):
        items = list(_extract_xml_items(str(bad_xml), "key"))
        assert items == []
        assert "Failed to parse XML" in caplog.text
