"""
tests/test_gentypos_xml.py

Tests for XML output format support in gentypos.py.
"""

import xml.etree.ElementTree as ET
import sys
from unittest.mock import patch

from gentypos import format_typos, _detect_format_from_extension, main


def test_format_typos_xml():
    typo_map = {
        "helo": "hello",
        "wrod": "world",
    }
    result_lines = format_typos(typo_map, "xml")
    xml_str = "\n".join(result_lines)

    assert "<root>" in xml_str
    assert "</root>" in xml_str
    assert "<typo>helo</typo>" in xml_str
    assert "<correction>hello</correction>" in xml_str
    assert "<typo>wrod</typo>" in xml_str
    assert "<correction>world</correction>" in xml_str

    # Validate XML structure
    root = ET.fromstring(xml_str)
    assert root.tag == "root"
    items = root.findall("item")
    assert len(items) == 2
    assert items[0].find("typo").text == "helo"
    assert items[0].find("correction").text == "hello"


def test_detect_xml_extension():
    allowed = ["arrow", "csv", "table", "toml", "list", "json", "yaml", "markdown", "md", "html", "htm", "xml"]
    assert _detect_format_from_extension("output.xml", allowed, "arrow") == "xml"
    assert _detect_format_from_extension("report.XML", allowed, "arrow") == "xml"


def test_cli_format_xml(tmp_path):
    output_xml = tmp_path / "output.xml"
    test_args = [
        "gentypos.py",
        "hello",
        "--format", "xml",
        "--output", str(output_xml),
        "--no-filter",
        "-t",
    ]

    with patch.object(sys, "argv", test_args):
        main()

    assert output_xml.exists()
    content = output_xml.read_text(encoding="utf-8")
    assert "<root>" in content
    assert "<typo>" in content
    assert "<correction>hello</correction>" in content

    root = ET.fromstring(content)
    assert root.tag == "root"
    assert len(root.findall("item")) > 0


def test_cli_auto_detect_xml_extension(tmp_path):
    output_xml = tmp_path / "auto.xml"
    test_args = [
        "gentypos.py",
        "world",
        "--output", str(output_xml),
        "--no-filter",
        "-t",
    ]

    with patch.object(sys, "argv", test_args):
        main()

    assert output_xml.exists()
    content = output_xml.read_text(encoding="utf-8")
    assert "<root>" in content

    root = ET.fromstring(content)
    assert root.tag == "root"
