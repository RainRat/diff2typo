import os
import json
import yaml
import pytest
from multitool import convert_mode, flatten_mode, _yield_structured_docs

def test_convert_xml_to_json(tmp_path):
    input_file = tmp_path / "input.xml"
    output_file = tmp_path / "output.json"

    xml_content = """<root>
    <title>Test Title</title>
    <author>Jane Doe</author>
</root>"""
    input_file.write_text(xml_content)

    convert_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        output_format='json'
    )

    assert output_file.exists()
    with open(output_file, 'r') as f:
        output_data = json.load(f)

    assert output_data == {
        "root": {
            "title": "Test Title",
            "author": "Jane Doe"
        }
    }

def test_convert_xml_with_key_extraction(tmp_path):
    input_file = tmp_path / "input.xml"
    output_file = tmp_path / "output.json"

    xml_content = """<data>
    <items>
        <item>first</item>
        <item>second</item>
    </items>
</data>"""
    input_file.write_text(xml_content)

    convert_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        key="data.items",
        output_format='json'
    )

    assert output_file.exists()
    with open(output_file, 'r') as f:
        output_data = json.load(f)

    assert output_data == {"item": ["first", "second"]}

def test_flatten_xml(tmp_path):
    input_file = tmp_path / "config.xml"
    output_file = tmp_path / "output.txt"

    xml_content = """<config>
    <settings>
        <theme>dark</theme>
        <fontSize>14</fontSize>
    </settings>
</config>"""
    input_file.write_text(xml_content)

    flatten_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        clean_items=False,
        output_format='line'
    )

    assert output_file.exists()
    content = output_file.read_text().splitlines()
    assert "config.settings.theme -> dark" in content
    assert "config.settings.fontSize -> 14" in content

def test_xml_attributes_handling(tmp_path):
    input_file = tmp_path / "item.xml"
    output_file = tmp_path / "output.json"

    xml_content = """<element id="123">Content</element>"""
    input_file.write_text(xml_content)

    convert_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        output_format='json'
    )

    with open(output_file, 'r') as f:
        output_data = json.load(f)

    assert output_data == {
        "element": {
            "@attributes": {"id": "123"},
            "#text": "Content"
        }
    }

def test_yield_structured_docs_xml(tmp_path):
    input_file = tmp_path / "data.xml"
    xml_content = "<root><status>ok</status></root>"
    input_file.write_text(xml_content)

    docs = list(_yield_structured_docs(str(input_file)))
    assert len(docs) == 1
    assert docs[0] == {"root": {"status": "ok"}}

def test_invalid_xml_handling(tmp_path, caplog):
    input_file = tmp_path / "invalid.xml"
    input_file.write_text("<root><unclosed></root>")

    docs = list(_yield_structured_docs(str(input_file)))
    assert len(docs) == 0
