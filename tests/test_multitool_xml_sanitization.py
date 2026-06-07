import xml.etree.ElementTree as ET
from multitool import _write_structured_data

def test_xml_sanitization_tags(tmp_path):
    """Test that XML tags are correctly sanitized when they start with invalid characters or contain them."""
    data = {
        "1": "numeric",
        ".dot": "starts with dot",
        "-dash": "starts with dash",
        "valid": "already valid",
        "a b": "contains space",
        "a!b": "contains exclamation",
        "": "empty string"
    }
    # root_tag starting with a digit
    root_tag = "0root"

    output_file = tmp_path / "test.xml"
    _write_structured_data(data, str(output_file), output_format='xml', root_tag=root_tag)
    xml_content = output_file.read_text()

    # Verify the root tag was sanitized: 0root -> _0root
    assert "<_0root>" in xml_content
    assert "</_0root>" in xml_content

    # Parse to verify structure and content
    root = ET.fromstring(xml_content)
    assert root.tag == "_0root"

    # 1 -> _1
    assert root.find("_1").text == "numeric"
    # .dot -> _.dot
    assert root.find("_.dot").text == "starts with dot"
    # -dash -> _-dash
    assert root.find("_-dash").text == "starts with dash"
    # valid -> valid
    assert root.find("valid").text == "already valid"
    # a b -> a_b
    # a!b -> a_b
    # Since there are two keys that map to a_b, findall should find two
    a_b_elements = root.findall("a_b")
    assert len(a_b_elements) == 2
    texts = sorted([e.text for e in a_b_elements])
    assert texts == ["contains exclamation", "contains space"]

    # "" -> _
    assert root.find("_").text == "empty string"

def test_xml_sanitization_root_only(tmp_path):
    """Test that root tag is sanitized even if data is simple."""
    data = "simple value"
    root_tag = "123"

    output_file = tmp_path / "test2.xml"
    _write_structured_data(data, str(output_file), output_format='xml', root_tag=root_tag)
    xml_content = output_file.read_text()

    root = ET.fromstring(xml_content)
    assert root.tag == "_123"
    assert root.text == "simple value"

def test_xml_sanitization_list(tmp_path):
    """Test that lists in XML are correctly handled with 'item' tags."""
    data = {"list": ["a", "b"]}
    root_tag = "root"

    output_file = tmp_path / "test3.xml"
    _write_structured_data(data, str(output_file), output_format='xml', root_tag=root_tag)
    xml_content = output_file.read_text()

    root = ET.fromstring(xml_content)
    list_elem = root.find("list")
    items = list_elem.findall("item")
    assert len(items) == 2
    assert items[0].text == "a"
    assert items[1].text == "b"
