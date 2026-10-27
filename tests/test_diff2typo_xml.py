import os
import xml.etree.ElementTree as ET
from diff2typo import format_typos


def test_format_typos_xml_paired_and_single():
    typos = ["teh -> the", "recieve -> receive", "unmatched"]
    lines = format_typos(typos, output_format="xml")
    xml_str = "\n".join(lines)

    assert xml_str.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    root = ET.fromstring(xml_str)
    assert root.tag == "typos"

    pairs = root.findall("pair")
    assert len(pairs) == 2
    assert pairs[0].find("typo").text == "teh"
    assert pairs[0].find("correction").text == "the"
    assert pairs[1].find("typo").text == "recieve"
    assert pairs[1].find("correction").text == "receive"

    items = root.findall("item")
    assert len(items) == 1
    assert items[0].find("typo").text == "unmatched"


def test_format_typos_xml_escaping():
    typos = ["foo&bar -> baz<qux>"]
    lines = format_typos(typos, output_format="xml")
    xml_str = "\n".join(lines)

    root = ET.fromstring(xml_str)
    pair = root.find("pair")
    assert pair.find("typo").text == "foo&bar"
    assert pair.find("correction").text == "baz<qux>"


def test_diff2typo_xml_auto_detection(tmp_path):
    output_file = tmp_path / "typos.xml"
    diff_file = tmp_path / "test.diff"
    diff_file.write_text("""--- a/file.py
+++ b/file.py
@@ -1,1 +1,1 @@
-teh
+the
""", encoding="utf-8")

    words_file = tmp_path / "words.csv"
    words_file.write_text("the\nreceive\n", encoding="utf-8")

    allowed_file = tmp_path / "allowed.csv"
    allowed_file.write_text("", encoding="utf-8")

    os.system(f"python3 diff2typo.py {diff_file} -o {output_file} -d {words_file} -a {allowed_file} -q")

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "<?xml" in content
    root = ET.fromstring(content)
    assert root.tag == "typos"
    assert root.find("pair/typo").text == "teh"
    assert root.find("pair/correction").text == "the"
