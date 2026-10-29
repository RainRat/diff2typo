import os
import pytest
import gentypos


def test_load_substitutions_file_xml_typostats_children(tmp_path):
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<report>
    <replacements>
        <item>
            <correct>ph</correct>
            <typo>f</typo>
        </item>
        <item>
            <correct>e</correct>
            <typo>a</typo>
        </item>
    </replacements>
</report>
"""
    xml_file = tmp_path / "subs.xml"
    xml_file.write_text(xml_content, encoding="utf-8")

    subs = gentypos._load_substitutions_file(str(xml_file))
    assert "ph" in subs
    assert "f" in subs["ph"]
    assert "e" in subs
    assert "a" in subs["e"]


def test_load_substitutions_file_xml_attributes(tmp_path):
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<substitutions>
    <rule key="th" value="teh" />
    <rule key="ing" value="in" />
</substitutions>
"""
    xml_file = tmp_path / "subs_attrs.xml"
    xml_file.write_text(xml_content, encoding="utf-8")

    subs = gentypos._load_substitutions_file(str(xml_file))
    assert "th" in subs
    assert "teh" in subs["th"]
    assert "ing" in subs
    assert "in" in subs["ing"]


def test_load_substitutions_file_xml_element_mapping(tmp_path):
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<mappings>
    <ph>f</ph>
    <qu>k</qu>
</mappings>
"""
    xml_file = tmp_path / "subs_elements.xml"
    xml_file.write_text(xml_content, encoding="utf-8")

    subs = gentypos._load_substitutions_file(str(xml_file))
    assert "ph" in subs
    assert "f" in subs["ph"]
    assert "qu" in subs
    assert "k" in subs["qu"]


def test_gentypos_xml_substitutions_end_to_end(tmp_path):
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<report>
    <item>
        <correct>ph</correct>
        <typo>f</typo>
    </item>
</report>
"""
    sub_file = tmp_path / "sub.xml"
    sub_file.write_text(xml_content, encoding="utf-8")

    # Run typo generation for 'phrase' with custom substitution ph->f
    word = "phrase"
    settings = gentypos.SimpleNamespace(
        custom_substitutions_config={},
        substitutions_file=str(sub_file),
        ad_hoc=None,
        enable_custom_substitutions=True,
        enable_adjacent_substitutions=False,
    )

    adjacent_keys, custom_subs = gentypos._setup_generation_tools(settings)
    assert "ph" in custom_subs
    assert "f" in custom_subs["ph"]

    typos = gentypos.generate_typos_by_replacement(
        word, adjacent_keys, custom_subs, use_adjacent=False, use_custom=True
    )
    assert "frase" in typos
