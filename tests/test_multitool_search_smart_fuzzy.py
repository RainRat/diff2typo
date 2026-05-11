import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multitool import search_mode

def test_search_smart_keyboard_filter(tmp_path):
    input_file = tmp_path / "smart_keyboard.txt"
    input_file.write_text("ThisIsMyWord", encoding='utf-8')
    output_file = tmp_path / "out.txt"

    search_mode(
        input_files=[str(input_file)],
        query="eord",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        max_dist=1,
        smart=True,
        keyboard=True,
        clean_items=True
    )

    content = output_file.read_text(encoding='utf-8')
    assert "ThisIsMyWord" in content

def test_search_smart_keyboard_filter_negative(tmp_path):
    input_file = tmp_path / "smart_keyboard_neg.txt"
    input_file.write_text("ThisIsMyWord", encoding='utf-8')
    output_file = tmp_path / "out.txt"

    search_mode(
        input_files=[str(input_file)],
        query="pord",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        max_dist=1,
        smart=True,
        keyboard=True,
        clean_items=True
    )

    content = output_file.read_text(encoding='utf-8')
    assert "ThisIsMyWord" not in content

def test_search_smart_transposition_filter(tmp_path):
    input_file = tmp_path / "smart_trans.txt"
    input_file.write_text("MyTehWord", encoding='utf-8')
    output_file = tmp_path / "out.txt"

    search_mode(
        input_files=[str(input_file)],
        query="the",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        max_dist=2,
        smart=True,
        transposition=True,
        clean_items=True
    )

    content = output_file.read_text(encoding='utf-8')
    assert "MyTehWord" in content

def test_search_smart_transposition_filter_negative(tmp_path):
    input_file = tmp_path / "smart_trans_neg.txt"
    input_file.write_text("MyToeWord", encoding='utf-8')
    output_file = tmp_path / "out.txt"

    search_mode(
        input_files=[str(input_file)],
        query="the",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        max_dist=2,
        smart=True,
        transposition=True,
        clean_items=True
    )

    content = output_file.read_text(encoding='utf-8')
    assert "MyToeWord" not in content
