import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool


@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)


def test_filter_to_letters():
    assert multitool.filter_to_letters("Hello, World!123") == "helloworld"


def test_clean_and_filter():
    items = ["Hello!", "A", "World123", ""]
    assert multitool.clean_and_filter(items, 3, 5) == ["hello", "world"]


def test_arrow_mode(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello -> world\nfoo -> bar\nnoarrow here\n")
    output_file = tmp_path / "output.txt"
    multitool.arrow_mode(str(input_file), str(output_file), 1, 10, True)
    assert output_file.read_text().splitlines() == ["foo", "hello"]


def test_backtick_mode(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("prefix `data` suffix\nno backtick here\nmultiple `words here` please\n")
    output_file = tmp_path / "output.txt"
    multitool.backtick_mode(str(input_file), str(output_file), 1, 20, False)
    assert output_file.read_text().splitlines() == ["data", "wordshere"]


def test_csv_mode(tmp_path):
    input_file = tmp_path / "input.csv"
    input_file.write_text("typo1,correct1\nword1,word2,word3\n")
    output_file = tmp_path / "output.txt"
    multitool.csv_mode(str(input_file), str(output_file), 1, 10, True, first_column=False)
    assert output_file.read_text().splitlines() == ["correct", "word"]
    output_file2 = tmp_path / "output2.txt"
    multitool.csv_mode(str(input_file), str(output_file2), 1, 10, True, first_column=True)
    assert output_file2.read_text().splitlines() == ["typo", "word"]


def test_line_mode(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("Hello\nWorld!\na\n")
    output_file = tmp_path / "output.txt"
    multitool.line_mode(str(input_file), str(output_file), 3, 10, True)
    assert output_file.read_text().splitlines() == ["hello", "world"]


def test_count_mode(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("Hello world hello HELLO test\nAnother line with world\n")
    output_file = tmp_path / "output.txt"
    multitool.count_mode(str(input_file), str(output_file), 5, 10, False)
    assert output_file.read_text().splitlines() == ["hello: 3", "world: 2", "another: 1"]


def test_filter_fragments_mode(tmp_path):
    list1 = tmp_path / "list1.txt"
    list1.write_text("apple\ncar\nplane\ncarpet\n")
    list2 = tmp_path / "list2.txt"
    list2.write_text("an applepie\ncarpeted floor\ncar\n")
    output_file = tmp_path / "output.txt"
    multitool.filter_fragments_mode(str(list1), str(list2), str(output_file), 1, 10, True)
    assert output_file.read_text().splitlines() == ["plane"]


def test_check_mode(tmp_path):
    csv_file = tmp_path / "typos.csv"
    csv_file.write_text("mispelled,misspelled\nteh,the\nfoo,bar,foo\nbar,foo\n")
    output_file = tmp_path / "output.txt"
    multitool.check_mode(str(csv_file), str(output_file), 3, 10, True)
    assert output_file.read_text().splitlines() == ["bar", "foo"]
