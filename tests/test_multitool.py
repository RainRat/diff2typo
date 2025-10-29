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


def test_load_and_clean_file(tmp_path):
    data_file = tmp_path / "data.txt"
    data_file.write_text("Alpha\nbeta\nAlpha!!!\nGamma delta\n")

    raw_items, cleaned_items, unique_items = multitool._load_and_clean_file(
        str(data_file),
        5,
        10,
    )

    assert raw_items == ["Alpha", "beta", "Alpha!!!", "Gamma delta"]
    assert cleaned_items == ["alpha", "alpha", "gammadelta"]
    assert unique_items == ["alpha", "gammadelta"]

    raw_ws, cleaned_ws, unique_ws = multitool._load_and_clean_file(
        str(data_file),
        1,
        10,
        split_whitespace=True,
        apply_length_filter=False,
    )

    assert raw_ws == ["Alpha", "beta", "Alpha!!!", "Gamma", "delta"]
    assert cleaned_ws == ["alpha", "beta", "alpha", "gamma", "delta"]
    assert unique_ws == ["alpha", "beta", "gamma", "delta"]


def test_arrow_mode(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello -> world\nfoo -> bar\nnoarrow here\n")
    output_file = tmp_path / "output.txt"
    multitool.arrow_mode(str(input_file), str(output_file), 1, 10, True)
    assert output_file.read_text().splitlines() == ["foo", "hello"]


def test_backtick_mode(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text(
        "prefix `data` suffix\n"
        "no backtick here\n"
        "multiple `words here` please\n"
        "path Tests_for_ty's_`instbuiltins`_-_Built-ins error: `mdtest` should be `mstest`\n"
    )
    output_file = tmp_path / "output.txt"
    multitool.backtick_mode(str(input_file), str(output_file), 1, 20, False)
    assert output_file.read_text().splitlines() == ["data", "wordshere", "mdtest"]


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


def test_set_operation_mode(tmp_path):
    file_a = tmp_path / "file_a.txt"
    file_b = tmp_path / "file_b.txt"
    file_a.write_text("Alpha\nBeta\nGamma\nAlpha\n")
    file_b.write_text("beta\nDelta\nGamma\n")

    intersection_output = tmp_path / "intersection.txt"
    multitool.set_operation_mode(
        str(file_a),
        str(file_b),
        str(intersection_output),
        1,
        10,
        False,
        'intersection',
    )
    assert intersection_output.read_text().splitlines() == ["beta", "gamma"]

    union_output = tmp_path / "union.txt"
    multitool.set_operation_mode(
        str(file_a),
        str(file_b),
        str(union_output),
        1,
        10,
        True,
        'union',
    )
    assert union_output.read_text().splitlines() == ["alpha", "beta", "delta", "gamma"]

    difference_output = tmp_path / "difference.txt"
    multitool.set_operation_mode(
        str(file_a),
        str(file_b),
        str(difference_output),
        1,
        10,
        False,
        'difference',
    )
    assert difference_output.read_text().splitlines() == ["alpha"]
