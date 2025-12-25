import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

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


def test_csv_mode_delimiter(tmp_path):
    # Comma-delimited
    input_comma = tmp_path / "input_comma.csv"
    input_comma.write_text("typo,correct\nword,another")
    output_comma = tmp_path / "output_comma.txt"
    multitool.csv_mode(str(input_comma), str(output_comma), 1, 10, True, delimiter=',')
    assert output_comma.read_text().splitlines() == ["another", "correct"]

    # Tab-delimited
    input_tab = tmp_path / "input_tab.tsv"
    input_tab.write_text("typo\tcorrect\nword\tanother")
    output_tab = tmp_path / "output_tab.txt"
    multitool.csv_mode(str(input_tab), str(output_tab), 1, 10, True, delimiter='\t')
    assert output_tab.read_text().splitlines() == ["another", "correct"]

    # Pipe-delimited
    input_pipe = tmp_path / "input_pipe.psv"
    input_pipe.write_text("typo|correct\nword|another")
    output_pipe = tmp_path / "output_pipe.txt"
    multitool.csv_mode(str(input_pipe), str(output_pipe), 1, 10, True, delimiter='|')
    assert output_pipe.read_text().splitlines() == ["another", "correct"]


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


@pytest.mark.performance
def test_filter_fragments_mode_performance(tmp_path):
    # Create large temporary files for performance testing
    num_words = 1000
    word_length = 10

    list1_path = tmp_path / "perf_list1.txt"
    list2_path = tmp_path / "perf_list2.txt"
    output_path = tmp_path / "perf_output.txt"

    # Generate a list of unique words
    words1 = {f"word{i}" for i in range(num_words)}
    # Generate a larger set of words for the second list, ensuring some overlap
    words2 = {f"longerword{i}" for i in range(num_words * 2)}

    with open(list1_path, "w") as f:
        for word in words1:
            f.write(word + "\n")

    with open(list2_path, "w") as f:
        for word in words2:
            f.write(word + "\n")

    import time

    start_time = time.time()
    multitool.filter_fragments_mode(str(list1_path), str(list2_path), str(output_path), 1, 20, True)
    end_time = time.time()

    duration = end_time - start_time
    print(f"filter_fragments_mode execution time: {duration:.4f} seconds")

    # This is not a strict performance assertion, but rather a check to ensure
    # the function completes within a reasonable timeframe. The actual performance
    # gain is hard to assert without a baseline, but this test will fail on
    # significant regressions.
    assert duration < 1.0  # e.g., assert it runs in less than 1 second


def test_check_mode(tmp_path):
    csv_file = tmp_path / "typos.csv"
    csv_file.write_text("mispelled,misspelled\nteh,the\nfoo,bar,foo\nbar,foo\n")
    output_file = tmp_path / "output.txt"
    multitool.check_mode(str(csv_file), str(output_file), 3, 10, True)
    assert output_file.read_text().splitlines() == ["bar", "foo"]


def test_combine_mode(tmp_path):
    file_a = tmp_path / "file_a.txt"
    file_b = tmp_path / "file_b.txt"
    file_a.write_text("Alpha\nBeta\nAlpha\n")
    file_b.write_text("gamma\ndelta\nbeta\n")

    output_file = tmp_path / "combined.txt"
    multitool.combine_mode(
        [str(file_a), str(file_b)],
        str(output_file),
        1,
        10,
        False,
    )

    assert output_file.read_text().splitlines() == [
        "alpha",
        "beta",
        "delta",
        "gamma",
    ]


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


def test_set_operation_invalid_operation(tmp_path):
    file_a = tmp_path / "file_a.txt"
    file_b = tmp_path / "file_b.txt"
    file_a.write_text("Alpha\n")
    file_b.write_text("Beta\n")

    with pytest.raises(ValueError):
        multitool.set_operation_mode(
            str(file_a),
            str(file_b),
            str(tmp_path / "out.txt"),
            1,
            10,
            False,
            'invalid',
        )


def test_main_min_length_validation(monkeypatch, tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("data\n")
    output_file = tmp_path / "output.txt"

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'multitool.py',
            'line',
            '--input',
            str(input_file),
            '--output',
            str(output_file),
            '--min-length',
            '0',
            '--max-length',
            '5',
        ],
    )

    with pytest.raises(SystemExit):
        multitool.main()


def test_main_max_less_than_min(monkeypatch, tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("data\n")
    output_file = tmp_path / "output.txt"

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'multitool.py',
            'line',
            '--input',
            str(input_file),
            '--output',
            str(output_file),
            '--min-length',
            '5',
            '--max-length',
            '4',
        ],
    )

    with pytest.raises(SystemExit):
        multitool.main()


def test_main_set_operation_integration(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("Alpha\nBeta\n")
    file_b.write_text("beta\nGamma\n")

    output_file = tmp_path / "union.txt"

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'multitool.py',
            'set_operation',
            '--input',
            str(file_a),
            '--output',
            str(output_file),
            '--file2',
            str(file_b),
            '--operation',
            'union',
            '--min-length',
            '1',
            '--max-length',
            '10',
            '--process-output',
        ],
    )

    multitool.main()

    assert output_file.read_text().splitlines() == ['alpha', 'beta', 'gamma']


def test_backtick_mode_context_markers(tmp_path):
    # Verify that warning and note markers are respected
    input_file = tmp_path / "context.txt"
    input_file.write_text(
        "`noise` warning: `warning`\n"
        "`noise` note: `note`\n"
        "`noise` error: `error`\n"
        "no marker `fallback`\n"
    )
    output_file = tmp_path / "output.txt"
    multitool.backtick_mode(str(input_file), str(output_file), 1, 20, False)
    assert output_file.read_text().splitlines() == ["warning", "note", "error", "fallback"]

def test_backtick_mode_marker_inside_backticks(tmp_path):
    # Regression test: Markers inside backticks should not trigger extraction of subsequent text
    input_file = tmp_path / "bug.txt"
    input_file.write_text("Do not use `error:` as a variable name.\n")
    output_file = tmp_path / "output.txt"

    # We use a large max_length to ensure we don't filter out the long result
    multitool.backtick_mode(str(input_file), str(output_file), 1, 100, False)

    content = output_file.read_text().strip()

    # Should fall back to extracting "error:" -> "error"
    assert content == "error"


def test_detect_encoding(caplog, monkeypatch, tmp_path):
    # Create a dummy file
    f = tmp_path / "test.txt"
    f.write_text("dummy")

    # Case 1: chardet not available
    monkeypatch.setattr(multitool, "_CHARDET_AVAILABLE", False)
    with caplog.at_level(logging.WARNING):
        assert multitool.detect_encoding(str(f)) is None
    assert "chardet not installed" in caplog.text
    caplog.clear()

    # Case 2: chardet available, low confidence
    monkeypatch.setattr(multitool, "_CHARDET_AVAILABLE", True)
    mock_chardet = MagicMock()
    mock_chardet.detect.return_value = {'encoding': 'utf-8', 'confidence': 0.3}
    monkeypatch.setattr(multitool, "chardet", mock_chardet)

    with caplog.at_level(logging.WARNING):
        assert multitool.detect_encoding(str(f)) is None
    assert "Failed to reliably detect" in caplog.text
    caplog.clear()

    # Case 3: chardet available, high confidence
    mock_chardet.detect.return_value = {'encoding': 'utf-8', 'confidence': 0.9}
    with caplog.at_level(logging.INFO):
        assert multitool.detect_encoding(str(f)) == 'utf-8'
    assert "Detected encoding 'utf-8'" in caplog.text
