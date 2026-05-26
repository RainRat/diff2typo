import pytest
from multitool import to_case, case_mode
import io
import contextlib

def test_to_case():
    s = "helloWorld_test-case"
    assert to_case(s, "lower") == "helloworld_test-case"
    assert to_case(s, "upper") == "HELLOWORLD_TEST-CASE"
    assert to_case(s, "snake") == "hello_world_test_case"
    assert to_case(s, "camel") == "helloWorldTestCase"
    assert to_case(s, "pascal") == "HelloWorldTestCase"
    assert to_case(s, "kebab") == "hello-world-test-case"
    assert to_case(s, "title") == "Hello World Test Case"
    assert to_case(s, "constant") == "HELLO_WORLD_TEST_CASE"
    assert to_case(s, "sentence") == "Hello world test case"
    assert to_case(s, "unknown") == s

def test_to_case_simple():
    s = "hello"
    assert to_case(s, "upper") == "HELLO"
    assert to_case(s, "snake") == "hello"
    assert to_case(s, "pascal") == "Hello"

def test_to_case_numbers():
    s = "hello123world"
    assert to_case(s, "snake") == "hello_123_world"
    assert to_case(s, "camel") == "hello123World"

def test_case_mode_basic(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("helloWorld\ntest-case\n")
    output_file = tmp_path / "output.txt"

    case_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        to='snake',
        pairs=False,
        output_format='line',
        quiet=True,
        clean_items=True
    )

    assert output_file.read_text() == "hello_world\ntest_case\n"

def test_case_mode_pairs(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("helloWorld\n")
    output_file = tmp_path / "output.txt"

    case_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        to='pascal',
        pairs=True,
        output_format='line',
        quiet=True,
        clean_items=True
    )

    assert output_file.read_text() == "helloWorld -> HelloWorld\n"

def test_case_mode_filter(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a\nabc\nabcdef\n")
    output_file = tmp_path / "output.txt"

    case_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=2,
        max_length=4,
        process_output=False,
        to='upper',
        pairs=False,
        output_format='line',
        quiet=True,
        clean_items=True
    )

    assert output_file.read_text() == "ABC\n"
