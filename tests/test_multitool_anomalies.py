import pytest
from multitool import anomalies_mode
import io
import contextlib

def test_anomalies_mode_basic(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("HEllo world gIT pyTHon w0rd Normal Word")

    output_file = tmp_path / "results.txt"

    anomalies_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        output_format='line',
        quiet=True
    )

    content = output_file.read_text()
    assert "HEllo -> [Shift]" in content
    assert "gIT -> [Caps]" in content
    assert "pyTHon -> [Bumpy]" in content
    assert "w0rd -> [Num]" in content
    assert "Normal" not in content
    assert "Word" not in content

def test_anomalies_mode_filtering(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("HEllo w0rd")

    output_file = tmp_path / "results.txt"

    # Filter out w0rd by length
    anomalies_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=5,
        max_length=100,
        process_output=True,
        output_format='line',
        quiet=True
    )

    content = output_file.read_text()
    assert "HEllo -> [Shift]" in content
    assert "w0rd" not in content

def test_anomalies_mode_empty(tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("Normal words only")

    output_file = tmp_path / "results.txt"

    anomalies_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        output_format='line',
        quiet=True
    )

    content = output_file.read_text()
    assert content.strip() == ""
