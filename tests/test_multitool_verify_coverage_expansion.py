import subprocess
import os
import pytest
import re
import logging
import multitool

def run_multitool(args, input_text=None):
    """Helper to run the multitool CLI and return the result with stripped ANSI codes."""
    cmd = ["python3", "multitool.py"] + args
    result = subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    # Strip ANSI escape codes
    result.stdout = re.sub(r'\x1b\[[0-9;]*m', '', result.stdout)
    result.stderr = re.sub(r'\x1b\[[0-9;]*m', '', result.stderr)
    return result

def test_verify_no_mapping_error_exit(tmp_path, caplog):
    """Test that verify_mode exits with error if no mapping is provided."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("content", encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as excinfo:
            multitool.verify_mode(
                input_files=[str(input_file)],
                mapping_file=None,
                output_file='-',
                min_length=3,
                max_length=100,
                process_output=False,
                ad_hoc=[]
            )
        assert excinfo.value.code == 1
        assert "No mapping provided to verify" in caplog.text

def test_verify_raw_mode_direct_matching(tmp_path, capsys):
    """Test that --raw (clean_items=False) correctly matches items with numbers and case."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("Teh123 world SomeTehWord123")

    multitool.verify_mode(
        input_files=[str(input_file)],
        mapping_file=None,
        output_file='-',
        min_length=3,
        max_length=100,
        process_output=False,
        clean_items=False,
        smart=True,
        ad_hoc=["Teh123:The", "Teh:The"]
    )
    captured = capsys.readouterr()
    assert "Entries found in files:   2" in re.sub(r'\x1b\[[0-9;]*m', '', captured.out)

def test_verify_prune_with_limit_and_sorting_direct(tmp_path, capsys):
    """Test prune mode with limit and sorting (process_output) enabled."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("a b c")

    multitool.verify_mode(
        input_files=[str(input_file)],
        mapping_file=None,
        output_file='-',
        min_length=1,
        max_length=100,
        process_output=True,
        prune=True,
        limit=2,
        ad_hoc=["c:C", "a:A", "b:B"],
        clean_items=False
    )
    captured = capsys.readouterr()
    clean_out = re.sub(r'\x1b\[[0-9;]*m', '', captured.out)
    assert "a -> A" in clean_out
    assert "b -> B" in clean_out
    assert "c -> C" not in clean_out

def test_verify_report_with_limit_and_sorting_direct(tmp_path, capsys):
    """Test human-readable report with limit and sorting enabled for missing entries."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("found")

    multitool.verify_mode(
        input_files=[str(input_file)],
        mapping_file=None,
        output_file='-',
        min_length=1,
        max_length=100,
        process_output=True,
        prune=False,
        limit=2,
        ad_hoc=["found:FOUND", "zmissing:Z", "amissing:A", "mmissing:M"],
        clean_items=True
    )
    captured = capsys.readouterr()
    clean_out = re.sub(r'\x1b\[[0-9;]*m', '', captured.out)
    assert "Entries missing:          3" in clean_out
    assert "  - amissing" in clean_out
    assert "  - mmissing" in clean_out
    assert "... and 1 more." in clean_out

def test_verify_early_exit_multiple_files(tmp_path, capsys):
    """Test that verify_mode exits early when all keys are found (line 4078)."""
    file1 = tmp_path / "file1.txt"
    file1.write_text("teh")
    file2 = tmp_path / "file2.txt"
    file2.write_text("wrld") # This file should not even be opened if teh is the only key

    multitool.verify_mode(
        input_files=[str(file1), str(file2)],
        mapping_file=None,
        output_file='-',
        min_length=1,
        max_length=100,
        process_output=False,
        ad_hoc=["teh:the"]
    )
    captured = capsys.readouterr()
    assert "Entries found in files:   1" in captured.out

def test_verify_raw_mode_cli_integration(tmp_path):
    """Verify raw mode works via CLI flags."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("Teh123 world")
    result = run_multitool(["verify", str(input_file), "--add", "Teh123:The", "--raw"])
    assert "Entries found in files:   1" in result.stdout

def test_verify_prune_limit_sorting_cli_integration(tmp_path):
    """Verify prune mode with limit and sorting works via CLI flags."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("a b c")
    result = run_multitool(["verify", str(input_file), "--add", "c:C", "a:A", "b:B", "--prune", "--process-output", "--limit", "2"])
    assert "a -> A" in result.stdout
    assert "b -> B" in result.stdout
    assert "c -> C" not in result.stdout

def test_verify_report_limit_sorting_cli_integration(tmp_path):
    """Verify report mode with limit and sorting works via CLI flags."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("found")
    result = run_multitool([
        "verify", str(input_file),
        "--add", "found:FOUND", "zmissing:Z", "amissing:A", "mmissing:M",
        "--process-output", "--limit", "2"
    ])
    assert "Entries missing:          3" in result.stdout
    assert "  - amissing" in result.stdout
    assert "  - mmissing" in result.stdout
    assert "... and 1 more." in result.stdout
