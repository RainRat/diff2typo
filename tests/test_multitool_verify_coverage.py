import pytest
import multitool
from unittest.mock import MagicMock, patch

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable=None, *_, **__: iterable if iterable is not None else MagicMock())

def test_verify_mode_no_mapping(tmp_path, caplog):
    # Tests lines 4068-4070
    input_file = tmp_path / "input.txt"
    input_file.write_text("content")

    with pytest.raises(SystemExit) as excinfo:
        multitool.verify_mode(
            input_files=[str(input_file)],
            mapping_file=None,
            output_file="-",
            min_length=1,
            max_length=100,
            process_output=False,
            ad_hoc=None
        )
    assert excinfo.value.code == 1
    assert "No mapping provided to verify" in caplog.text

def test_verify_mode_partial_matches(tmp_path):
    # Tests lines 4131-4159
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh quick brown fox")

    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("teh -> the\nmissing1 -> m1\nmissing2 -> m2")

    output_file = tmp_path / "report.txt"

    # We use clean_items=False to avoid filter_to_letters removing the '1' and '2' in keys
    multitool.verify_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        ad_hoc=None,
        clean_items=False
    )

    content = output_file.read_text()
    assert "VERIFICATION REPORT" in content
    assert "Total entries in mapping: 3" in content
    assert "Entries found in files:   1" in content
    assert "Entries missing:          2" in content
    assert "MISSING ENTRIES:" in content
    assert "  - missing1" in content
    assert "  - missing2" in content

def test_verify_mode_all_verified(tmp_path):
    # Tests line 4139 (All entries verified branch)
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh")

    output_file = tmp_path / "report.txt"

    multitool.verify_mode(
        input_files=[str(input_file)],
        mapping_file=None,
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        ad_hoc=["teh:the"]
    )

    content = output_file.read_text()
    assert "All entries verified." in content
    assert "MISSING ENTRIES:" not in content

def test_verify_mode_smart(tmp_path):
    # Tests lines 4090-4099
    input_file = tmp_path / "input.txt"
    input_file.write_text("tehWord") # Smart splitting needed to find 'teh'

    output_file = tmp_path / "report.txt"

    # Without smart
    multitool.verify_mode(
        input_files=[str(input_file)],
        mapping_file=None,
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        ad_hoc=["teh:the"],
        smart=False
    )
    assert "Entries found in files:   0" in output_file.read_text()

    # With smart
    multitool.verify_mode(
        input_files=[str(input_file)],
        mapping_file=None,
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        ad_hoc=["teh:the"],
        smart=True
    )
    assert "Entries found in files:   1" in output_file.read_text()

def test_verify_mode_prune(tmp_path):
    # Tests lines 4107-4129
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh")

    output_file = tmp_path / "pruned.txt"

    # With prune and process_output (sorting)
    multitool.verify_mode(
        input_files=[str(input_file)],
        mapping_file=None,
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        ad_hoc=["teh:the", "unused:u"],
        prune=True
    )

    content = output_file.read_text()
    assert "teh" in content and "the" in content
    assert "unused" not in content

def test_verify_mode_prune_limit(tmp_path):
    # Tests line 4114
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh recieve")

    output_file = tmp_path / "pruned.txt"

    multitool.verify_mode(
        input_files=[str(input_file)],
        mapping_file=None,
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        ad_hoc=["teh:the", "recieve:receive"],
        prune=True,
        limit=1
    )

    content = output_file.read_text()
    # Check that we have headers and exactly one data row
    # Data rows in arrow format have " │ "
    data_lines = [line for line in content.splitlines() if " │ " in line and "Left" not in line]
    assert len(data_lines) == 1

def test_verify_mode_early_break(tmp_path):
    # Tests early breaks in loops (lines 4077-4078, 4087-4088, 4096-4101)
    file1 = tmp_path / "file1.txt"
    file1.write_text("teh")
    file2 = tmp_path / "file2.txt"
    file2.write_text("unused")

    output_file = tmp_path / "report.txt"

    # Mock _read_file_lines_robust to verify it's NOT called for file2
    with patch("multitool._read_file_lines_robust", side_effect=multitool._read_file_lines_robust) as mock_read:
        multitool.verify_mode(
            input_files=[str(file1), str(file2)],
            mapping_file=None,
            output_file=str(output_file),
            min_length=1,
            max_length=100,
            process_output=False,
            ad_hoc=["teh:the"]
        )

    # Check that file2 was NOT read
    called_files = [call.args[0] for call in mock_read.call_args_list]
    assert str(file1) in called_files
    assert str(file2) not in called_files

def test_verify_mode_missing_limit(tmp_path):
    # Tests lines 4150-4152
    input_file = tmp_path / "input.txt"
    input_file.write_text("nothing")

    output_file = tmp_path / "report.txt"

    multitool.verify_mode(
        input_files=[str(input_file)],
        mapping_file=None,
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        ad_hoc=["ma:1", "mb:2", "mc:3"],
        limit=2,
        clean_items=False
    )

    content = output_file.read_text()
    assert "  - ma" in content
    assert "  - mb" in content
    assert "  - mc" not in content
    assert "... and 1 more." in content
