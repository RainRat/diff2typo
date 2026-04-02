import logging
from multitool import standardize_mode

def test_standardize_mode_basic(tmp_path):
    # Setup test file with inconsistent casing
    # "Database" appears 3 times, "database" 2 times, "DATABASE" 1 time.
    # "Winner" should be "Database".
    content = """
    Database is a Database.
    database and database.
    Database!
    DATABASE.
    """
    test_file = tmp_path / "test.txt"
    test_file.write_text(content)

    output_file = tmp_path / "output.txt"

    # Run standardize mode
    standardize_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=False,
        quiet=True,
        clean_items=True,
    )

    result = output_file.read_text()

    # Check that all variations were replaced by "Database"
    assert "Database is a Database." in result
    assert "Database and Database." in result
    assert "Database!" in result
    assert "Database." in result
    assert "database" not in result
    assert "DATABASE" not in result

def test_standardize_mode_in_place(tmp_path):
    # Setup test file
    # "word" (2), "WORD" (1) -> "word" wins
    content = "word word WORD"
    test_file = tmp_path / "test.txt"
    test_file.write_text(content)

    # Run standardize mode in-place
    standardize_mode(
        input_files=[str(test_file)],
        output_file="-",
        min_length=1,
        max_length=100,
        process_output=False,
        quiet=True,
        clean_items=True,
        in_place="", # No extension for backup
    )

    result = test_file.read_text().strip()
    assert result == "word word word"

def test_standardize_mode_smart_split(tmp_path):
    # Setup test file with CamelCase
    # "SmartSplit" (2), "smartsplit" (1) -> "SmartSplit" wins
    # "SubWord" (2), "subword" (1) -> "SubWord" wins
    content = "SmartSplit SmartSplit smartsplit SubWord SubWord subword"
    test_file = tmp_path / "test.txt"
    test_file.write_text(content)

    output_file = tmp_path / "output.txt"

    standardize_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=False,
        quiet=True,
        clean_items=True,
    )

    result = output_file.read_text()
    assert "SmartSplit SmartSplit SmartSplit" in result
    assert "SubWord SubWord SubWord" in result

def test_standardize_mode_dry_run(tmp_path, caplog):
    # Setup test file
    content = "word word WORD"
    test_file = tmp_path / "test.txt"
    test_file.write_text(content)

    with caplog.at_level(logging.WARNING):
        # Run standardize mode dry-run
        standardize_mode(
            input_files=[str(test_file)],
            output_file="-",
            min_length=1,
            max_length=100,
            process_output=False,
            quiet=False,
            clean_items=True,
            in_place="",
            dry_run=True,
        )

    # File should NOT be modified
    assert test_file.read_text() == content
    # Should see a warning in logs
    assert any("[Dry Run] Would make 1 replacement(s)" in record.message for record in caplog.records)
