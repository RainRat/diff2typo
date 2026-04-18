import multitool

def test_count_mode_mapping_audit(tmp_path):
    # Create an input file with some text
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh quick brown fox jumps over teh lazy dog. The teh word is a typo.")

    # Create an output file
    output_file = tmp_path / "output.csv"

    # Run count mode with an extra mapping
    multitool.count_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=1000,
        process_output=False,
        ad_hoc=["teh:the"],
        output_format='csv',
        quiet=True
    )

    # Check the output
    content = output_file.read_text()
    # CSV header: typo,correction,count
    # Row: teh,the,3
    assert "teh,the,3" in content

def test_count_mode_mapping_file_audit(tmp_path):
    # Create an input file
    input_file = tmp_path / "input.txt"
    input_file.write_text("misstake misstake mistake error misstake")

    # Create a mapping file
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("misstake,mistake\nmistake,correct")

    output_file = tmp_path / "output.csv"

    multitool.count_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        mapping_file=str(mapping_file),
        output_format='csv',
        quiet=True
    )

    content = output_file.read_text()
    assert "misstake,mistake,3" in content
    assert "mistake,correct,1" in content
    assert "error" not in content

def test_count_mode_no_mapping_behavior(tmp_path):
    # Ensure default behavior is preserved
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple banana apple cherry")

    output_file = tmp_path / "output.csv"

    multitool.count_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=1000,
        process_output=False,
        output_format='csv',
        quiet=True
    )

    content = output_file.read_text()
    assert "apple,2" in content
    assert "banana,1" in content
    assert "cherry,1" in content
