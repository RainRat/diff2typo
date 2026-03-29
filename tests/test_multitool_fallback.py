import subprocess

def run_multitool(args, input_data=None):
    cmd = ["python", "multitool.py"] + args
    result = subprocess.run(
        cmd,
        input=input_data,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    return result

def test_map_fallback_stdin(tmp_path):
    mapping_file = tmp_path / "mapping.arrow"
    mapping_file.write_text("teh -> the", encoding='utf-8')

    # Test: echo "teh" | python multitool.py map mapping.arrow
    result = run_multitool(["map", str(mapping_file)], input_data="teh\n")

    assert result.returncode == 0
    assert "the" in result.stdout
    assert "Reading from standard input" in result.stderr

def test_scrub_fallback_stdin(tmp_path):
    mapping_file = tmp_path / "mapping.arrow"
    mapping_file.write_text("teh -> the", encoding='utf-8')

    # Test: echo "teh" | python multitool.py scrub mapping.arrow
    result = run_multitool(["scrub", str(mapping_file)], input_data="teh\n")

    assert result.returncode == 0
    assert "the" in result.stdout
    assert "Reading from standard input" in result.stderr

def test_highlight_fallback_stdin(tmp_path):
    mapping_file = tmp_path / "mapping.arrow"
    mapping_file.write_text("teh -> the", encoding='utf-8')

    # Test: echo "teh" | python multitool.py highlight mapping.arrow
    # Output contains ANSI color codes by default
    result = run_multitool(["highlight", str(mapping_file)], input_data="teh\n")

    assert result.returncode == 0
    assert "teh" in result.stdout
    assert "Reading from standard input" in result.stderr

def test_zip_fallback_stdin(tmp_path):
    file2 = tmp_path / "file2.txt"
    file2.write_text("the", encoding='utf-8')

    # Test: echo "teh" | python multitool.py zip file2.txt
    result = run_multitool(["zip", str(file2), "-f", "csv"], input_data="teh\n")

    assert result.returncode == 0
    assert "teh,the" in result.stdout
    assert "Reading from standard input" in result.stderr

def test_diff_fallback_stdin(tmp_path):
    file2 = tmp_path / "file2.txt"
    file2.write_text("the", encoding='utf-8')

    # Test: echo "teh" | python multitool.py diff file2.txt
    result = run_multitool(["diff", str(file2)], input_data="teh\n")

    assert result.returncode == 0
    assert "- teh" in result.stdout
    assert "+ the" in result.stdout

def test_filterfragments_fallback_stdin(tmp_path):
    file2 = tmp_path / "file2.txt"
    file2.write_text("there", encoding='utf-8')

    # Test: echo "the" | python multitool.py filterfragments file2.txt
    # "the" is a fragment of "there", so it should be filtered out.
    # Non-matches are output. If "the" matches, output is empty.
    result = run_multitool(["filterfragments", str(file2)], input_data="the\n")

    assert result.returncode == 0
    assert "the" not in result.stdout

def test_set_operation_fallback_stdin(tmp_path):
    file2 = tmp_path / "file2.txt"
    file2.write_text("the", encoding='utf-8')

    # Test: echo "teh" | python multitool.py set_operation file2.txt --operation union
    result = run_multitool(["set_operation", str(file2), "--operation", "union"], input_data="teh\n")

    assert result.returncode == 0
    assert "teh" in result.stdout
    assert "the" in result.stdout

def test_fuzzymatch_fallback_stdin(tmp_path):
    file2 = tmp_path / "file2.txt"
    file2.write_text("correction", encoding='utf-8')

    # Test: echo "typo" | python multitool.py fuzzymatch file2.txt
    result = run_multitool(["fuzzymatch", str(file2), "-f", "csv", "--max-dist", "10"], input_data="typo\n")

    assert result.returncode == 0
    assert "typo,correction" in result.stdout

def test_map_explicit_stdin(tmp_path):
    # Test that explicit '-' still works
    mapping_file = tmp_path / "mapping.arrow"
    mapping_file.write_text("teh -> the", encoding='utf-8')

    result = run_multitool(["map", "-", str(mapping_file)], input_data="teh\n")
    assert result.returncode == 0
    assert "the" in result.stdout
