import subprocess

def run_multitool(args, input_text=None):
    cmd = ["python3", "multitool.py"] + args
    result = subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    return result

def test_verify_basic(tmp_path):
    # Create a mapping file
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("teh -> the\nwrld -> world\nmissing -> present", encoding="utf-8")

    # Create an input file
    input_file = tmp_path / "input.txt"
    input_file.write_text("This is teh world.", encoding="utf-8")

    # Run verify mode
    result = run_multitool(["verify", str(input_file), "--mapping", str(mapping_file)])

    assert result.returncode == 0
    assert "VERIFICATION REPORT" in result.stdout
    assert "Total entries in mapping: 3" in result.stdout
    assert "Entries found in files:   1" in result.stdout # 'teh' found. 'wrld' is not in input.
    assert "Entries missing:          2" in result.stdout
    assert "teh" not in result.stdout # It's found, so not in missing list
    assert "  - wrld" in result.stdout
    assert "  - missing" in result.stdout

def test_verify_smart(tmp_path):
    # Create a mapping file
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("teh -> the", encoding="utf-8")

    # Create an input file with subword
    input_file = tmp_path / "input.txt"
    # Using a string that contains teh but NOT as a standalone word that [a-zA-Z0-9]+ would match perfectly without smart splitting
    input_file.write_text("Check teh_word.", encoding="utf-8")

    # Run verify mode without smart
    # pattern.findall('Check teh_word.') -> ['Check', 'teh', 'word']
    # Wait, pattern.findall(line) will actually find 'teh'.

    input_file.write_text("Check tehWord.", encoding="utf-8")
    # pattern.findall('Check tehWord.') -> ['Check', 'tehWord']
    # Without smart, 'tehWord' != 'teh'

    result = run_multitool(["verify", str(input_file), "--mapping", str(mapping_file)])
    assert "Entries found in files:   0" in result.stdout

    # Run verify mode with smart
    result = run_multitool(["verify", str(input_file), "--mapping", str(mapping_file), "--smart"])
    assert result.returncode == 0
    assert "Entries found in files:   1" in result.stdout

def test_verify_prune(tmp_path):
    # Create a mapping file
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("teh -> the\nwrld -> world", encoding="utf-8")

    # Create an input file
    input_file = tmp_path / "input.txt"
    input_file.write_text("This is teh world.", encoding="utf-8")

    # Run verify mode with prune
    result = run_multitool(["verify", str(input_file), "--mapping", str(mapping_file), "--prune"])

    assert result.returncode == 0
    assert "teh -> the" in result.stdout
    assert "wrld -> world" not in result.stdout

def test_verify_extra(tmp_path):
    # Create an input file
    input_file = tmp_path / "input.txt"
    input_file.write_text("This is teh world.", encoding="utf-8")

    # Run verify mode with extra pairs
    result = run_multitool(["verify", str(input_file), "--add", "teh:the", "wrld:world"])

    assert result.returncode == 0
    assert "Total entries in mapping: 2" in result.stdout
    assert "Entries found in files:   1" in result.stdout
    assert "  - wrld" in result.stdout

def test_verify_all_found(tmp_path):
    # Create a mapping file
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("teh -> the", encoding="utf-8")

    # Create an input file
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh", encoding="utf-8")

    # Run verify mode
    result = run_multitool(["verify", str(input_file), "--mapping", str(mapping_file)])

    assert result.returncode == 0
    assert "All entries verified." in result.stdout
    assert "MISSING ENTRIES:" not in result.stdout
