import subprocess
import os

def test_sentences_mode_basic():
    """Verify that sentences mode extracts simple sentences correctly."""
    input_content = "This is a sentence. And another one! What about this? Yes."
    input_file = "test_sentences.txt"
    with open(input_file, "w") as f:
        f.write(input_content)

    try:
        result = subprocess.run(
            ["python3", "multitool.py", "sentences", input_file, "--raw"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        output = result.stdout.strip().split("\n")
        assert "This is a sentence." in output
        assert "And another one!" in output
        assert "What about this?" in output
        # 'Yes.' might be filtered if it's too short, but we didn't set min_length
        # default for sentences is 10
        assert "Yes." not in output # length 4 < 10
    finally:
        if os.path.exists(input_file):
            os.remove(input_file)

def test_sentences_mode_multiline():
    """Verify that sentences mode handles multi-line sentences."""
    input_content = "This is a\nmulti-line sentence. It should\nbe joined."
    input_file = "test_multiline.txt"
    with open(input_file, "w") as f:
        f.write(input_content)

    try:
        result = subprocess.run(
            ["python3", "multitool.py", "sentences", input_file, "--raw"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        output = result.stdout.strip().split("\n")
        assert "This is a multi-line sentence." in output
        assert "It should be joined." in output
    finally:
        if os.path.exists(input_file):
            os.remove(input_file)

def test_count_sentences():
    """Verify that count mode works with sentences."""
    input_content = "Repeat this sentence. Repeat this sentence. Unique sentence here."
    input_file = "test_count_sentences.txt"
    with open(input_file, "w") as f:
        f.write(input_content)

    try:
        result = subprocess.run(
            ["python3", "multitool.py", "count", input_file, "--sentences", "--raw", "--output-format", "json"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        import json
        data = json.loads(result.stdout)
        # Find "Repeat this sentence."
        repeat_found = False
        unique_found = False
        for item in data:
            if item["item"] == "Repeat this sentence.":
                assert item["count"] == 2
                repeat_found = True
            if item["item"] == "Unique sentence here.":
                assert item["count"] == 1
                unique_found = True
        assert repeat_found
        assert unique_found
    finally:
        if os.path.exists(input_file):
            os.remove(input_file)

def test_sentences_abbreviations():
    """Verify heuristic doesn't over-split on common abbreviations (if it can)."""
    # Our heuristic: (?<=[.!?])\s+(?=[^a-z]|$)
    # It splits if there's a space followed by a capital letter.
    # "e.g. This" will split. "e.g. this" will not.
    input_content = "Visit the U.S.A. today. It is great."
    input_file = "test_abbr.txt"
    with open(input_file, "w") as f:
        f.write(input_content)

    try:
        result = subprocess.run(
            ["python3", "multitool.py", "sentences", input_file, "--raw"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        output = result.stdout.strip().split("\n")
        # "Visit the U.S.A. today." might be split into "Visit the U.S.A." and "today." if "today" was capitalized.
        # But here it is "today." (lowercase), so it should stay together.
        assert "Visit the U.S.A. today." in output
        assert "It is great." in output
    finally:
        if os.path.exists(input_file):
            os.remove(input_file)
