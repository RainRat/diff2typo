import subprocess
import os

def test_words_smart_split():
    input_text = "myVariable some_other_word 123Numbers\n"
    with open("test_input.txt", "w") as f:
        f.write(input_text)

    # Run multitool words mode with --smart, --raw and --min-length 1
    result = subprocess.run(
        ["python3", "multitool.py", "words", "test_input.txt", "--smart", "--raw", "--min-length", "1", "--output", "-"],
        capture_output=True,
        text=True
    )

    output = result.stdout.splitlines()
    print(f"Words output with --smart (raw): {output}")

    # These should be split now
    expected = ["my", "Variable", "some", "other", "word", "123", "Numbers"]
    for word in expected:
        assert word in output, f"Expected {word} to be in {output}"

    # Cleanup
    os.remove("test_input.txt")

def test_words_no_smart_split():
    input_text = "myVariable some_other_word 123Numbers\n"
    with open("test_input.txt", "w") as f:
        f.write(input_text)

    # Run multitool words mode WITHOUT --smart but with --raw and --min-length 1
    result = subprocess.run(
        ["python3", "multitool.py", "words", "test_input.txt", "--raw", "--min-length", "1", "--output", "-"],
        capture_output=True,
        text=True
    )

    output = result.stdout.splitlines()
    print(f"Words output WITHOUT --smart (raw): {output}")

    # These should NOT be split
    assert "myVariable" in output
    assert "some_other_word" in output
    assert "123Numbers" in output

    # Cleanup
    os.remove("test_input.txt")

if __name__ == "__main__":
    try:
        test_words_no_smart_split()
        print("Confirmed: WITHOUT --smart, no splitting occurs.")
        test_words_smart_split()
        print("Confirmed: WITH --smart, splitting occurs correctly.")
        print("All tests in tests/test_smart_split.py passed.")
    except AssertionError as e:
        print(f"Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
