import subprocess
import os

def run_multitool(args):
    result = subprocess.run(
        ['python3', 'multitool.py'] + args,
        capture_output=True,
        text=True
    )
    return result

def test_standardize_filters():
    test_file = 'test_standardize.txt'
    with open(test_file, 'w') as f:
        # 'above' is frequent, 'abovf' is rare and a nearby key error (f is next to e)
        # 'the' is frequent, 'teh' is rare and a transposition
        content = (
            "above above above above above above above above above above\n"
            "abovf\n"
            "the the the the the the the the the the\n"
            "teh\n"
        )
        f.write(content)

    print("--- Test 1: No filters, fuzzy 2 ---")
    res = run_multitool(['standardize', test_file, '--fuzzy', '2'])
    print(res.stdout)
    assert 'above' in res.stdout
    assert 'abovf' not in res.stdout
    assert 'the' in res.stdout
    assert 'teh' not in res.stdout
    print("Test 1 passed (both fixed)")

    print("--- Test 2: Keyboard filter only ---")
    # Should fix abovf -> above, but NOT teh -> the
    res = run_multitool(['standardize', test_file, '--fuzzy', '2', '--keyboard'])
    print(res.stdout)
    assert 'above' in res.stdout
    assert 'abovf' not in res.stdout # Fixed
    assert 'teh' in res.stdout       # Not fixed
    print("Test 2 passed (only keyboard fixed)")

    print("--- Test 3: Transposition filter only ---")
    # Should fix teh -> the, but NOT abovf -> above
    res = run_multitool(['standardize', test_file, '--fuzzy', '2', '--transposition'])
    print(res.stdout)
    assert 'abovf' in res.stdout     # Not fixed
    assert 'the' in res.stdout
    assert 'teh' not in res.stdout   # Fixed
    print("Test 3 passed (only transposition fixed)")

    print("--- Test 4: Both filters ---")
    res = run_multitool(['standardize', test_file, '--fuzzy', '2', '--keyboard', '--transposition'])
    print(res.stdout)
    assert 'abovf' not in res.stdout
    assert 'teh' not in res.stdout
    print("Test 4 passed (both fixed with both filters)")

    print("--- Test 5: Automatic distance boosting ---")
    # Even with fuzzy 0, --transposition should boost to 2 and fix teh
    res = run_multitool(['standardize', test_file, '--transposition'])
    print(res.stdout)
    assert 'teh' not in res.stdout
    assert 'abovf' in res.stdout # fuzzy 0 -> boosted to 2 for trans, but keyboard still needs its flag or fuzzy
    print("Test 5 passed (boosted distance worked)")

    os.remove(test_file)

if __name__ == "__main__":
    try:
        test_standardize_filters()
        print("\nALL TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
