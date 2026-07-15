import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import multitool

def test_get_best_suggestion_tie_breaking_length():
    query = "test"
    candidates = ["tests", "tst"]
    suggestion = multitool._get_best_suggestion("test", candidates, max_dist=1)
    assert suggestion == "tst"

def test_get_best_suggestion_tie_breaking_alphabetical():
    query = "test"
    candidates = ["west", "best", "rest"]
    suggestion = multitool._get_best_suggestion("test", candidates, max_dist=1)
    assert suggestion == "best"

def test_get_best_suggestion_no_match():
    query = "test"
    candidates = ["totally", "different", "words"]
    suggestion = multitool._get_best_suggestion(query, candidates, max_dist=1)
    assert suggestion is None

def test_get_best_suggestion_exact_match():
    query = "test"
    candidates = ["test", "tests", "best"]
    suggestion = multitool._get_best_suggestion(query, candidates, max_dist=1)
    assert suggestion == "test"
