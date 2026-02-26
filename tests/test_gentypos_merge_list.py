import sys
import yaml
import pytest
from pathlib import Path
from types import SimpleNamespace

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

def test_setup_generation_tools_merge_list(tmp_path):
    """
    Test merging logic when existing custom substitution is already a list.
    Targets line 573 in gentypos.py (after recent changes).
    """
    subs_file = tmp_path / "extra_subs.yaml"
    # File contains: a -> i
    subs_file.write_text(yaml.dump({"a": ["i"]}))

    settings = SimpleNamespace(
        custom_substitutions_config={"a": ["e"]}, # Existing is a list
        substitutions_file=str(subs_file),
        enable_custom_substitutions=True,
        enable_adjacent_substitutions=False
    )

    # We expect 'a' to now map to both 'e' and 'i'
    # After _setup_generation_tools, custom_subs will have sets
    adj, custom = gentypos._setup_generation_tools(settings)

    assert "a" in custom
    assert custom["a"] == {"e", "i"}
