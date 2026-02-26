import sys
import types
from pathlib import Path
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import gentypos

def test_extract_config_settings_defaults():
    config = {}
    settings = gentypos._extract_config_settings(config)

    assert settings.output_format == 'arrow'
    assert settings.min_length == 8
    assert settings.max_length is None
    assert settings.repeat_modifications == 1
    assert settings.transposition_distance == 1
    assert settings.include_diagonals is True

def test_extract_config_settings_repeat_modifications_clamping():
    config = {'repeat_modifications': -5}
    settings = gentypos._extract_config_settings(config)
    assert settings.repeat_modifications == 1

    config = {'repeat_modifications': 0}
    settings = gentypos._extract_config_settings(config)
    assert settings.repeat_modifications == 1

    config = {'repeat_modifications': 3}
    settings = gentypos._extract_config_settings(config)
    assert settings.repeat_modifications == 3

def test_extract_config_settings_output_format_fallback(caplog):
    config = {'output_format': 'invalid_format'}
    settings = gentypos._extract_config_settings(config)

    assert settings.output_format == 'arrow'
    assert "Unknown output format 'invalid_format'. Defaulting to 'arrow'." in caplog.text

def test_extract_config_settings_valid_formats():
    for fmt in ['csv', 'table', 'list', 'arrow']:
        config = {'output_format': fmt}
        settings = gentypos._extract_config_settings(config)
        assert settings.output_format == fmt

def test_extract_config_settings_output_header_table_default():
    # If output_format is table and header is missing, it should default
    config = {'output_format': 'table'}
    settings = gentypos._extract_config_settings(config)
    assert settings.output_header == "[default.extend-words]"

def test_extract_config_settings_output_header_custom():
    config = {'output_format': 'table', 'output_header': '# My Header'}
    settings = gentypos._extract_config_settings(config)
    assert settings.output_header == "# My Header"

def test_extract_config_settings_nested_options():
    config = {
        'replacement_options': {
            'include_diagonals': False,
            'enable_adjacent_substitutions': False,
        },
        'transposition_options': {
            'distance': 2
        },
        'word_length': {
            'min_length': 5,
            'max_length': 10
        }
    }
    settings = gentypos._extract_config_settings(config)

    assert settings.include_diagonals is False
    assert settings.enable_adjacent_substitutions is False
    assert settings.enable_custom_substitutions is True  # Default
    assert settings.transposition_distance == 2
    assert settings.min_length == 5
    assert settings.max_length == 10
