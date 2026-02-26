'''
gentypos.py

Purpose:
    Generate common typing errors (typos) for a list of words. This tool helps you
    predict mistakes people might make when typing specific words, which is
    useful for building robust spell-checkers.

Features:
    - Generates typos based on common typing patterns (skipping letters, swapping neighbors, etc.).
    - Uses keyboard adjacency to predict likely finger-slips on a QWERTY layout.
    - Supports custom substitution rules for specific character patterns (e.g., 'ph' -> 'f').
    - Filters out generated typos that are actually valid words in a dictionary.
    - Can process words directly from the command line or from a large text file.
    - Integrates with results from 'typostats.py' to use your own personal typo history.

Usage:
    python gentypos.py hello world
    python gentypos.py --config gentypos.yaml
'''

import sys
import argparse
import yaml
import logging
import time
import os
import copy
import json
import csv
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, Set, Optional
from tqdm import tqdm  # For progress bars; install via `pip install tqdm`


# ANSI Color Codes
BLUE = "\033[1;34m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Disable colors if not running in a terminal
if not sys.stdout.isatty():
    BLUE = GREEN = YELLOW = RESET = BOLD = ""


class MinimalFormatter(logging.Formatter):
    """A logging formatter that removes prefixes for INFO level messages."""

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            return record.getMessage()
        return f"{record.levelname}: {record.getMessage()}"


DEFAULT_CONFIG: dict[str, Any] = {
    'typo_types': {
        'deletion': True,
        'transposition': True,
        'replacement': True,
        'duplication': True,
    },
    'word_length': {
        'min_length': 8,
        'max_length': None,
    },
    'output_header': None,
}


def _merge_defaults(
    config: MutableMapping[str, Any],
    defaults: Mapping[str, Any],
    path: list[str] | None = None,
) -> None:
    """Recursively merge default configuration values into the provided config."""

    if path is None:
        path = []

    for key, default_value in defaults.items():
        dotted_path = '.'.join(path + [key])
        if isinstance(default_value, dict):
            existing = config.setdefault(key, {})
            if not isinstance(existing, dict):
                logging.error(f"Configuration value for '{dotted_path}' must be a mapping.")
                sys.exit(1)
            _merge_defaults(existing, default_value, path + [key])
        else:
            if key not in config:
                logging.debug(f"Applying default for '{dotted_path}': {default_value}")
            config.setdefault(key, default_value)

def get_adjacent_keys(include_diagonals: bool = True) -> dict[str, set[str]]:
    """
    Returns a dictionary of adjacent keys on a QWERTY keyboard.
    Can include diagonally adjacent keys based on the 'include_diagonals' flag.

    Args:
        include_diagonals (bool): Whether to include diagonally adjacent keys.

    Returns:
        dict: A mapping from each character to its adjacent characters.
    """
    keyboard = [
        'qwertyuiop',
        'asdfghjkl',
        'zxcvbnm',
    ]

    # Map each character to its (row, column) coordinate for quick lookup
    coords: dict[str, tuple[int, int]] = {}
    for r, row in enumerate(keyboard):
        for c, ch in enumerate(row):
            coords[ch] = (r, c)

    adjacent: dict[str, set[str]] = {ch: set() for ch in coords}

    for ch, (r, c) in coords.items():
        # Examine neighbouring positions within a 1-key radius
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue  # Skip the key itself

                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= len(keyboard):
                    continue
                if nc < 0 or nc >= len(keyboard[nr]):
                    continue

                # Exclude diagonal keys if requested
                if not include_diagonals and dr != 0 and dc != 0:
                    continue

                adjacent_char = keyboard[nr][nc]
                adjacent[ch].add(adjacent_char)

    return adjacent


def _load_substitutions_file(path: str) -> dict[str, list[str]]:
    """
    Load substitution rules from a file. Supports JSON, CSV, and YAML.
    JSON and CSV formats match the output of typostats.py.
    """
    subs = defaultdict(list)
    if not os.path.exists(path):
        logging.error(f"Substitutions file '{path}' not found.")
        sys.exit(1)

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # typostats format: {"replacements": [{"correct": "a", "typo": "e", ...}, ...]}
                if isinstance(data, dict) and 'replacements' in data:
                    for item in data['replacements']:
                        if 'correct' in item and 'typo' in item:
                            subs[str(item['correct'])].append(str(item['typo']))
                # Plain mapping: {"a": ["e", "i"], ...}
                elif isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, list):
                            subs[str(k)].extend([str(i) for i in v])
                        else:
                            subs[str(k)].append(str(v))
        elif ext == '.csv':
            with open(path, 'r', encoding='utf-8') as f:
                # Use a simple check for typostats header instead of complex sniffing
                first_line = f.readline()
                f.seek(0)
                if 'correct_char' in first_line and 'typo_char' in first_line:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('correct_char') and row.get('typo_char'):
                            subs[str(row['correct_char'])].append(str(row['typo_char']))
                else:
                    # Fallback: plain typo,correction CSV.
                    # If it has a header like "typo,correction", skip it.
                    reader = csv.reader(f)
                    first_row = next(reader, None)
                    if first_row:
                        # Check if first row looks like a header
                        header_keywords = {'typo', 'correction', 'before', 'after', 'correct', 'word'}
                        if any(k in first_row[0].lower() or (len(first_row) > 1 and k in first_row[1].lower()) for k in header_keywords):
                            # Skip header, process remaining
                            pass
                        else:
                            # Process first row as data
                            if len(first_row) >= 2:
                                subs[str(first_row[0])].append(str(first_row[1]))

                    for row in reader:
                        if len(row) >= 2:
                            subs[str(row[0])].append(str(row[1]))
        else:
            # Assume YAML
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, list):
                            subs[str(k)].extend([str(i) for i in v])
                        else:
                            subs[str(k)].append(str(v))
    except Exception as e:
        logging.error(f"Error loading substitutions from '{path}': {e}")
        sys.exit(1)

    return dict(subs)


def load_custom_substitutions(
    custom_subs: Mapping[str, Iterable[str]] | None,
) -> dict[str, set[str]]:
    """
    Load custom substitution rules from a dictionary.

    Args:
        custom_subs (dict): Dictionary containing custom substitution rules.

    Returns:
        dict: A dictionary with characters as keys and sets of substitution characters as values.
    """
    if not custom_subs:
        return {}

    substitutions = {}
    for k, v in custom_subs.items():
        if v is None:
            continue

        # Ensure v is iterable
        if isinstance(v, (str, bytes)):
            v = [v]

        try:
            # Ensure all keys and values are strings and lowercase
            substitutions[str(k).lower()] = set(str(vv).lower() for vv in v if vv is not None)
        except TypeError:
            # If v is not iterable, skip it or handle as single item
            substitutions[str(k).lower()] = {str(v).lower()}

    return substitutions


def generate_typos_by_replacement(
    word: str,
    adjacent_keys: Mapping[str, Set[str]],
    custom_subs: Mapping[str, Set[str]] | None = None,
    use_adjacent: bool = True,
    use_custom: bool = True,
) -> set[str]:
    """
    Generate typos by replacing characters or substrings with adjacent keys or custom substitutions.

    Args:
        word (str): The input word.
        adjacent_keys (dict): Mapping of each character to its adjacent keys.
        custom_subs (dict, optional): Custom substitution rules (supports multi-character keys).
        use_adjacent (bool): Whether to include adjacent key substitutions.
        use_custom (bool): Whether to include custom substitutions.

    Returns:
        set: A set of typo variations generated by replacement.
    """
    typos = set()

    # Character-by-character replacements (handles adjacent keys and single-char custom subs)
    for i, char in enumerate(word):
        replacement_chars = set()

        # Adjacent key replacements
        if use_adjacent and char in adjacent_keys:
            replacement_chars.update(adjacent_keys[char])

        # Custom substitutions (single character)
        if use_custom and custom_subs and char in custom_subs:
            replacement_chars.update(custom_subs[char])

        for replace_char in replacement_chars:
            typo = word[:i] + replace_char + word[i+1:]
            typos.add(typo)

    # Multi-character substring replacements
    if use_custom and custom_subs:
        for sub_key, sub_values in custom_subs.items():
            if len(sub_key) > 1:
                start = 0
                while True:
                    idx = word.find(sub_key, start)
                    if idx == -1:
                        break
                    for replace_val in sub_values:
                        typo = word[:idx] + replace_val + word[idx + len(sub_key):]
                        typos.add(typo)
                    start = idx + 1

    return typos


def generate_typos_by_deletion(word: str) -> set[str]:
    """
    Generate typos by deleting each character in the word.

    Args:
        word (str): The input word.

    Returns:
        set: A set of typo variations generated by deletion.
    """
    variations = set()
    for i in range(len(word)):
        # Don't remove 's' or 'd' from the end of the word
        if i == len(word) - 1 and word[i] in {'s', 'd'}:
            continue
        variation = word[:i] + word[i + 1:]
        variations.add(variation)
    return variations


def generate_typos_by_transposition(word: str, distance: int = 1) -> set[str]:
    """
    Generate typos by swapping characters that are a certain distance apart.

    Args:
        word (str): The input word.
        distance (int): Distance between letters to swap for transposition typos.

    Returns:
        set: A set of typo variations generated by transposition.
    """
    variations = set()
    distance = max(1, distance)
    for i in range(len(word) - distance):
        if word[i] == word[i + distance]:
            continue  # Don't swap identical letters
        # Swap characters that are 'distance' apart
        middle = word[i + 1:i + distance]
        variation = word[:i] + word[i + distance] + middle + word[i] + word[i + distance + 1:]
        variations.add(variation)
    return variations


def generate_typos_by_duplication(word: str) -> set[str]:
    """
    Generate typos by duplicating each character in the word.

    Args:
        word (str): The input word.

    Returns:
        set: A set of typo variations generated by duplication.
    """
    typos = set()

    for i in range(len(word)):
        duplicated_char = word[i]
        typo = word[:i+1] + duplicated_char + word[i+1:]
        typos.add(typo)

    return typos


def generate_all_typos(
    word: str,
    adjacent_keys: Mapping[str, Set[str]],
    custom_subs: Mapping[str, Set[str]],
    typo_types: Mapping[str, bool],
    transposition_distance: int = 1,
    use_adjacent: bool = True,
    use_custom: bool = True,
) -> set[str]:
    """
    Generate all possible typos for a given word using selected typo types.

    Args:
        word (str): The input word.
        adjacent_keys (dict): Mapping of each character to its adjacent keys.
        custom_subs (dict): Custom substitution rules.
        typo_types (dict): Dictionary indicating which typo types to apply.
        transposition_distance (int): Distance between letters to swap for transposition typos.
        use_adjacent (bool): Whether to include adjacent key substitutions.
        use_custom (bool): Whether to include custom substitutions.

    Returns:
        set: A set of all unique typo variations.
    """
    typos = set()

    # Deletion
    if typo_types.get('deletion', False):
        typos.update(generate_typos_by_deletion(word))

    # Transposition
    if typo_types.get('transposition', False):
        typos.update(generate_typos_by_transposition(word, transposition_distance))

    # Replacement
    if typo_types.get('replacement', False):
        typos.update(
            generate_typos_by_replacement(
                word,
                adjacent_keys,
                custom_subs,
                use_adjacent,
                use_custom,
            )
        )

    # Duplication
    if typo_types.get('duplication', False):
        typos.update(generate_typos_by_duplication(word))

    return typos


def load_file(file_path: Optional[str]) -> set[str]:
    """
    Generic function to load words from a file into a set.
    Filters out non-ASCII words and converts them to lowercase.

    Args:
        file_path (Optional[str]): Path to the file containing words.

    Returns:
        set: A set of cleaned words.
    """
    if file_path is None:
        return set()

    try:
        words = set()
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line and all(ord(c) < 128 for c in line):
                    words.add(line.lower())
        logging.debug(f"Loaded {len(words)} unique words from '{file_path}'.")
        return words
    except FileNotFoundError:
        logging.error(f"File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading '{file_path}': {e}")
        sys.exit(1)


def parse_yaml_config(config_path: str) -> dict[str, Any]:
    """
    Parse the YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            logging.debug(f"Parsed YAML configuration from '{config_path}'.")
            return config
    except FileNotFoundError:
        logging.error(f"Configuration file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file '{config_path}': {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading '{config_path}': {e}")
        sys.exit(1)


def validate_config(config: MutableMapping[str, Any], cli_mode: bool = False, config_defaults: Mapping[str, Any] | None = None) -> None:
    """
    Validate the YAML configuration to ensure all required fields are present.

    Args:
        config (dict): Parsed YAML configuration.
        cli_mode (bool): If True, relax some requirements (like dictionary_file).
        config_defaults (dict): Defaults to merge into config.
    """
    required_fields = ['input_file', 'dictionary_file', 'output_file', 'output_format']

    # In CLI mode, these might be overridden or managed by args
    if cli_mode:
        required_fields = []

    for field in required_fields:
        if field not in config:
            logging.error(f"Missing required configuration field: '{field}'")
            sys.exit(1)

    _merge_defaults(config, config_defaults or DEFAULT_CONFIG)


def format_typos(
    typo_to_correct_word: Mapping[str, str], output_format: str
) -> list[str]:
    """
    Formats the typos based on the specified output format.

    Args:
        typo_to_correct_word (dict): Mapping from typo to correct word.
        output_format (str): Desired output format ('arrow', 'csv', 'table', 'list').

    Returns:
        list: Formatted list of typo strings.
    """
    formatted = []
    for typo, correct_word in typo_to_correct_word.items():
        if output_format == 'arrow':
            formatted.append(f"{typo} -> {correct_word}")
        elif output_format == 'csv':
            formatted.append(f"{typo},{correct_word}")
        elif output_format == 'table':
            formatted.append(f'{typo} = "{correct_word}"')
        elif output_format == 'list':
            formatted.append(f"{typo}")
    return formatted


def _extract_config_settings(config: MutableMapping[str, Any], quiet: bool = False) -> SimpleNamespace:
    """Extract validated configuration values into a structured namespace."""

    input_file = config.get('input_file')
    dictionary_file = config.get('dictionary_file')
    output_file = config.get('output_file')
    output_format = config.get('output_format', 'arrow').lower()
    output_header = config.get('output_header')

    valid_formats = {'arrow', 'csv', 'table', 'list'}
    if output_format not in valid_formats:
        logging.warning(
            f"Unknown output format '{output_format}'. Defaulting to 'arrow'."
        )
        output_format = 'arrow'

    if output_header is None and output_format == 'table':
        output_header = "[default.extend-words]"

    replacement_options = config.get(
        'replacement_options',
        {
            'include_diagonals': True,
            'enable_adjacent_substitutions': True,
            'enable_custom_substitutions': True,
        },
    )
    include_diagonals = replacement_options.get('include_diagonals', True)
    enable_adjacent_substitutions = replacement_options.get(
        'enable_adjacent_substitutions', True
    )
    enable_custom_substitutions = replacement_options.get(
        'enable_custom_substitutions', True
    )

    transposition_options = config.get('transposition_options', {'distance': 1})
    transposition_distance = transposition_options.get('distance', 1)

    repeat_modifications = int(config.get('repeat_modifications', 1))
    repeat_modifications = max(1, repeat_modifications)

    word_length = config.get('word_length', {'min_length': 8, 'max_length': None})
    min_length = word_length.get('min_length', 8)
    max_length = word_length.get('max_length', None)

    settings = SimpleNamespace(
        input_file=input_file,
        dictionary_file=dictionary_file,
        output_file=output_file,
        output_format=output_format,
        output_header=output_header,
        typo_types=config.get('typo_types', DEFAULT_CONFIG['typo_types']),
        include_diagonals=include_diagonals,
        enable_adjacent_substitutions=enable_adjacent_substitutions,
        enable_custom_substitutions=enable_custom_substitutions,
        transposition_distance=transposition_distance,
        repeat_modifications=repeat_modifications,
        min_length=min_length,
        max_length=max_length,
        custom_substitutions_config=config.get('custom_substitutions', {}),
        substitutions_file=config.get('substitutions_file'),
        quiet=quiet,
    )

    return settings


def _setup_generation_tools(
    settings: SimpleNamespace,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Prepare substitution helpers based on configuration settings."""

    logging.info("Loading custom substitutions...")

    # Start with substitutions from config
    custom_subs_raw = copy.deepcopy(getattr(settings, 'custom_substitutions_config', {}))

    # Merge substitutions from file if provided
    substitutions_file = getattr(settings, 'substitutions_file', None)
    if substitutions_file:
        file_subs = _load_substitutions_file(substitutions_file)
        for k, v in file_subs.items():
            if k in custom_subs_raw:
                existing = custom_subs_raw[k]
                if existing is None:
                    custom_subs_raw[k] = v
                elif isinstance(existing, list):
                    existing.extend(v)
                else:
                    custom_subs_raw[k] = [existing] + v
            else:
                custom_subs_raw[k] = v

    enable_custom_substitutions = getattr(settings, 'enable_custom_substitutions', True)
    if not enable_custom_substitutions:
        logging.info("Custom substitutions disabled.")
        custom_subs = {}
    else:
        custom_subs = load_custom_substitutions(custom_subs_raw)
        if custom_subs:
            total_custom_entries = len(custom_subs)
            total_custom_replacements = sum(len(v) for v in custom_subs.values())
            logging.info(
                "Loaded %d custom substitution entries with a total of %d replacements.",
                total_custom_entries,
                total_custom_replacements,
            )
        else:
            logging.info("No custom substitutions loaded.")

    enable_adjacent_substitutions = getattr(settings, 'enable_adjacent_substitutions', True)
    if enable_adjacent_substitutions:
        logging.info("Generating adjacent keys mapping...")
        adjacent_keys = get_adjacent_keys(getattr(settings, 'include_diagonals', True))
        total_adjacent_mappings = len(adjacent_keys)

        adjacent_substitutions = set()
        for char, adj_set in adjacent_keys.items():
            for adj_char in adj_set:
                adjacent_substitutions.add((char, adj_char))
        total_adjacent_substitutions = len(adjacent_substitutions)
        logging.info(
            "Generated adjacent keys for %d characters with %d unique adjacent substitutions (non-reflexive).",
            total_adjacent_mappings,
            total_adjacent_substitutions,
        )
    else:
        adjacent_keys = {}
        logging.info("Adjacent substitutions disabled.")

    return adjacent_keys, custom_subs


def _run_typo_generation(
    word_list: Sequence[str],
    all_words: set[str],
    settings: SimpleNamespace,
    adjacent_keys: Mapping[str, Set[str]],
    custom_subs: Mapping[str, Set[str]],
    quiet: bool = False,
) -> dict[str, str]:
    """Generate, filter, and sort typos based on the provided settings."""

    logging.info("Generating synthetic typos...")
    typo_to_correct_word = defaultdict(list)

    for word in tqdm(word_list, desc="Processing words", disable=quiet):
        word_len = len(word)
        if word_len < settings.min_length:
            continue
        if settings.max_length and word_len > settings.max_length:
            continue

        typos_current = {word}
        accumulated_typos = set()
        for _ in range(settings.repeat_modifications):
            new_typos = set()
            for base_word in typos_current:
                new_typos.update(
                    generate_all_typos(
                        base_word,
                        adjacent_keys,
                        custom_subs,
                        settings.typo_types,
                        settings.transposition_distance,
                        settings.enable_adjacent_substitutions,
                        settings.enable_custom_substitutions,
                    )
                )
            accumulated_typos.update(new_typos)
            typos_current = new_typos
        for typo in accumulated_typos:
            typo_to_correct_word[typo].append(word)

    total_typos_generated = len(typo_to_correct_word)
    logging.info(
        "Generated %d unique synthetic typos before filtering.", total_typos_generated
    )

    filtered_typo_to_correct_word = {}
    filtered_typos_count = 0

    if all_words:
        logging.info("Filtering typos against the large dictionary...")
        filter_start_time = time.perf_counter()

        for typo, correct_words in typo_to_correct_word.items():
            if typo in all_words:
                filtered_typos_count += 1
                logging.debug(
                    "Filtered out typo '%s' as it exists in the large dictionary.", typo
                )
                continue
            filtered_typo_to_correct_word[typo] = ', '.join(correct_words)

        final_typo_count = len(filtered_typo_to_correct_word)
        filter_duration = time.perf_counter() - filter_start_time
        logging.info(
            "After filtering, %d typos remain (filtered out %d typos) in %.2f seconds.",
            final_typo_count,
            filtered_typos_count,
            filter_duration,
        )
    else:
        # No dictionary filtering
        logging.info("Skipping dictionary filtering (dictionary empty or disabled).")
        for typo, correct_words in typo_to_correct_word.items():
            filtered_typo_to_correct_word[typo] = ', '.join(correct_words)

    sorted_typos = sorted(filtered_typo_to_correct_word.items())
    return dict(sorted_typos)


def main() -> None:
    """
    Main function to generate synthetic typos and save them to a file based on YAML configuration.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description=f"{BOLD}Synthetic Typo Generator: Create lists of common typing mistakes.{RESET}",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"""{BLUE}Examples:{RESET}
  {GREEN}python gentypos.py "hello" "world"{RESET}
  {GREEN}python gentypos.py --config my_config.yaml --output typos.txt{RESET}
  {GREEN}python gentypos.py word1 word2 --format csv --no-filter{RESET}
""",
    )

    # Input/Output Options
    io_group = parser.add_argument_group(f"{BLUE}INPUT/OUTPUT OPTIONS{RESET}")
    io_group.add_argument(
        'words',
        nargs='*',
        help="One or more words to process. If you provide words here, the tool ignores the input file in your config.",
    )
    io_group.add_argument(
        '-c', '--config',
        type=str,
        default="gentypos.yaml",
        help="The path to your YAML configuration file.",
    )
    io_group.add_argument(
        '-o', '--output',
        type=str,
        help="Save results to this file. Use '-' to print to the screen.",
    )
    io_group.add_argument(
        '-f', '--format',
        choices=['arrow', 'csv', 'table', 'list'],
        metavar='FMT',
        help="Choose an output format (default: arrow).",
    )
    io_group.add_argument(
        '-s', '--substitutions',
        type=str,
        help="The path to a file with your own typo patterns (JSON, CSV, or YAML).",
    )
    # Legacy flag, suppressed from help
    parser.add_argument(
        '--word',
        nargs='+',
        help=argparse.SUPPRESS,
    )

    # Generation Options
    gen_group = parser.add_argument_group(f"{BLUE}GENERATION OPTIONS{RESET}")
    gen_group.add_argument(
        '-m', '--min-length',
        type=int,
        help="Ignore words shorter than this length.",
    )
    gen_group.add_argument(
        '--max-length',
        type=int,
        help="Ignore words longer than this length.",
    )
    gen_group.add_argument(
        '--no-filter',
        action='store_true',
        help="Do not check typos against the dictionary (this is faster).",
    )
    gen_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Show more detailed log messages.",
    )
    gen_group.add_argument(
        '-q', '--quiet',
        action='store_true',
        help="Hide progress bars and show fewer log messages.",
    )

    args = parser.parse_args()

    # Determine if we are in CLI Mode (adhoc words provided)
    # Support both legacy --word and new positional args
    cli_words = args.words or []
    if args.word:
        cli_words.extend(args.word)

    is_cli_mode = bool(cli_words)

    log_level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    # Use a custom handler and formatter to keep output clean
    handler = logging.StreamHandler()
    handler.setFormatter(MinimalFormatter())
    logging.basicConfig(level=log_level, handlers=[handler])
    if args.verbose:
        logging.debug("Verbose mode enabled.")
    elif args.quiet:
        logging.debug("Quiet mode enabled.")

    # Load configuration
    config: dict[str, Any] = {}

    # Try loading config file
    config_loaded = False
    if os.path.exists(args.config):
        config = parse_yaml_config(args.config)
        config_loaded = True
    elif args.config != "gentypos.yaml":
        # User specified a config file that doesn't exist
        logging.error(f"Configuration file '{args.config}' not found.")
        sys.exit(1)

    # If in CLI mode and no config found, start with empty to rely on defaults
    if is_cli_mode and not config_loaded:
        config = {}

    # Prepare defaults for this run
    # Avoid mutating global DEFAULT_CONFIG to ensure thread/process safety and testability
    run_defaults = copy.deepcopy(DEFAULT_CONFIG)

    # In CLI mode, we might want to adjust default word length if not specified in config
    if is_cli_mode and 'word_length' not in config:
        # Default to a smaller min_length for adhoc queries if user didn't provide a config
        run_defaults['word_length']['min_length'] = 0

    # Apply CLI overrides
    if is_cli_mode:
        config['input_file'] = None # Will use CLI words
        if not args.output and 'output_file' not in config:
            config['output_file'] = '-' # Default to stdout
            config['output_format'] = 'arrow' # Safer default for stdout

    # Universal CLI overrides
    if args.output:
        config['output_file'] = args.output
    if args.format:
        config['output_format'] = args.format
    if args.substitutions:
        config['substitutions_file'] = args.substitutions
    if args.no_filter:
        config['dictionary_file'] = None

    if args.min_length is not None or args.max_length is not None:
        if 'word_length' not in config:
            config['word_length'] = {}
        if args.min_length is not None:
            config['word_length']['min_length'] = args.min_length
        if args.max_length is not None:
            config['word_length']['max_length'] = args.max_length

    validate_config(config, cli_mode=is_cli_mode, config_defaults=run_defaults)

    settings = _extract_config_settings(config, quiet=args.quiet)

    # Load words
    if is_cli_mode:
        word_list = [w.lower() for w in cli_words]
        logging.info(f"Processing {len(word_list)} words from CLI arguments.")
    else:
        logging.info("Loading wordlist (small dictionary)...")
        word_set = load_file(settings.input_file)
        word_list = list(word_set)
        logging.info(
            "Loaded %d words from the small dictionary ('%s').",
            len(word_list),
            settings.input_file,
        )

    # Load dictionary if needed
    all_words: set[str] = set()
    if settings.dictionary_file:
        if is_cli_mode and not os.path.exists(settings.dictionary_file):
            # In CLI mode, if the dictionary file (likely from default config) is missing,
            # just log a warning and skip filtering instead of exiting.
            logging.warning(
                "Dictionary file '%s' not found. Skipping filtering in CLI mode.",
                settings.dictionary_file,
            )
        else:
            logging.info("Loading dictionary (large dictionary)...")
            all_words = load_file(settings.dictionary_file)
            logging.info(
                "Loaded %d words from the large dictionary ('%s').",
                len(all_words),
                settings.dictionary_file,
            )

    adjacent_keys, custom_subs = _setup_generation_tools(settings)

    sorted_typo_dict = _run_typo_generation(
        word_list,
        all_words,
        settings,
        adjacent_keys,
        custom_subs,
        quiet=settings.quiet,
    )

    # Format typos based on the selected output format
    logging.info("Formatting typos in '%s' format...", settings.output_format)
    formatted_typos = format_typos(sorted_typo_dict, settings.output_format)

    # Write to output file
    try:
        output_target = settings.output_file
        if output_target == '-':
            # Write to stdout
            if settings.output_header:
                print(settings.output_header)
            for typo in formatted_typos:
                print(typo)
        else:
            with open(output_target, 'w', encoding='utf-8') as file:
                if settings.output_header:
                    file.write(settings.output_header + "\n")
                for typo in formatted_typos:
                    file.write(f"{typo}\n")
            logging.info(
                "Successfully generated %d synthetic typos and saved to '%s'.",
                len(formatted_typos),
                output_target,
            )
    except Exception as e:
        logging.error("Error writing to '%s': %s", settings.output_file, e)
        sys.exit(1)


if __name__ == "__main__":
    main()
