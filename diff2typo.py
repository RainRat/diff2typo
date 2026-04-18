'''
diff2typo.py

Purpose:
    Find typo corrections in a Git diff and prepare an update for the `typos` tool.
    This helps ensure that typos you find are caught in future changes.

Features:
    - Finds typo corrections in Git diffs.
    - Splits compound words based on spaces, underscores, and capital letters.
    - Skips corrections where the "before" word is in the large dictionary.
    - Works with the `typos` tool to avoid duplicate entries.
    - Automatically detects the word list file format.
    - Allows customization through command-line options.
    - Uses the `--mode` option to find typos, corrections for existing typos, or cases where a correct word was changed into a typo.

Usage:
    python diff2typo.py diff.txt --output=typos.txt --format=list

Examples:
    - Find typos: python diff2typo.py diff.txt --output=typos.txt --mode typos
    - Corrections for existing typos: python diff2typo.py diff.txt --output=typos.txt --mode corrections
    - Both typos and corrections: python diff2typo.py diff.txt --output=typos.txt --mode both
    - Find correct words changed into typos: python diff2typo.py diff.txt --mode audit

Output Formats:
    - arrow: typo -> correction
    - csv: typo,correction
    - table: typo = "correction"
    - list: typo
'''

import argparse
import contextlib
import csv
import glob
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Set, TextIO

from tqdm import tqdm


# ANSI Color Codes
BLUE = "\033[1;34m"
GREEN = "\033[1;32m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Disable colors if not running in a terminal or if NO_COLOR is set
if not sys.stdout.isatty() or os.environ.get('NO_COLOR'):
    BLUE = GREEN = RED = YELLOW = RESET = BOLD = ""


class MinimalFormatter(logging.Formatter):
    """A logging formatter that removes prefixes for INFO level messages."""

    LEVEL_COLORS = {
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            return record.getMessage()

        levelname = record.levelname
        # Colorize the level name if stderr is a terminal and color is available
        if sys.stderr.isatty() and levelname:
            color = self.LEVEL_COLORS.get(record.levelno)
            if color:
                levelname = f"{color}{levelname}{RESET}"

        return f"{levelname}: {record.getMessage()}"


def filter_to_letters(text: str) -> str:
    """Return text containing only lowercase a-z characters."""
    return re.sub("[^a-z]", "", text.lower())


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the number of character changes needed to turn one word into another."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if not s2:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _read_csv_rows(file_path, description, required=False):
    """Return CSV rows from ``file_path`` with shared error handling."""

    try:
        with open(file_path, "r", encoding="utf-8") as file_handle:
            return list(csv.reader(file_handle))
    except FileNotFoundError:
        message = f"{description} '{file_path}' not found."
        if required:
            logging.error(message)
            sys.exit(1)
        logging.warning(message + " Skipping.")
        return []
    except Exception as exc:  # pragma: no cover - extremely unlikely
        logging.error(f"Error reading {description.lower()} '{file_path}': {exc}")
        if required:
            sys.exit(1)
        return []


def read_allowed_words(allowed_file: str) -> Set[str]:
    """
    Reads allowed words from a CSV file and returns a set of lowercase words.
    These are words that have been explicitly rejected from being considered typos.

    Args:
        allowed_file (str): Path to the allowed words CSV file.

    Returns:
        set: A set of allowed words in lowercase.
    """
    rows = _read_csv_rows(allowed_file, "Allowed words file", required=False)
    allowed_words = {row[0].strip().lower() for row in rows if row}
    if rows:
        logging.info(f"Loaded {len(allowed_words)} allowed words from '{allowed_file}'.")
    return allowed_words

def split_into_subwords(word: str) -> List[str]:
    """
    Splits a word into subwords based on spaces, underscores, hyphens, and casing boundaries.

    Args:
        word (str): The word to split.

    Returns:
        list: A list of subwords.
    """
    pattern = r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+'
    subwords = []
    for part in re.split(r'[ _-]+', word):
        subwords.extend(re.findall(pattern, part) or [part])
    return subwords

def read_words_mapping(file_path: str, required: bool = True) -> Dict[str, Set[str]]:
    """
    Reads a CSV file of typo fixes and returns a list:
         incorrect_word -> corrections

    Each row should be in the form:
         incorrect_word, correction1, correction2, ...

    We can also accept a list of words for the large dictionary. They will
        not have any corrections.
    """
    mapping: Dict[str, Set[str]] = {}
    rows = _read_csv_rows(file_path, "Large dictionary file", required=required)
    for row in rows:
        if row:
            incorrect = row[0].strip().lower()
            corrections = {col.strip().lower() for col in row[1:] if col.strip()}
            mapping[incorrect] = corrections
    if rows:
        logging.info(f"Loaded mapping for {len(mapping)} words from '{file_path}'.")
    return mapping

def _compare_word_lists(
    before_words: Sequence[str],
    after_words: Sequence[str],
    min_length: int,
    max_dist: Optional[int] = None,
) -> List[str]:
    """Return typo pairs discovered when comparing two word sequences."""
    import difflib

    # Use sequence alignment to find corresponding changes in words.
    # This allows correctly finding typo corrections even when words
    # are added or removed within the same diff block.
    matcher = difflib.SequenceMatcher(None, before_words, after_words)
    typos: List[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Extraction of words from the identified replaced blocks.
            # We match them 1-to-1 if the block sizes are identical.
            # Otherwise, we attempt to find the most likely pairing.
            removals = before_words[i1:i2]
            additions = after_words[j1:j2]

            # If the number of removed and added words in the block matches,
            # we can process them as individual substitutions.
            if len(removals) == len(additions):
                for k, (before_word, after_word) in enumerate(zip(removals, additions)):
                    if before_word == after_word:
                        continue

                    before_clean = filter_to_letters(before_word)
                    after_clean = filter_to_letters(after_word)

                    if before_clean == after_clean:
                        continue

                    if len(before_clean) >= min_length and len(after_clean) >= min_length:
                        if max_dist is None or levenshtein_distance(before_clean, after_clean) <= max_dist:
                            typos.append(f"{before_clean} -> {after_clean}")
            else:
                # If block sizes differ (for example, "teh house" -> "the big house"),
                # we perform a local similar word matching to find the best candidate pair.
                for b_word in removals:
                    b_clean = filter_to_letters(b_word)
                    if len(b_clean) < min_length:
                        continue

                    best_match = None
                    best_dist = float('inf')

                    for a_word in additions:
                        a_clean = filter_to_letters(a_word)
                        if len(a_clean) < min_length or a_clean == b_clean:
                            continue

                        dist = levenshtein_distance(b_clean, a_clean)
                        # We only consider it a typo if the distance is low relative to the word length
                        # and fits within the global max_dist constraint.
                        if dist < best_dist and dist <= (max_dist if max_dist is not None else 2):
                            best_match = a_clean
                            best_dist = dist

                    if best_match:
                        typos.append(f"{b_clean} -> {best_match}")

    return typos


def process_diff_block(
    removals: List[str], additions: List[str], min_length: int, max_dist: Optional[int] = None
) -> List[str]:
    """Return typos generated from matching removal/addition blocks."""

    if not removals or not additions:
        return []

    before_text = " ".join(removals)
    after_text = " ".join(additions)
    before_words = split_into_subwords(before_text)
    after_words = split_into_subwords(after_text)
    return _compare_word_lists(before_words, after_words, min_length, max_dist)


def find_typos(diff_text: str, min_length: int = 2, max_dist: Optional[int] = None) -> List[str]:
    """
    Parses the diff text to find typo corrections.

    Args:
        diff_text (str): The Git diff text.
        min_length (int): Minimum length of differing substrings to consider as typos.
        max_dist (int, optional): Maximum Levenshtein distance for typos.

    Returns:
        list: A list of typo candidates in the format "before -> after".
    """
    typos: List[str] = []
    lines = diff_text.split("\n")
    removals: List[str] = []
    additions: List[str] = []

    for line in lines:
        if line.startswith('---') or line.startswith('+++'):
            continue
        if line.startswith('-'):
            removals.append(line[1:].strip())
        elif line.startswith('+'):
            additions.append(line[1:].strip())
        else:
            typos.extend(process_diff_block(removals, additions, min_length, max_dist))
            removals = []
            additions = []

    typos.extend(process_diff_block(removals, additions, min_length, max_dist))

    return typos

@contextlib.contextmanager
def smart_open_output(filename: str, encoding: str = 'utf-8') -> Iterable[TextIO]:
    """
    Context manager that yields a file object for writing.
    If filename is '-', yields the screen.
    Otherwise, opens the file for writing.
    """
    if filename == '-':
        yield sys.stdout
    else:
        with open(filename, 'w', encoding=encoding) as f:
            yield f


def format_typos(typos: Iterable[str], output_format: str) -> List[str]:
    """
    Formats the list of typos based on the specified output format.

    Args:
        typos (list): List of typo strings in the format "before -> after".
        output_format (str): Desired output format ('arrow', 'csv', 'table', 'list').

    Returns:
        list: Formatted list of typo strings.
    """
    formatted: List[str] = []
    for typo in typos:
        if ' -> ' in typo:
            before, after = typo.split(' -> ')
            if output_format == 'arrow':
                formatted.append(f"{before} -> {after}")
            elif output_format == 'csv':
                formatted.append(f"{before},{after}")
            elif output_format == 'table':
                formatted.append(f'{before} = "{after}"')
            elif output_format == 'list':
                formatted.append(f"{before}")
        else:
            # In case the typo does not follow the expected format
            formatted.append(filter_to_letters(typo))
    return formatted


def _decode_with_fallback(data: bytes, description: str) -> str:
    """Decode ``data`` using UTF-8 with a latin-1 fallback and log the outcome."""

    try:
        text = data.decode("utf-8")
        logging.info(f"Successfully read {description}.")
        return text
    except UnicodeDecodeError:
        text = data.decode("latin-1")
        logging.info(f"Successfully read {description} with 'latin-1' encoding.")
        return text


def _read_stdin_text() -> str:
    """Return standard input contents, supporting both binary and text streams."""

    stream = getattr(sys.stdin, "buffer", sys.stdin)
    data = stream.read()
    if isinstance(data, str):
        logging.info("Successfully read input diff from standard input.")
        return data
    return _decode_with_fallback(data, "input diff from standard input")


def _read_diff_file(file_path: str) -> str:
    """Return diff text from ``file_path`` with encoding fallback."""

    try:
        with open(file_path, "rb") as file_handle:
            data = file_handle.read()
        return _decode_with_fallback(data, f"input diff file '{file_path}'")
    except FileNotFoundError:
        logging.error(f"Input file '{file_path}' not found. Exiting.")
        sys.exit(1)


def _read_git_diff(git_args: Optional[str]) -> str:
    """Fetch diff directly from Git using the provided arguments."""
    command = ["git", "diff"]
    if git_args:
        command.extend(shlex.split(git_args))

    try:
        logging.info(f"Running git command: {' '.join(command)}")
        result = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        logging.error("Git executable not found.")
        sys.exit(1)


def _read_diff_sources(input_files: Optional[Sequence[str]]) -> str:
    """Return concatenated diff text from standard input or the provided file patterns."""

    if not input_files:
        return _read_stdin_text()

    contents: List[str] = []
    for pattern in input_files:
        if pattern == "-":
            contents.append(_read_stdin_text())
            continue

        matches = glob.glob(pattern)
        if not matches:
            logging.error(f"Input file '{pattern}' not found. Exiting.")
            sys.exit(1)

        for match in matches:
            if not os.path.isfile(match):
                logging.error(f"Input file '{match}' not found. Exiting.")
                sys.exit(1)
            contents.append(_read_diff_file(match))

    return "\n".join(contents)


def filter_known_typos(candidates, typos_tool_path):
    """
    Filters out typos that are known by the 'typos' tool.

    Args:
        candidates (list): A list of typo candidates in "before -> after" format.
        typos_tool_path (str): The path to the 'typos' tool.

    Returns:
        list: A filtered list of typo candidates.
    """
    with tempfile.TemporaryDirectory(prefix="typos_") as temp_dir:
        temp_file = os.path.join(temp_dir, "candidates.txt")
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                for typo in candidates:
                    f.write(f"{typo}\n")
        except Exception as e:
            logging.error(f"Error writing to temporary file '{temp_file}': {e}")
            return candidates

        typos_executable = shutil.which(typos_tool_path)
        if not typos_executable and os.path.exists(typos_tool_path):
            typos_executable = typos_tool_path
        if not typos_executable:
            logging.warning(
                f"Typos tool '{typos_tool_path}' not found in PATH. Skipping known typo filtering."
            )
            return candidates

        command = [typos_executable, '--format', 'brief', temp_file]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            known_typos = {s.lower() for s in re.findall(r'`([^`]+)`', result.stdout) if len(s) > 1}
            filtered = [
                line for line in candidates
                if line.split(' -> ')[0].lower() not in known_typos
            ]
            logging.info(f"Filtered out {len(candidates) - len(filtered)} known typo(s).")
            return filtered
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logging.warning(f"Error running typos tool: {e}. Skipping known typo filtering.")
            return candidates

def _filter_candidates_by_set(candidates, filter_set, desc, quiet=False):
    """Return candidate typos whose ``before`` word is not in ``filter_set``."""

    if not filter_set:
        return candidates

    filtered_list = []
    progress = None
    iterator = candidates
    if not quiet:
        progress = tqdm(candidates, desc=desc, unit="typo", leave=False)
        iterator = progress

    for typo in iterator:
        if typo.split(' -> ')[0].lower() not in filter_set:
            filtered_list.append(typo)

    if progress:
        progress.close()

    logging.info(
        f"Excluded {len(candidates) - len(filtered_list)} typo(s) based on {desc.lower()}."
    )
    return filtered_list


def process_typos_mode(candidates, args, large_dictionary, allowed_words):
    """
    Find typos that are not known.
    Uses allowed words and the large dictionary to filter the results.
    The large dictionary can be a simple word list (one word per line) or a
    CSV file where the first word is a typo and the rest are corrections.
    Returns the formatted list of typos.
    """
    candidates = filter_known_typos(candidates, typos_tool_path=args.typos_tool_path)
    candidates = _filter_candidates_by_set(
        candidates,
        filter_set=allowed_words,
        desc="Filtering allowed words",
        quiet=args.quiet,
    )
    filtered_candidates = _filter_candidates_by_set(
        candidates,
        filter_set=large_dictionary,
        desc="Filtering large dictionary words",
        quiet=args.quiet,
    )

    # Deduplicate and sort.
    filtered_candidates = sorted(set(filtered_candidates))
    # Format the output according to the requested output format.
    formatted = format_typos(filtered_candidates, args.output_format)
    return formatted


def process_corrections_mode(candidates, words_mapping, quiet=False):
    """
    Find corrections for typos that are known.
    It reads a word list and for each potential correction,
    if the "before" word is known but the "after" word is not,
    then it is saved.
    Returns a sorted list of corrections in "before -> after" form.

    Args:
        candidates (list): Candidate "before -> after" strings.
        words_mapping (dict): Mapping of known typos to their corrections.
        quiet (bool): When True, suppress progress display.
    """

    corrections = []

    if not words_mapping:
        logging.info("Large dictionary mapping is empty; skipping corrections search.")
        return corrections

    progress = None
    iterator = candidates
    if not quiet:
        progress = tqdm(candidates, desc="Checking corrections", unit="candidate", leave=False)
        iterator = progress

    for candidate in iterator:
        if '->' in candidate:
            before, after = [s.strip().lower() for s in candidate.split('->')]
            # Only consider cases where the "before" word is known in the mapping as a typo.
            if before in words_mapping:
                if after not in words_mapping[before]:
                    corrections.append(f"{before} -> {after}")
    if progress:
        progress.close()
    corrections = sorted(set(corrections))
    return corrections


def process_audit_typos(candidates, args, large_dictionary, allowed_words):
    """
    Find cases where a correct word was changed into a typo.
    Identifies cases where a word that used to be valid
    was changed to a word that is not in the large dictionary.
    """
    audit_candidates = []
    for candidate in candidates:
        if ' -> ' in candidate:
            before, after = [s.strip().lower() for s in candidate.split(' -> ')]
            # Find cases where a valid word was changed to an invalid one
            if before in large_dictionary:
                if after not in large_dictionary and after not in allowed_words:
                    audit_candidates.append(candidate)

    audit_candidates = sorted(set(audit_candidates))
    formatted = format_typos(audit_candidates, args.output_format)
    return formatted


def main():

    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        description=f"{BOLD}Process a Git diff to find typos for the `typos` tool.{RESET}",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"""{BLUE}Examples:{RESET}
  {GREEN}python diff2typo.py diff.txt --output typos.txt --mode typos{RESET}
  {GREEN}git diff | python diff2typo.py -o found.txt -f csv{RESET}
""",
    )

    # Input/Output Options
    io_group = parser.add_argument_group(f"{BLUE}INPUT/OUTPUT OPTIONS{RESET}")
    io_group.add_argument(
        'input_files',
        nargs='*',
        metavar='FILE',
        help="One or more input Git diff files or patterns. Use '-' to read from standard input.",
    )
    io_group.add_argument(
        '--git',
        nargs='?',
        const='',
        help="Fetch diff directly from Git. Optional arguments are passed to 'git diff'.",
    )
    io_group.add_argument(
        '--input',
        '-i',
        dest='input_files_flag',
        nargs='+',
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    # Hidden alias for backward compatibility
    parser.add_argument('--input_file', dest='input_files_flag', nargs='+', type=str, help=argparse.SUPPRESS, default=argparse.SUPPRESS)

    io_group.add_argument(
        '--output',
        '-o',
        dest='output_file',
        type=str,
        default='-',
        help="Path to the output file. Use '-' to print to the screen (default: the screen).",
    )
    # Hidden alias for backward compatibility
    parser.add_argument('--output_file', type=str, help=argparse.SUPPRESS, default=argparse.SUPPRESS)

    io_group.add_argument(
        '--format',
        '-f',
        dest='output_format',
        type=str,
        choices=['arrow', 'csv', 'table', 'list'],
        default='arrow',
        help='Format of the output typos. Choices are: arrow (typo -> correction), csv (typo,correction), table (typo = "correction"), list (typo). Default is arrow.',
    )
    # Hidden alias for backward compatibility
    parser.add_argument('--output_format', type=str, choices=['arrow', 'csv', 'table', 'list'], help=argparse.SUPPRESS, default=argparse.SUPPRESS)

    # Analysis Options
    analysis_group = parser.add_argument_group(f"{BLUE}ANALYSIS OPTIONS{RESET}")
    analysis_group.add_argument(
        '--mode',
        type=str,
        choices=['typos', 'corrections', 'both', 'audit'],
        default='typos',
        help=(
            f"{YELLOW}Analysis mode:{RESET}\n"
            f"  {GREEN}typos{RESET}:       Find typos that are not in your large dictionary (default).\n"
            f"  {GREEN}corrections{RESET}: Find corrections for typos in your large dictionary.\n"
            f"  {GREEN}both{RESET}:        Run both analyses and label the results.\n"
            f"  {GREEN}audit{RESET}:       Find cases where a correct word was changed into a typo."
        ),
    )
    analysis_group.add_argument(
        '--min-length',
        '-m',
        dest='min_length',
        type=int,
        default=2,
        help='Ignore words shorter than this (default: 2).',
    )
    # Hidden alias for backward compatibility
    parser.add_argument('--min_length', type=int, help=argparse.SUPPRESS, default=argparse.SUPPRESS)

    analysis_group.add_argument(
        '--max-dist',
        type=int,
        default=None,
        help='Only include typos with a number of character changes up to this value (default: no limit).',
    )

    analysis_group.add_argument(
        '--dictionary',
        '-d',
        dest='dictionary_file',
        type=str,
        default='words.csv',
        help='The file containing the large dictionary (default: words.csv).',
    )
    # Hidden alias for backward compatibility
    parser.add_argument('--dictionary_file', type=str, help=argparse.SUPPRESS, default=argparse.SUPPRESS)

    analysis_group.add_argument(
        '--allowed',
        dest='allowed_file',
        type=str,
        default='allowed.csv',
        help='The file with allowed words to ignore (default: allowed.csv).',
    )
    # Hidden alias for backward compatibility
    parser.add_argument('--allowed_file', type=str, help=argparse.SUPPRESS, default=argparse.SUPPRESS)

    analysis_group.add_argument(
        '--typos-path',
        dest='typos_tool_path',
        type=str,
        default='typos',
        help='The command or path to the typos tool (default: typos).',
    )
    # Hidden alias for backward compatibility
    parser.add_argument('--typos_tool_path', type=str, help=argparse.SUPPRESS, default=argparse.SUPPRESS)

    analysis_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress bars and other non-essential output.'
    )

    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    # Use a custom handler and formatter to keep output clean
    handler = logging.StreamHandler()
    handler.setFormatter(MinimalFormatter('%(levelname)s: %(message)s'))
    logging.basicConfig(level=log_level, handlers=[handler])

    logging.info("Starting typo search...")

    # Combine positional and flag inputs
    pos_inputs = getattr(args, 'input_files', []) or []
    flag_inputs = getattr(args, 'input_files_flag', []) or []
    input_files = pos_inputs + flag_inputs

    if args.git is not None:
        diff_text = _read_git_diff(args.git)
    else:
        diff_text = _read_diff_sources(input_files)

    # Load the large dictionary (words mapping) once.
    # If the file is missing, we don't exit. Instead we just warn and continue without filtering.
    if args.dictionary_file == 'words.csv' and not os.path.exists(args.dictionary_file):
        logging.warning("Default large dictionary file 'words.csv' not found. Skipping filtering.")
        large_dictionary_mapping = {}
    else:
        # If it's NOT the default words.csv, it will also warn and continue if missing.
        large_dictionary_mapping = read_words_mapping(args.dictionary_file, required=False)

    try:
        allowed_words = read_allowed_words(args.allowed_file)
    except Exception as exc:
        logging.error(
            "Failed to read allowed words file '%s': %s", args.allowed_file, exc
        )
        sys.exit(1)
    # Build a set of words for the large dictionary. For simple word lists, every
    # entry is treated as correct. For words.csv files, only the corrections
    # (columns after the first) are considered correct words.
    large_dictionary = set()
    for typo, fixes in large_dictionary_mapping.items():
        if fixes:
            large_dictionary.update(fixes)
        else:
            large_dictionary.add(typo)

    # Find candidate typo corrections from the diff.
    logging.info("Finding typo corrections from the diff...")
    candidates = find_typos(diff_text, min_length=args.min_length, max_dist=args.max_dist)
    candidates = sorted(set(candidates))
    logging.info(f"Found {len(candidates)} candidate typo correction(s).")

    # Prepare lists to hold results.
    typos_result = []
    corrections_result = []
    audit_result = []

    # Process typos if requested.
    if args.mode in ['typos', 'both']:
        logging.info("Processing typos (filtering out known typos)...")
        typos_result = process_typos_mode(candidates, args, large_dictionary, allowed_words)
        logging.info(f"Found {len(typos_result)} typo(s).")

    # Process corrections if requested.
    if args.mode in ['corrections', 'both']:
        logging.info("Processing corrections to existing typos...")
        corrections_raw = process_corrections_mode(candidates, large_dictionary_mapping, quiet=args.quiet)
        corrections_result = format_typos(corrections_raw, args.output_format)
        logging.info(f"Found {len(corrections_result)} correction(s).")

    # Check for correct words changed into typos if requested.
    if args.mode == 'audit':
        logging.info("Checking for cases where correct words were changed into typos...")
        audit_result = process_audit_typos(candidates, args, large_dictionary, allowed_words)
        logging.info(f"Found {len(audit_result)} case(s) where a correct word was changed to a typo.")

    # Combine results if needed.
    final_output = []
    if args.mode == 'both':
        if typos_result:
            final_output.append("=== Typos ===")
            final_output.extend(typos_result)
            final_output.append("")  # Blank line for separation.
        if corrections_result:
            final_output.append("=== Corrections ===")
            final_output.extend(corrections_result)
    elif args.mode == 'typos':
        final_output = typos_result
    elif args.mode == 'corrections':
        final_output = corrections_result
    elif args.mode == 'audit':
        final_output = audit_result

    # Write the final output to the specified file.
    try:
        with smart_open_output(args.output_file, encoding='utf-8') as f:
            for line in final_output:
                f.write(f"{line}\n")
        logging.info(
            f"Wrote {len(final_output)} line(s) to '{args.output_file}'."
        )
    except Exception as e:
        logging.error(f"Error writing to output file '{args.output_file}': {e}")
        sys.exit(1)

    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
