'''
diff2typo.py

Purpose:
    Process a git diff to extract typo corrections and prepare a data update for the `typos` utility,
    a spell-checking tool that uses a list of known typos. This ensures that the identified typos
    are not missed in future code changes.

Features:
    - Identifies typo corrections from git diffs.
    - Splits compound words based on spaces, underscores, and casing boundaries.
    - Filters out corrections where the "before" word is a valid dictionary word to exclude grammar fixes.
    - Integrates with the `typos` tool to avoid duplicate typo entries.
    - Automatically detects dictionary file format (single-word or typo-correction pairs).
    - Allows customization via command-line options, including output format.
    - Uses the `--mode` option to output new typos, new corrections for existing typos, or both.

Usage:
    python diff2typo.py \
        --input_file=diff.txt \
        --output_file=typos.txt \
        --output_format=list \
        --mode [typos|corrections|both] \
        --typos_tool_path=/path/to/typos \
        --allowed_file=allowed.csv \
        --dictionary_file=/path/to/dictionary.txt \
        --min_length=2

Examples:
    - Only new typos: python diff2typo.py --input_file=diff.txt --output_file=typos.txt --mode typos
    - Only corrections for existing typos: python diff2typo.py --input_file=diff.txt --output_file=typos.txt --mode corrections
    - Both typos and corrections: python diff2typo.py --input_file=diff.txt --output_file=typos.txt --mode both

Output Formats:
    - arrow: typo -> correction
    - csv: typo,correction
    - table: typo = "correction"
    - list: typo
'''

import argparse
import csv
import glob
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Set

from tqdm import tqdm


def filter_to_letters(text: str) -> str:
    """Return text containing only lowercase a-z characters."""
    return re.sub("[^a-z]", "", text.lower())

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
    Splits a word into subwords based on spaces, underscores, and casing boundaries.

    Args:
        word (str): The word to split.

    Returns:
        list: A list of subwords.
    """
    # First, split by underscores and spaces
    parts = re.split(r'[ _]+', word)
    subwords = []
    for part in parts:
        # Split based on casing (camelCase, PascalCase)
        # This regex will split before uppercase letters that follow lowercase letters
        split_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+', part)
        if split_parts:
            subwords.extend(split_parts)
        else:
            subwords.append(part)
    return subwords

def read_words_mapping(file_path: str) -> Dict[str, Set[str]]:
    """
    Reads a CSV file of typo fixes and returns a mapping:
         incorrect_word (lowercase) -> set(corrections)

    Each row should be in the form:
         incorrect_word, correction1, correction2, ...

    We can also accept a list of valid words. They will
        just map to nothing.
    """
    mapping: Dict[str, Set[str]] = {}
    rows = _read_csv_rows(file_path, "Dictionary file", required=True)
    for row in rows:
        if row:
            incorrect = row[0].strip().lower()
            corrections = {col.strip().lower() for col in row[1:] if col.strip()}
            mapping[incorrect] = corrections
    logging.info(f"Loaded mapping for {len(mapping)} words from '{file_path}'.")
    return mapping

def _compare_word_lists(before_words: Sequence[str], after_words: Sequence[str], min_length: int) -> List[str]:
    """Return typo pairs discovered when comparing two word sequences."""

    if len(before_words) != len(after_words):
        return []

    typos: List[str] = []
    for index, (before_word, after_word) in enumerate(zip(before_words, after_words)):
        if before_word == after_word:
            continue

        if index > 0 and before_words[index - 1] != after_words[index - 1]:
            continue
        if index < len(before_words) - 1 and before_words[index + 1] != after_words[index + 1]:
            continue

        before_clean = filter_to_letters(before_word)
        after_clean = filter_to_letters(after_word)
        if (
            len(before_clean) >= min_length
            and len(after_clean) >= min_length
            and before_clean
            and after_clean
        ):
            typos.append(f"{before_clean} -> {after_clean}")
    return typos


def process_diff_block(removals: List[str], additions: List[str], min_length: int) -> List[str]:
    """Return typos generated from matching removal/addition blocks."""

    if not removals or not additions:
        return []

    before_text = " ".join(removals)
    after_text = " ".join(additions)
    before_words = split_into_subwords(before_text)
    after_words = split_into_subwords(after_text)
    return _compare_word_lists(before_words, after_words, min_length)


def find_typos(diff_text: str, min_length: int = 2) -> List[str]:
    """
    Parses the diff text to identify typo corrections.

    Args:
        diff_text (str): The git diff text.
        min_length (int): Minimum length of differing substrings to consider as typos.

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
            typos.extend(process_diff_block(removals, additions, min_length))
            removals = []
            additions = []

    typos.extend(process_diff_block(removals, additions, min_length))

    return typos

def sort_and_deduplicate(input_list: Iterable[str]) -> List[str]:
    """
    Removes duplicates from the input list and sorts them.

    Args:
        input_list (list): List of strings.

    Returns:
        list: Sorted and deduplicated list of strings.
    """
    return sorted(set(input_list))

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
            before = filter_to_letters(before)
            after = filter_to_letters(after)
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
    """Return stdin contents, supporting both binary and text streams."""

    stream = getattr(sys.stdin, "buffer", sys.stdin)
    data = stream.read()
    if isinstance(data, str):
        logging.info("Successfully read input diff from stdin.")
        return data
    return _decode_with_fallback(data, "input diff from stdin")


def _read_diff_file(file_path: str) -> str:
    """Return diff text from ``file_path`` with encoding fallback."""

    try:
        with open(file_path, "rb") as file_handle:
            data = file_handle.read()
        return _decode_with_fallback(data, f"input diff file '{file_path}'")
    except FileNotFoundError:
        logging.error(f"Input file '{file_path}' not found. Exiting.")
        sys.exit(1)


def _read_diff_sources(input_files: Optional[Sequence[str]]) -> str:
    """Return concatenated diff text from stdin or the provided file patterns."""

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
    Filters out typos that are already known by the 'typos' tool.

    Args:
        candidates (list): A list of typo candidates in "before -> after" format.
        typos_tool_path (str): The path to the 'typos' tool executable.

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
            already_known = [s for s in re.findall(r'`([^`]+)`', result.stdout) if len(s) > 1]
            filtered = [
                line for line in candidates
                if line.split(' -> ')[0].lower() not in [word.lower() for word in already_known]
            ]
            logging.info(f"Filtered out {len(candidates) - len(filtered)} already-known typo(s).")
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


def process_new_typos(candidates, args, valid_words, allowed_words):
    """
    Process candidate typos (list of "before -> after") to produce
    new typos—that is, typo corrections not already registered by the typos tool.
    Applies additional filtering using allowed words and a dictionary of valid
    words. The dictionary may be a simple word list (one word per line) or a
    words.csv file, where only the correction columns are treated as valid
    words. Returns the formatted list of new typos.
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
        filter_set=valid_words,
        desc="Filtering dictionary words",
        quiet=args.quiet,
    )

    # Deduplicate and sort.
    filtered_candidates = sort_and_deduplicate(filtered_candidates)
    # Format the output according to the requested output format.
    formatted = format_typos(filtered_candidates, args.output_format)
    return formatted


def process_new_corrections(candidates, words_mapping, quiet=False):
    """
    Process candidate typos to produce new corrections for known typos.
    It loads a words mapping file (ie. words.csv) and for each candidate correction,
    if the "before" word is already known but the "after" is not among its registered fixes,
    then it is output.
    Returns a sorted, deduplicated list of new corrections in "before -> after" form.

    Args:
        candidates (list): Candidate "before -> after" strings.
        words_mapping (dict): Mapping of known typos to their corrections.
        quiet (bool): When True, suppress progress display.
    """

    new_corrections = []

    if not words_mapping:
        logging.info("Dictionary mapping is empty; skipping new corrections search.")
        return new_corrections

    sample_key = next(iter(words_mapping))
    if not words_mapping[sample_key]:
        logging.info(
            "Dictionary contains only standalone words; cannot compute new corrections."
        )
        return new_corrections

    progress = None
    iterator = candidates
    if not quiet:
        progress = tqdm(candidates, desc="Checking corrections", unit="candidate", leave=False)
        iterator = progress

    for candidate in iterator:
        if '->' in candidate:
            before, after = [s.strip().lower() for s in candidate.split('->')]
            # Only consider cases where the "before" word is already known in the mapping.
            if before in words_mapping:
                if after not in words_mapping[before]:
                    new_corrections.append(f"{before} -> {after}")
    if progress:
        progress.close()
    new_corrections = sort_and_deduplicate(new_corrections)
    return new_corrections

def main():

    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Process a git diff to identify typos for the `typos` utility.")
    parser.add_argument(
        'input_files_pos',
        nargs='*',
        help="One or more input git diff files or glob patterns. Use '-' to read from stdin.",
    )
    parser.add_argument(
        '--input_file',
        '-i',
        dest='input_files_flag',
        nargs='+',
        type=str,
        default=None,
        help=(
            "One or more input git diff files or glob patterns (legacy flag). "
            "Use '-' to read from stdin. If omitted, stdin is read by default."
        ),
    )
    parser.add_argument('--output_file', type=str, default='output.txt', help='Path to the output typos file.')
    parser.add_argument('--output_format', type=str, choices=['arrow', 'csv', 'table', 'list'], default='arrow',
                        help='Format of the output typos. Choices are: arrow (typo -> correction), csv (typo,correction), table (typo = "correction"), list (typo). Default is arrow.')
    parser.add_argument('--typos_tool_path', type=str, default='typos', help='Path to the typos tool executable.')
    parser.add_argument('--allowed_file', type=str, default='allowed.csv', help='CSV file with allowed words to exclude from typos.')
    parser.add_argument('--min_length', type=int, default=2, help='Minimum length of differing substrings to consider as typos.')
    parser.add_argument('--dictionary_file', type=str, default='words.csv', help='Path to the dictionary file for filtering valid words.')
    parser.add_argument('--mode', type=str, choices=['typos', 'corrections', 'both'],
                        default='typos', help='Which mode to run: "typos", "corrections", or "both".')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress bars and other non-essential output.')
    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting typo extraction process...")

    # Combine positional and flag inputs
    pos_inputs = getattr(args, 'input_files_pos', []) or []
    flag_inputs = getattr(args, 'input_files_flag', []) or []
    input_files = pos_inputs + flag_inputs

    diff_text = _read_diff_sources(input_files)

    # Load the dictionary (words mapping) once.
    dictionary_mapping = read_words_mapping(args.dictionary_file)

    try:
        allowed_words = read_allowed_words(args.allowed_file)
    except Exception as exc:
        logging.error(
            "Failed to read allowed words file '%s': %s", args.allowed_file, exc
        )
        sys.exit(1)
    # Build a set of valid words. For simple word lists, each entry is treated as
    # valid. For words.csv files, only the corrections (columns after the first)
    # are considered valid dictionary words.
    valid_words = set()
    for typo, fixes in dictionary_mapping.items():
        if fixes:
            valid_words.update(fixes)
        else:
            valid_words.add(typo)

    # Extract candidate typo corrections from the diff.
    logging.info("Identifying potential typo corrections from the diff...")
    candidates = find_typos(diff_text, min_length=args.min_length)
    candidates = sort_and_deduplicate(candidates)
    logging.info(f"Identified {len(candidates)} candidate typo correction(s).")

    # Prepare lists to hold results.
    new_typos_result = []
    new_corrections_result = []

    # Process new typos if requested.
    if args.mode in ['typos', 'both']:
        logging.info("Processing new typos (filtering out known typos)...")
        new_typos_result = process_new_typos(candidates, args, valid_words, allowed_words)
        logging.info(f"Found {len(new_typos_result)} new typo(s).")

    # Process new corrections if requested.
    if args.mode in ['corrections', 'both']:
        logging.info("Processing new corrections to existing typos...")
        new_corrections_raw = process_new_corrections(candidates, dictionary_mapping, quiet=args.quiet)
        new_corrections_result = format_typos(new_corrections_raw, args.output_format)
        logging.info(f"Found {len(new_corrections_result)} new correction(s).")

    # Combine results if needed.
    final_output = []
    if args.mode == 'both':
        if new_typos_result:
            final_output.append("=== New Typos ===")
            final_output.extend(new_typos_result)
            final_output.append("")  # Blank line for separation.
        if new_corrections_result:
            final_output.append("=== New Corrections ===")
            final_output.extend(new_corrections_result)
    elif args.mode == 'typos':
        final_output = new_typos_result
    elif args.mode == 'corrections':
        final_output = new_corrections_result

    # Write the final output to the specified file.
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for line in final_output:
                f.write(f"{line}\n")
        logging.info(
            f"Successfully wrote {len(final_output)} line(s) to '{args.output_file}'."
        )
    except Exception as e:
        logging.error(f"Error writing to output file '{args.output_file}': {e}")
        sys.exit(1)

    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
