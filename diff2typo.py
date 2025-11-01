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
    - Output new typos, new corrections for existing typos, or both.

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

Output Formats:
    - arrow: typo -> correction
    - csv: typo,correction
    - table: typo = "correction"
    - list: typo
'''

import re
import subprocess
import argparse
import csv
import os
import sys
import logging
import tempfile

from tqdm import tqdm


def filter_to_letters(text):
    """Return text containing only lowercase a-z characters."""
    return re.sub("[^a-z]", "", text.lower())

def extract_backticks(input_text):
    """
    Extracts all backtick-enclosed strings from the input text.

    Args:
        input_text (str): The text to extract backtick-enclosed strings from.

    Returns:
        list: A list of extracted strings without the backticks.
    """
    return [s for s in re.findall(r'`([^`]+)`', input_text) if len(s) > 1]

def read_allowed_words(allowed_file):
    """
    Reads allowed words from a CSV file and returns a set of lowercase words.
    These are words that have been explicitly rejected from being considered typos.

    Args:
        allowed_file (str): Path to the allowed words CSV file.

    Returns:
        set: A set of allowed words in lowercase.
    """
    allowed_words = set()
    if os.path.exists(allowed_file):
        try:
            with open(allowed_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        allowed_words.add(row[0].strip().lower())
            logging.info(f"Loaded {len(allowed_words)} allowed words from '{allowed_file}'.")
        except Exception as e:
            logging.error(f"Error reading allowed words file '{allowed_file}': {e}")
    else:
        logging.warning(f"Allowed words file '{allowed_file}' not found. Skipping allowed word filtering.")
    return allowed_words

def split_into_subwords(word):
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
        split_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', part)
        if split_parts:
            subwords.extend(split_parts)
        else:
            subwords.append(part)
    return subwords

def read_words_mapping(file_path):
    """
    Reads a CSV file of typo fixes and returns a mapping:
         incorrect_word (lowercase) -> set(corrections)
    
    Each row should be in the form:
         incorrect_word, correction1, correction2, ...

    We can also accept a list of valid words. They will
        just map to nothing.
    """
    mapping = {}
    if not os.path.exists(file_path):
        logging.error(f"Words mapping file '{file_path}' not found.")
        sys.exit(1)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    incorrect = row[0].strip().lower()
                    corrections = {col.strip().lower() for col in row[1:] if col.strip()}
                    mapping[incorrect] = corrections
        logging.info(f"Loaded mapping for {len(mapping)} words from '{file_path}'.")
    except Exception as e:
        logging.error(f"Error reading words mapping file '{file_path}': {e}")
        sys.exit(1)
    return mapping

def _validate_adjacent_context(before_words, after_words, index):
    """Return True when the neighbouring words surrounding ``index`` match."""

    previous_matches = index == 0 or before_words[index - 1] == after_words[index - 1]
    next_matches = (
        index == len(before_words) - 1
        or before_words[index + 1] == after_words[index + 1]
    )
    return previous_matches and next_matches


def _compare_word_lists(before_words, after_words, min_length):
    """Return typo pairs discovered when comparing two word sequences."""

    if len(before_words) != len(after_words):
        return []

    typos = []
    for index, (before_word, after_word) in enumerate(zip(before_words, after_words)):
        if before_word == after_word:
            continue
        if not _validate_adjacent_context(before_words, after_words, index):
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


def find_typos(diff_text, min_length=2):
    """
    Parses the diff text to identify typo corrections.

    Args:
        diff_text (str): The git diff text.
        min_length (int): Minimum length of differing substrings to consider as typos.

    Returns:
        list: A list of typo candidates in the format "before -> after".
    """
    typos = []
    lines = diff_text.split('\n')
    removals = []
    additions = []
    
    def flush_pairs():
        nonlocal typos, removals, additions
        if removals and additions:
            before_text = " ".join(removals)
            after_text = " ".join(additions)
            before_words = split_into_subwords(before_text)
            after_words = split_into_subwords(after_text)
            typos.extend(_compare_word_lists(before_words, after_words, min_length))
        removals = []
        additions = []

    for line in lines:
        if line.startswith('---') or line.startswith('+++'):
            continue
        if line.startswith('-'):
            removals.append(line[1:].strip())
        elif line.startswith('+'):
            additions.append(line[1:].strip())
        else:
            flush_pairs()

    flush_pairs()

    return typos

def lowercase_sort_dedup(input_list):
    """
    Converts all strings in the input list to lowercase, removes duplicates, and sorts them.

    Args:
        input_list (list): List of strings.

    Returns:
        list: Lowercased, sorted, and deduplicated list of strings.
    """
    return sorted(set([line.lower() for line in input_list]))

def format_typos(typos, output_format):
    """
    Formats the list of typos based on the specified output format.

    Args:
        typos (list): List of typo strings in the format "before -> after".
        output_format (str): Desired output format ('arrow', 'csv', 'table', 'list').

    Returns:
        list: Formatted list of typo strings.
    """
    formatted = []
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


class TempTypoFile:
    """Context manager that creates and cleans up a temporary typo candidate file."""

    def __enter__(self):
        handle = tempfile.NamedTemporaryFile(prefix="typos_", suffix=".txt", delete=False)
        self.path = handle.name
        handle.close()
        return self.path

    def __exit__(self, exc_type, exc, exc_tb):
        if getattr(self, "path", None) and os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                logging.debug(f"Failed to remove temporary file '{self.path}'.")


def process_new_typos(candidates, args, valid_words):
    """
    Process candidate typos (list of "before -> after") to produce
    new typos—that is, typo corrections not already registered by the typos tool.
    Applies additional filtering using allowed words and a dictionary of valid
    words. The dictionary may be a simple word list (one word per line) or a
    words.csv file, where only the correction columns are treated as valid
    words. Returns the formatted list of new typos.
    """
    filtered_candidates = candidates
    with TempTypoFile() as temp_file:
        # Save candidates to a temporary file for the typos tool.
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                for typo in candidates:
                    f.write(f"{typo}\n")
            logging.info(f"Saved candidate typos to temporary file '{temp_file}'.")
        except Exception as e:
            logging.error(f"Error writing to temporary file '{temp_file}': {e}")
            sys.exit(1)

        # Run the typos tool to filter out already known typos
        if os.path.exists(args.typos_tool_path) or os.path.exists(f"{args.typos_tool_path}.exe"):
            # Determine the correct executable based on the operating system
            if os.name == 'nt' and not args.typos_tool_path.lower().endswith('.exe'):
                typos_executable = f"{args.typos_tool_path}.exe"
            else:
                typos_executable = args.typos_tool_path

            logging.info(f"Running typos tool at '{typos_executable}' to filter known typos...")
            command = [typos_executable, '--format', 'brief', temp_file]
            try:
                result = subprocess.run(command, capture_output=True, text=True)
                already_known = extract_backticks(result.stdout)
                # Remove any candidate whose "before" word is in the list of already known typos.
                filtered_candidates = [
                    line for line in candidates
                    if line.split(' -> ')[0].lower() not in [word.lower() for word in already_known]
                ]
                logging.info(f"Filtered out {len(already_known)} already-known typo(s).")
            except subprocess.CalledProcessError:
                logging.warning("Typos tool returned a non-zero exit status. Skipping known typo filtering.")
                filtered_candidates = candidates
            except FileNotFoundError:
                logging.warning(f"Typos tool '{typos_executable}' not found. Skipping known typo filtering.")
                filtered_candidates = candidates
        else:
            logging.warning(f"Typos tool '{args.typos_tool_path}' not found. Skipping known typo filtering.")
            filtered_candidates = candidates

    # Filter out allowed words
    allowed_words = read_allowed_words(args.allowed_file)
    if allowed_words:
        before_count = len(filtered_candidates)
        progress = None
        iterator = filtered_candidates
        if not getattr(args, 'quiet', False):
            progress = tqdm(filtered_candidates, desc="Filtering allowed words", unit="typo", leave=False)
            iterator = progress
        filtered_list = []
        for typo in iterator:
            if typo.split(' -> ')[0].lower() not in allowed_words:
                filtered_list.append(typo)
        if progress:
            progress.close()
        filtered_candidates = filtered_list
        logging.info(
            f"Excluded {before_count - len(filtered_candidates)} typo(s) based on allowed words."
        )

    # Filter out cases where the "before" word is in the valid dictionary.
    # It will filter out cases where the "before" word is anywhere in the mapping. Because
    # for the new typos, you want it to be neither a correct word, nor an already-known typo.

    if valid_words:
        before_count = len(filtered_candidates)
        progress = None
        iterator = filtered_candidates
        if not getattr(args, 'quiet', False):
            progress = tqdm(filtered_candidates, desc="Filtering dictionary words", unit="typo", leave=False)
            iterator = progress
        filtered_list = []
        for typo in iterator:
            if typo.split(' -> ')[0].lower() not in valid_words:
                filtered_list.append(typo)
        if progress:
            progress.close()
        filtered_candidates = filtered_list
        logging.info(
            "Excluded "
            f"{before_count - len(filtered_candidates)} typo(s) based on valid dictionary words "
            "(or typos already in words.csv)."
        )

    # Deduplicate and sort.
    filtered_candidates = lowercase_sort_dedup(filtered_candidates)
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
    new_corrections = lowercase_sort_dedup(new_corrections)
    return new_corrections

def main():

    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Process a git diff to identify typos for the `typos` utility.")
    parser.add_argument('--input_file', type=str, default='diff.txt',
                        help="Path to the input git diff file. Use '-' to read from stdin.")
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

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting typo extraction process...")

    # Read the diff file or stdin.
    if args.input_file == '-':
        diff_text = sys.stdin.read()
        logging.info("Successfully read input diff from stdin.")
    else:
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                diff_text = f.read()
            logging.info(f"Successfully read input diff file '{args.input_file}'.")
        except UnicodeDecodeError:
            with open(args.input_file, 'r', encoding='latin-1') as f:
                diff_text = f.read()
            logging.info(
                f"Successfully read input diff file '{args.input_file}' with 'latin-1' encoding."
            )
        except FileNotFoundError:
            logging.error(f"Input file '{args.input_file}' not found. Exiting.")
            sys.exit(1)

    # Load the dictionary (words mapping) once.
    dictionary_mapping = read_words_mapping(args.dictionary_file)
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
    candidates = lowercase_sort_dedup(candidates)
    logging.info(f"Identified {len(candidates)} candidate typo correction(s).")

    # Prepare lists to hold results.
    new_typos_result = []
    new_corrections_result = []

    # Process new typos if requested.
    if args.mode in ['typos', 'both']:
        logging.info("Processing new typos (filtering out known typos)...")
        new_typos_result = process_new_typos(candidates, args, valid_words)
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
