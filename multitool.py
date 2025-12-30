import argparse
import csv
from collections import Counter
import random
import contextlib
import sys
import re
from textwrap import dedent
from typing import Callable, Iterable, List, Sequence, Tuple, TextIO, Union
from tqdm import tqdm
import logging
import ahocorasick
import json

try:
    import chardet  # type: ignore

    _CHARDET_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    chardet = None
    _CHARDET_AVAILABLE = False


def filter_to_letters(text: str) -> str:
    """Return text containing only lowercase a-z characters."""
    return re.sub("[^a-z]", "", text.lower())


def clean_and_filter(items: Iterable[str], min_length: int, max_length: int) -> List[str]:
    """Clean items to letters only and apply length filtering."""
    cleaned = [filter_to_letters(item) for item in items]
    return [c for c in cleaned if min_length <= len(c) <= max_length]


def detect_encoding(file_path: str) -> str | None:
    """Attempt to detect a file's encoding using chardet if available."""

    if not _CHARDET_AVAILABLE:
        logging.warning("chardet not installed. Install via 'pip install chardet'.")
        return None

    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result.get('encoding')
    confidence = result.get('confidence', 0)
    if encoding and confidence > 0.5:
        logging.info(
            "Detected encoding '%s' for '%s' (confidence %.2f)",
            encoding,
            file_path,
            confidence,
        )
        return encoding

    logging.warning("Failed to reliably detect encoding for '%s'.", file_path)
    return None

@contextlib.contextmanager
def smart_open_input(filename: str, encoding: str = 'utf-8', newline: str | None = None) -> Iterable[TextIO]:
    """
    Context manager that yields a file object for reading.
    If filename is '-', yields sys.stdin.
    Otherwise, opens the file for reading.
    """
    if filename == '-':
        yield sys.stdin
    else:
        with open(filename, 'r', encoding=encoding, newline=newline) as f:
            yield f

def _load_and_clean_file(
    path: str,
    min_length: int,
    max_length: int,
    *,
    split_whitespace: bool = False,
    apply_length_filter: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    """Load text items from *path* and normalize them for set-style operations."""

    raw_items = []
    cleaned_items = []
    lines = None
    used_encoding = 'utf-8'

    if path == '-':
        # For stdin, we rely on sys.stdin which is already open.
        # We assume text mode with default encoding (usually utf-8).
        # Re-opening stdin with specific encoding is possible but might be overkill.
        # We'll trust the environment or sys.stdin.encoding.
        try:
            lines = sys.stdin.readlines()
            used_encoding = sys.stdin.encoding or 'utf-8'
        except UnicodeDecodeError:
             logging.warning("Reading from stdin failed with encoding errors.")
             # Fallback logic for stdin is complex without buffering, so we might abort or try reading binary.
             # For now, let's assume valid text stream.
             lines = []
    else:
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                lines = handle.readlines()
                used_encoding = 'utf-8'
        except UnicodeDecodeError:
            logging.warning("UTF-8 decoding failed for '%s'. Attempting detection...", path)
            detected_encoding = detect_encoding(path)
            if detected_encoding:
                logging.warning(
                    "Using detected encoding '%s' for '%s'.", detected_encoding, path
                )
                try:
                    with open(path, 'r', encoding=detected_encoding) as handle:
                        lines = handle.readlines()
                    used_encoding = detected_encoding
                except UnicodeDecodeError:
                    logging.warning(
                        "Detected encoding '%s' failed for '%s'. Fallback to latin-1.",
                        detected_encoding,
                        path,
                    )
                    with open(path, 'r', encoding='latin-1') as handle:
                        lines = handle.readlines()
                    used_encoding = 'latin-1'
            else:
                logging.warning("Encoding detection failed. Fallback to latin-1 for '%s'.", path)
                with open(path, 'r', encoding='latin-1') as handle:
                    lines = handle.readlines()
                used_encoding = 'latin-1'

    logging.info("Loaded '%s' using %s encoding.", path, used_encoding)

    for line in lines or []:
        line_content = line.strip()
        if not line_content:
            continue

        parts = line_content.split() if split_whitespace else [line_content]
        for part in parts:
            raw_items.append(part)
            cleaned = filter_to_letters(part)
            if cleaned:
                cleaned_items.append(cleaned)

    if apply_length_filter:
        upper_bound = max_length
        cleaned_items = [
            item for item in cleaned_items if min_length <= len(item) <= upper_bound
        ]

    unique_items = list(dict.fromkeys(cleaned_items))
    return raw_items, cleaned_items, unique_items


def print_processing_stats(
    raw_item_count: int, filtered_items: Sequence[str], item_label: str = "item"
) -> None:
    """Print summary statistics for processed text items."""
    item_label_plural = f"{item_label}s"
    logging.info("Statistics:")
    logging.info(f"  Total {item_label_plural} encountered: {raw_item_count}")
    logging.info(
        f"  Total {item_label_plural} after filtering: {len(filtered_items)}"
    )
    if filtered_items:
        unique_items = list(dict.fromkeys(filtered_items))
        shortest = min(unique_items, key=len)
        longest = max(unique_items, key=len)
        logging.info(
            f"  Shortest {item_label}: '{shortest}' (length: {len(shortest)})"
        )
        logging.info(
            f"  Longest {item_label}: '{longest}' (length: {len(longest)})"
        )
    else:
        logging.info(f"  No {item_label_plural} passed the filtering criteria.")


@contextlib.contextmanager
def smart_open_output(filename: str, encoding: str = 'utf-8', newline: str | None = None) -> Iterable[TextIO]:
    """
    Context manager that yields a file object for writing.
    If filename is '-', yields sys.stdout.
    Otherwise, opens the file for writing.
    """
    if filename == '-':
        yield sys.stdout
    else:
        with open(filename, 'w', encoding=encoding, newline=newline) as f:
            yield f

def write_output(
    items: Iterable[str],
    output_file: str,
    output_format: str = 'line',
    quiet: bool = False
) -> None:
    """
    Writes a collection of strings to the output file in the specified format.

    Supported formats:
      - line: One item per line (default)
      - json: JSON array of strings
      - csv: One item per row (vertical)
      - markdown: Markdown bullet list
    """
    items_list = list(items)  # Consume generator to know length/content

    # Use newline='' for CSV format to ensure correct line endings across platforms
    newline = '' if output_format == 'csv' else None

    with smart_open_output(output_file, newline=newline) as outfile:
        if output_format == 'json':
            json.dump(items_list, outfile, indent=2)
            outfile.write('\n')
        elif output_format == 'csv':
            writer = csv.writer(outfile)
            for item in items_list:
                writer.writerow([item])
        elif output_format == 'markdown':
            for item in items_list:
                outfile.write(f"- {item}\n")
        else:  # 'line' or fallback
            for item in items_list:
                outfile.write(item + '\n')


def _process_items(
    extractor_func: Callable[[str, bool], Iterable[str]],
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    mode_name: str,
    success_msg: str,
    output_format: str = 'line',
    quiet: bool = False,
) -> None:
    """Generic processing for modes that extract raw string items from one or more files."""

    def chained_extractor() -> Iterable[str]:
        for input_file in input_files:
            yield from extractor_func(input_file, quiet=quiet)

    raw_items = list(chained_extractor())
    filtered_items = clean_and_filter(raw_items, min_length, max_length)
    if process_output:
        filtered_items = sorted(set(filtered_items))

    write_output(filtered_items, output_file, output_format, quiet)

    print_processing_stats(len(raw_items), filtered_items)
    logging.info(
        f"[{mode_name} Mode] {success_msg} Output written to '{output_file}'."
    )


def _extract_arrow_items(input_file: str, right_side: bool = False, quiet: bool = False) -> Iterable[str]:
    """Yield text before (or after) ' -> ' from each line."""
    with smart_open_input(input_file, encoding='utf-8') as infile:
        for line in tqdm(infile, desc=f'Processing {input_file} (arrow)', unit=' lines', disable=quiet):
            if " -> " in line:
                parts = line.split(" -> ", 1)
                idx = 1 if right_side else 0
                if len(parts) > idx:
                    yield parts[idx].strip()


def _extract_backtick_items(input_file: str, quiet: bool = False) -> Iterable[str]:
    """Yield text found between backticks with heuristics for diagnostics."""

    context_markers = ("error:", "warning:", "note:")

    with smart_open_input(input_file, encoding='utf-8') as infile:
        for line in tqdm(infile, desc=f'Processing {input_file} (backtick)', unit=' lines', disable=quiet):
            # Split the line on backticks to inspect the surrounding context of
            # each candidate substring. This helps avoid extracting identifiers
            # from file paths when a later pair of backticks contains the actual
            # typo from messages such as "error: `foo` should be `bar`".
            parts = line.split('`')
            selected = None
            if len(parts) >= 3:
                for index in range(1, len(parts), 2):
                    preceding = parts[index - 1].lower() if index - 1 >= 0 else ""
                    for marker in context_markers:
                        if marker in preceding:
                            selected = parts[index].strip()
                            break
                    if selected:
                        break

            if selected is None and len(parts) >= 3:
                # Fallback: extract the content of the first backticked item.
                # parts[0] is text before first `, parts[1] is text between first and second `, etc.
                selected = parts[1].strip()

            if selected:
                yield selected


def _extract_json_items(input_file: str, key_path: str, quiet: bool = False) -> Iterable[str]:
    """Yield values from JSON objects based on a dotted key path."""

    def traverse_json(data, path_parts):
        # If it's a list, apply the current path traversal to every item
        if isinstance(data, list):
            for item in data:
                yield from traverse_json(item, path_parts)
            return

        # If we are at the end of the path, yield the current item
        if not path_parts:
            # We yield the string representation of the data
            yield str(data)
            return

        current_key = path_parts[0]
        if isinstance(data, dict):
            if current_key in data:
                yield from traverse_json(data[current_key], path_parts[1:])

    path_parts = key_path.split('.') if key_path else []

    with smart_open_input(input_file, encoding='utf-8') as infile:
        # Load the entire file content as JSON
        # Note: Standard JSON parsers expect the whole file. Streaming JSON (JSONL) is handled differently.
        # Here we assume standard JSON as output by typostats.py.
        try:
            content = infile.read()
            if not content.strip():
                return
            data = json.loads(content)
            yield from traverse_json(data, path_parts)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON in '{input_file}': {e}")
            return


def _extract_csv_items(
    input_file: str, first_column: bool, delimiter: str = ',', quiet: bool = False
) -> Iterable[str]:
    """Yield fields from CSV rows based on column selection."""
    with smart_open_input(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in tqdm(reader, desc=f'Processing {input_file} (CSV)', unit=' rows', disable=quiet):
            if first_column:
                if row:
                    yield row[0].strip()
            else:
                if len(row) >= 2:
                    for field in row[1:]:
                        yield field.strip()


def _extract_line_items(input_file: str, quiet: bool = False) -> Iterable[str]:
    """Yield each line from the file."""
    with smart_open_input(input_file, encoding='utf-8') as infile:
        for line in tqdm(infile, desc=f'Processing {input_file} (lines)', unit=' lines', disable=quiet):
            yield line.rstrip('\n')


def arrow_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    right_side: bool = False,
    output_format: str = 'line',
    quiet: bool = False,
) -> None:
    """Wrapper for processing items separated by ' -> '."""
    extractor = lambda f, quiet=False: _extract_arrow_items(f, right_side=right_side, quiet=quiet)
    _process_items(extractor, input_files, output_file, min_length, max_length, process_output, 'Arrow', 'File(s) processed successfully.', output_format, quiet)


def backtick_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    output_format: str = 'line',
    quiet: bool = False,
) -> None:
    """Wrapper for extracting text between backticks."""
    _process_items(_extract_backtick_items, input_files, output_file, min_length, max_length, process_output, 'Backtick', 'Strings extracted successfully.', output_format, quiet)


def json_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    key: str,
    output_format: str = 'line',
    quiet: bool = False,
) -> None:
    """Wrapper for extracting fields from JSON files."""
    extractor = lambda f, quiet=False: _extract_json_items(f, key, quiet=quiet)
    _process_items(extractor, input_files, output_file, min_length, max_length, process_output, 'JSON', 'JSON values extracted successfully.', output_format, quiet)


def count_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    quiet: bool = False,
) -> None:
    """
    Counts the frequency of each word in the input file(s) and writes the
    sorted results to the output file. Only words with length between
    min_length and max_length are counted.
    The stats are based on the raw count of words versus the filtered words.
    Note: process_output is ignored in count mode.
    """
    raw_count = 0
    filtered_words = []
    word_counts = Counter()

    for input_file in input_files:
        with smart_open_input(input_file, encoding='utf-8') as file:
            for line in tqdm(file, desc=f'Counting words in {input_file}', unit=' lines', disable=quiet):
                words = [word.strip() for word in line.split()]
                raw_count += len(words)
                filtered = []
                for word in words:
                    cleaned = filter_to_letters(word)
                    if min_length <= len(cleaned) <= max_length:
                        filtered.append(cleaned)
                filtered_words.extend(filtered)
                word_counts.update(filtered)

    sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
    with smart_open_output(output_file) as out_file:
        for word, count in sorted_words:
            out_file.write(f"{word}: {count}\n")
    print_processing_stats(raw_count, filtered_words, item_label="word")
    logging.info(
        f"[Count Mode] Word frequencies have been written to '{output_file}'."
    )

def check_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    output_format: str = 'line',
    quiet: bool = False,
) -> None:
    """
    Checks CSV file(s) of typos and corrections for any words that appear
    as both a typo and a correction anywhere in the dataset.
    """
    typos = set()
    corrections = set()

    for input_file in input_files:
        with smart_open_input(input_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in tqdm(reader, desc=f'Checking {input_file}', unit=' rows', disable=quiet):
                if not row:
                    continue
                typos.add(row[0].strip())
                for field in row[1:]:
                    corrections.add(field.strip())

    duplicates = list(typos & corrections)
    filtered_items = clean_and_filter(duplicates, min_length, max_length)
    if process_output:
        filtered_items = list(set(filtered_items))
    filtered_items.sort()

    write_output(filtered_items, output_file, output_format, quiet)

    print_processing_stats(len(duplicates), filtered_items)
    logging.info(
        f"[Check Mode] Found {len(filtered_items)} overlapping words across {len(input_files)} file(s). Output written to '{output_file}'."
    )


def csv_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    first_column: bool = False,
    delimiter: str = ',',
    output_format: str = 'line',
    quiet: bool = False,
) -> None:
    """Wrapper for extracting fields from CSV files."""
    extractor = lambda f, quiet=False: _extract_csv_items(f, first_column, delimiter, quiet=quiet)
    _process_items(extractor, input_files, output_file, min_length, max_length, process_output, 'CSV', 'CSV fields extracted successfully.', output_format, quiet)


def line_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    output_format: str = 'line',
    quiet: bool = False,
) -> None:
    """Wrapper for processing raw lines from file(s)."""
    _process_items(_extract_line_items, input_files, output_file, min_length, max_length, process_output, 'Line', 'Lines processed successfully.', output_format, quiet)


def combine_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    output_format: str = 'line',
    quiet: bool = False,
) -> None:
    """Merge cleaned contents from multiple files into one deduplicated list."""

    raw_item_count = 0
    combined_unique: list[str] = []

    for file_path in input_files:
        raw_items, cleaned_items, unique_items = _load_and_clean_file(
            file_path,
            min_length,
            max_length,
        )
        raw_item_count += len(raw_items)
        combined_unique.extend(unique_items)

    combined_unique = list(dict.fromkeys(combined_unique))
    if process_output:
        combined_unique = sorted(set(combined_unique))
    else:
        combined_unique = sorted(combined_unique)

    write_output(combined_unique, output_file, output_format, quiet)

    print_processing_stats(raw_item_count, combined_unique)
    logging.info(
        "[Combine Mode] Combined %d file(s). Output written to '%s'.",
        len(input_files),
        output_file,
    )


def sample_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    sample_count: int | None = None,
    sample_percent: float | None = None,
    output_format: str = 'line',
    quiet: bool = False,
) -> None:
    """Randomly sample lines from the input file(s)."""

    def chained_extractor() -> Iterable[str]:
        for input_file in input_files:
            yield from _extract_line_items(input_file, quiet=quiet)

    # Extract raw items first
    raw_items = list(chained_extractor())

    if not raw_items:
        logging.warning("Input is empty or no lines found.")
        # Create empty output using write_output to ensure consistent formatting (e.g. empty JSON list)
        write_output([], output_file, output_format, quiet)
        return

    # Clean and filter BEFORE sampling to ensure the requested count is accurate relative to valid items
    cleaned_items = clean_and_filter(raw_items, min_length, max_length)

    total_valid_items = len(cleaned_items)

    if sample_count is not None:
        k = min(sample_count, total_valid_items)
    elif sample_percent is not None:
        k = int(total_valid_items * (sample_percent / 100.0))
        k = max(0, min(k, total_valid_items))
    else:
        k = total_valid_items

    sampled_items = random.sample(cleaned_items, k)

    if process_output:
        sampled_items = sorted(set(sampled_items))

    write_output(sampled_items, output_file, output_format, quiet)

    print_processing_stats(len(raw_items), sampled_items)
    logging.info(
        f"[Sample Mode] Sampled {k}/{total_valid_items} valid lines from {len(input_files)} file(s). Output written to '{output_file}'."
    )


def _add_common_mode_arguments(
    subparser: argparse.ArgumentParser, include_process_output: bool = True
) -> None:
    """Attach shared CLI arguments to a mode-specific subparser."""
    subparser.add_argument(
        'input_files_pos',
        nargs='*',
        help="Path(s) to the input file(s). Defaults to 'input.txt' if none provided.",
    )
    subparser.add_argument(
        '--input',
        dest='input_files_flag',
        type=str,
        nargs='+',
        help="Path(s) to the input file(s) (legacy flag, supports multiple).",
    )
    subparser.add_argument(
        '--output',
        type=str,
        default='-',
        help="Path to the output file (default: stdout).",
    )
    subparser.add_argument(
        '--output-format',
        choices=['line', 'json', 'csv', 'markdown'],
        default='line',
        help="Format of the output (default: line). Options: line, json, csv, markdown.",
    )
    subparser.add_argument(
        '--min-length',
        type=int,
        default=3,
        help="Minimum string length to process (default: 3)",
    )
    subparser.add_argument(
        '--max-length',
        type=int,
        default=1000,
        help="Maximum string length to process (default: 1000)",
    )
    if include_process_output:
        subparser.add_argument(
            '--process-output',
            action='store_true',
            help="If set, converts output to lowercase, sorts it, and removes duplicates.",
        )
    else:
        subparser.set_defaults(process_output=False)


def filter_fragments_mode(
    input_files: Sequence[str],
    file2: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    output_format: str = 'line',
    quiet: bool = False,
) -> None:
    """
    Filters words from input_files (list1) that do not appear as substrings of any
    word in file2 (list2).
    """

    # Load and merge all input files
    all_raw_list1 = []
    all_cleaned_list1 = []

    for input_file in input_files:
        raw, cleaned, _ = _load_and_clean_file(
            input_file,
            min_length,
            max_length,
            apply_length_filter=False,
        )
        all_raw_list1.extend(raw)
        all_cleaned_list1.extend(cleaned)

    _, _, unique_list2 = _load_and_clean_file(
        file2,
        min_length,
        max_length,
        split_whitespace=True,
        apply_length_filter=False,
    )

    # Aho-Corasick automaton for efficient substring matching
    auto = ahocorasick.Automaton()
    for keyword in all_cleaned_list1:
        auto.add_word(keyword, keyword)
    auto.make_automaton()

    matched_words = set()
    for item in tqdm(unique_list2, desc="Finding matches", disable=quiet):
        for end_index, keyword in auto.iter(item):
            matched_words.add(keyword)

    non_matches = [word for word in all_cleaned_list1 if word not in matched_words]
    filtered_items = clean_and_filter(non_matches, min_length, max_length)

    if process_output:
        filtered_items = list(set(filtered_items))
        filtered_items.sort()

    write_output(filtered_items, output_file, output_format, quiet)

    print_processing_stats(len(all_raw_list1), filtered_items)
    logging.info(
        f"[FilterFragments Mode] Filtering complete. Results saved to '{output_file}'."
    )


def set_operation_mode(
    input_files: Sequence[str],
    file2: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    operation: str,
    output_format: str = 'line',
    quiet: bool = False,
) -> None:
    """Perform set operations (intersection, union, difference) between input files (merged) and a second file."""
    allowed_operations = {'intersection', 'union', 'difference'}
    if operation not in allowed_operations:
        raise ValueError(
            f"Invalid operation '{operation}'. Must be one of: {', '.join(sorted(allowed_operations))}."
        )

    # Load and merge all input files
    raw_item_count_a = 0
    unique_a_list = []

    for input_file in input_files:
        raw, _, unique = _load_and_clean_file(
            input_file, min_length, max_length
        )
        raw_item_count_a += len(raw)
        unique_a_list.extend(unique)

    unique_a = list(dict.fromkeys(unique_a_list))

    raw_items_b, _, unique_b = _load_and_clean_file(
        file2, min_length, max_length
    )

    set_b = set(unique_b)

    if operation == 'intersection':
        result_items = [item for item in unique_a if item in set_b]
    elif operation == 'union':
        result_items = list(dict.fromkeys(unique_a + unique_b))
    else:  # difference
        result_items = [item for item in unique_a if item not in set_b]

    if process_output:
        result_items = sorted(set(result_items))

    write_output(result_items, output_file, output_format, quiet)

    print_processing_stats(raw_item_count_a + len(raw_items_b), result_items)
    logging.info(
        f"[Set Operation Mode] Completed {operation} between {len(input_files)} input file(s) and "
        f"'{file2}'. Output written to '{output_file}'."
    )

MODE_DETAILS = {
    "arrow": {
        "summary": "Extract the left (or right) side of '->' arrows.",
        "description": "Reads lines like 'typo -> correction' and saves just the 'typo' part. Use --right to extract the correction instead.",
        "example": "python multitool.py arrow typos.log --right --output corrections.txt",
    },
    "combine": {
        "summary": "Merge and sort multiple files.",
        "description": "Reads multiple files, combines them into one list, removes duplicates, and sorts them alphabetically.",
        "example": "python multitool.py combine typos1.txt typos2.txt --output all_typos.txt",
    },
    "backtick": {
        "summary": "Extract text inside backticks.",
        "description": "Extracts text enclosed in backticks (like `this`). Smartly picks the most relevant item from lines containing 'error:', 'warning:', or 'note:'.",
        "example": "python multitool.py backtick build.log --output suspects.txt",
    },
    "csv": {
        "summary": "Extract columns from a CSV file.",
        "description": "Extracts data from CSV files. By default, it grabs everything *except* the first column, which is perfect for getting a list of corrections.",
        "example": "python multitool.py csv typos.csv --output corrections.txt",
    },
    "json": {
        "summary": "Extract values from JSON files.",
        "description": "Extracts values associated with a specific key path from a JSON file. Supports dot notation for nested keys (e.g. 'replacements.typo'). Handles lists automatically.",
        "example": "python multitool.py json report.json --key replacements.typo --output typos.txt",
    },
    "line": {
        "summary": "Process a file line by line.",
        "description": "Reads each line, filters it (if requested), and writes it to the output. A simple way to clean up a text file.",
        "example": "python multitool.py line raw_words.txt --output filtered.txt",
    },
    "count": {
        "summary": "Count how often words appear.",
        "description": "Counts the frequency of each word in the file and lists them from most frequent to least frequent.",
        "example": "python multitool.py count typos.log --output counts.txt",
    },
    "filterfragments": {
        "summary": "Remove words found inside another file.",
        "description": "Filters out words that are already present (even as substrings) in a second file, such as a dictionary.",
        "example": "python multitool.py filterfragments generated.txt --file2 dictionary.txt --output unique.txt",
    },
    "check": {
        "summary": "Find words that are both a typo and a correction.",
        "description": "Identifies words that are listed as both a typo and a correction. This helps find errors in your typo database.",
        "example": "python multitool.py check typos.csv --output duplicates.txt",
    },
    "set_operation": {
        "summary": "Compare two files (intersection, union, difference).",
        "description": "Finds common lines (intersection), combines all lines (union), or finds lines in one file but not the other (difference).",
        "example": "python multitool.py set_operation fileA.txt --file2 fileB.txt --operation intersection --output shared.txt",
    },
    "sample": {
        "summary": "Randomly sample lines from a file.",
        "description": "Extracts a random subset of lines from the input file. You can specify exact count (--n) or percentage (--percent).",
        "example": "python multitool.py sample big_log.txt --n 100 --output sample.txt",
    },
}


class ModeHelpAction(argparse.Action):
    """Custom argparse action that prints detailed help for one or all modes."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | None,
        option_string: str | None = None,
    ) -> None:
        if values in (None, "all"):
            # Show a summary table of all modes
            print("\nAvailable Modes:\n")
            max_len = max(len(m) for m in MODE_DETAILS.keys())
            width = max_len + 4
            for mode, details in MODE_DETAILS.items():
                print(f"  {mode:<{width}} {details['summary']}")
            print("\nRun 'python multitool.py --mode-help <mode>' for details on a specific mode.\n")
            parser.exit()
        else:
            # Show detailed help for a single mode
            details = MODE_DETAILS.get(values)
            if not details:
                parser.error(f"Unknown mode: {values}")

            block = [f"Mode: {values}", f"  Summary: {details['summary']}"]
            if details.get("description"):
                block.append(f"  Description: {details['description']}")
            if details.get("example"):
                block.append(f"  Example: {details['example']}")
            parser.exit(message="\n" + "\n".join(block) + "\n\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multipurpose File Processing Tool",
        epilog=dedent(
            """
            Examples:
              python multitool.py --mode-help             # Show a summary of every mode
              python multitool.py --mode-help csv         # Describe the CSV extraction mode
              python multitool.py arrow --input file.txt  # Run a specific mode
              python multitool.py --mode csv --input file.txt  # Legacy --mode flag
            """
        ).strip(),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--mode-help",
        nargs="?",
        choices=[*MODE_DETAILS.keys(), "all"],
        action=ModeHelpAction,
        help="Display extended documentation for a specific mode or all modes.",
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress bars and informational log output.',
    )

    subparsers = parser.add_subparsers(dest='mode', required=True, metavar='mode')

    arrow_parser = subparsers.add_parser(
        'arrow',
        help=MODE_DETAILS['arrow']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['arrow']['description'],
    )
    _add_common_mode_arguments(arrow_parser)
    arrow_parser.add_argument(
        '--right',
        action='store_true',
        help="Extract the right side (correction) instead of the left side (typo).",
    )

    backtick_parser = subparsers.add_parser(
        'backtick',
        help=MODE_DETAILS['backtick']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['backtick']['description'],
    )
    _add_common_mode_arguments(backtick_parser)

    csv_parser = subparsers.add_parser(
        'csv',
        help=MODE_DETAILS['csv']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['csv']['description'],
    )
    _add_common_mode_arguments(csv_parser)
    csv_parser.add_argument(
        '--first-column',
        action='store_true',
        help='Extract the first column instead of subsequent columns.',
    )
    csv_parser.add_argument(
        '--delimiter',
        type=str,
        default=',',
        help='The delimiter character for CSV files (default: ,).',
    )

    json_parser = subparsers.add_parser(
        'json',
        help=MODE_DETAILS['json']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['json']['description'],
    )
    _add_common_mode_arguments(json_parser)
    json_parser.add_argument(
        '--key',
        type=str,
        required=True,
        help="The key path to extract (e.g. 'items.name').",
    )

    combine_parser = subparsers.add_parser(
        'combine',
        help=MODE_DETAILS['combine']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['combine']['description'],
    )
    combine_parser.add_argument(
        'input_files_pos',
        nargs='*',
        help='Paths to the input files to merge.',
    )
    combine_parser.add_argument(
        '--input',
        dest='input_files_flag',
        type=str,
        nargs='+',
        help='Paths to the input files to merge (legacy flag).',
    )
    combine_parser.add_argument(
        '--output',
        type=str,
        default='-',
        help="Path to the output file (default: stdout).",
    )
    combine_parser.add_argument(
        '--output-format',
        choices=['line', 'json', 'csv', 'markdown'],
        default='line',
        help="Format of the output (default: line). Options: line, json, csv, markdown.",
    )
    combine_parser.add_argument(
        '--min-length',
        type=int,
        default=3,
        help="Minimum string length to process (default: 3)",
    )
    combine_parser.add_argument(
        '--max-length',
        type=int,
        default=1000,
        help="Maximum string length to process (default: 1000)",
    )
    combine_parser.add_argument(
        '--process-output',
        action='store_true',
        help="If set, converts output to lowercase, sorts it, and removes duplicates.",
    )

    line_parser = subparsers.add_parser(
        'line',
        help=MODE_DETAILS['line']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['line']['description'],
    )
    _add_common_mode_arguments(line_parser)

    count_parser = subparsers.add_parser(
        'count',
        help=MODE_DETAILS['count']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['count']['description'],
    )
    _add_common_mode_arguments(count_parser, include_process_output=False)

    filter_parser = subparsers.add_parser(
        'filterfragments',
        help=MODE_DETAILS['filterfragments']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['filterfragments']['description'],
    )
    _add_common_mode_arguments(filter_parser)
    filter_parser.add_argument(
        '--file2',
        type=str,
        required=True,
        help='Path to the second file used for comparison.',
    )

    check_parser = subparsers.add_parser(
        'check',
        help=MODE_DETAILS['check']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['check']['description'],
    )
    _add_common_mode_arguments(check_parser)

    set_parser = subparsers.add_parser(
        'set_operation',
        help=MODE_DETAILS['set_operation']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['set_operation']['description'],
    )
    _add_common_mode_arguments(set_parser)
    set_parser.add_argument(
        '--file2',
        type=str,
        required=True,
        help='Path to the second input file for set comparisons.',
    )
    set_parser.add_argument(
        '--operation',
        type=str,
        choices=['intersection', 'union', 'difference'],
        required=True,
        help='Set operation to perform between the two files.',
    )

    sample_parser = subparsers.add_parser(
        'sample',
        help=MODE_DETAILS['sample']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['sample']['description'],
    )
    _add_common_mode_arguments(sample_parser)
    group = sample_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--n',
        dest='sample_count',
        type=int,
        help='Number of lines to sample.',
    )
    group.add_argument(
        '--percent',
        dest='sample_percent',
        type=float,
        help='Percentage of lines to sample (0-100).',
    )

    return parser


def _normalize_mode_args(
    argv: Sequence[str], parser: argparse.ArgumentParser
) -> List[str]:
    """Normalize legacy --mode usage into a positional subcommand."""
    if "--mode" not in argv:
        return list(argv)

    argv_list = list(argv)
    if argv_list.count("--mode") > 1:
        parser.error("Only one --mode flag may be provided.")

    mode_index = argv_list.index("--mode")
    if mode_index == len(argv_list) - 1:
        parser.error("--mode requires a value.")

    mode_value = argv_list[mode_index + 1]
    positional_mode = (
        argv_list[0] if argv_list and argv_list[0] in MODE_DETAILS else None
    )
    if positional_mode and positional_mode != mode_value:
        parser.error(
            f"--mode '{mode_value}' conflicts with positional mode '{positional_mode}'."
        )

    del argv_list[mode_index : mode_index + 2]
    if not positional_mode:
        argv_list.insert(0, mode_value)

    return argv_list


def main() -> None:
    parser = _build_parser()
    argv = _normalize_mode_args(sys.argv[1:], parser)

    args = parser.parse_args(argv)

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.min_length < 1:
        logging.error("[Error] --min-length must be a positive integer.")
        sys.exit(1)
    if args.max_length < args.min_length:
        logging.error("[Error] --max-length must be greater than or equal to --min-length.")
        sys.exit(1)

    logging.info(f"Selected Mode: {args.mode}")

    # Resolve input arguments (positional vs flag)
    pos_inputs = getattr(args, 'input_files_pos', []) or []
    flag_inputs = getattr(args, 'input_files_flag', []) or []
    input_paths = pos_inputs + flag_inputs

    # Default to stdin ('-') if neither is provided
    if not input_paths:
        input_paths = ['-']

    # Store for logging and handler
    args.input = input_paths

    input_label = "Input Files" if len(input_paths) > 1 else "Input File"
    logging.info(f"{input_label}: {', '.join(input_paths)}")
    logging.info(f"Output File: {args.output}")

    logging.info(f"Minimum String Length: {args.min_length}")
    logging.info(f"Maximum String Length: {args.max_length}")

    if args.mode != 'count':
        logging.info(
            f"Process Output: {'Enabled' if args.process_output else 'Disabled'}"
        )

    file2 = getattr(args, 'file2', None)
    operation = getattr(args, 'operation', None)
    first_column = getattr(args, 'first_column', False)
    delimiter = getattr(args, 'delimiter', ',')
    sample_count = getattr(args, 'sample_count', None)
    sample_percent = getattr(args, 'sample_percent', None)
    output_format = getattr(args, 'output_format', 'line')

    if args.mode in {'filterfragments', 'set_operation'} and file2:
        logging.info(f"File2: {file2}")
    if args.mode == 'set_operation' and operation:
        logging.info(f"Set Operation: {operation}")
    if args.mode == 'csv':
        logging.info(f"First Column Only: {'Yes' if first_column else 'No'}")
        logging.info(f"Delimiter: '{delimiter}'")
    if args.mode == 'json':
        logging.info(f"JSON Key Path: '{getattr(args, 'key', '')}'")
    if args.mode == 'sample':
        if sample_count is not None:
            logging.info(f"Sampling Count: {sample_count}")
        if sample_percent is not None:
            logging.info(f"Sampling Percentage: {sample_percent}%")

    logging.info(f"Output Format: {output_format}")

    common_kwargs = {
        'input_files': args.input,
        'output_file': args.output,
        'min_length': args.min_length,
        'max_length': args.max_length,
        'process_output': getattr(args, 'process_output', False),
        'quiet': args.quiet,
    }

    # Check for arrow-specific args
    right_side = getattr(args, 'right', False)

    handler_map = {
        'arrow': (arrow_mode, {**common_kwargs, 'right_side': right_side, 'output_format': output_format}),
        'backtick': (backtick_mode, {**common_kwargs, 'output_format': output_format}),
        'csv': (
            csv_mode,
            {
                **common_kwargs,
                'first_column': first_column,
                'delimiter': delimiter,
                'output_format': output_format,
            },
        ),
        'json': (
            json_mode,
            {
                **common_kwargs,
                'key': getattr(args, 'key', ''),
                'output_format': output_format,
            },
        ),
        'line': (line_mode, {**common_kwargs, 'output_format': output_format}),
        'count': (count_mode, dict(common_kwargs)), # count_mode ignores output_format
        'filterfragments': (
            filter_fragments_mode,
            {**common_kwargs, 'file2': file2, 'output_format': output_format},
        ),
        'check': (check_mode, {**common_kwargs, 'output_format': output_format}),
        'set_operation': (
            set_operation_mode,
            {**common_kwargs, 'file2': file2, 'operation': operation, 'output_format': output_format},
        ),
        'combine': (
            combine_mode,
            {
                'input_files': input_paths,
                'output_file': args.output,
                'min_length': args.min_length,
                'max_length': args.max_length,
                'process_output': getattr(args, 'process_output', False),
                'quiet': args.quiet,
                'output_format': output_format,
            },
        ),
        'sample': (
            sample_mode,
            {
                **common_kwargs,
                'sample_count': sample_count,
                'sample_percent': sample_percent,
                'output_format': output_format,
            },
        ),
    }

    handler, handler_args = handler_map[args.mode]
    try:
        handler(**handler_args)
    except FileNotFoundError as e:
        # If the exception has a filename attribute (common in OSError), use it.
        # Otherwise, fall back to a generic message.
        filename = getattr(e, 'filename', None)
        if filename:
            logging.error(f"[Error] File not found: '{filename}'")
        else:
            logging.error(f"[Error] File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"[Error] An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
