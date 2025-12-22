import argparse
import csv
from collections import Counter
import sys
import re
from textwrap import dedent
from typing import Callable, Iterable, List, Sequence, Tuple
from tqdm import tqdm
import logging
import ahocorasick

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


def _process_items(
    extractor_func: Callable[[str, bool], Iterable[str]],
    input_file: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    mode_name: str,
    success_msg: str,
    quiet: bool = False,
) -> None:
    """Generic processing for modes that extract raw string items from a file."""
    try:
        raw_items = list(extractor_func(input_file, quiet=quiet))
        filtered_items = clean_and_filter(raw_items, min_length, max_length)
        if process_output:
            filtered_items = sorted(set(filtered_items))
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for item in filtered_items:
                outfile.write(item + '\n')
        print_processing_stats(len(raw_items), filtered_items)
        logging.info(
            f"[{mode_name} Mode] {success_msg} Output written to '{output_file}'."
        )
    except FileNotFoundError:
        logging.error(f"[{mode_name} Mode] Error: Input file not found at '{input_file}'")
    except Exception as e:
        logging.error(f"[{mode_name} Mode] An unexpected error occurred: {e}")


def _extract_arrow_items(input_file: str, quiet: bool = False) -> Iterable[str]:
    """Yield text before ' -> ' from each line."""
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, desc='Processing lines (arrow mode)', unit=' lines', disable=quiet):
            if " -> " in line:
                yield line.split(" -> ", 1)[0].strip()


def _extract_backtick_items(input_file: str, quiet: bool = False) -> Iterable[str]:
    """Yield text found between backticks with heuristics for diagnostics."""

    context_markers = ("error:", "warning:", "note:")

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, desc='Processing lines (backtick mode)', unit=' lines', disable=quiet):
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

            if selected is None:
                start_index = line.find('`')
                end_index = line.find('`', start_index + 1) if start_index != -1 else -1
                if start_index != -1 and end_index != -1:
                    selected = line[start_index + 1:end_index].strip()

            if selected:
                yield selected


def _extract_csv_items(
    input_file: str, first_column: bool, delimiter: str = ',', quiet: bool = False
) -> Iterable[str]:
    """Yield fields from CSV rows based on column selection."""
    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in tqdm(reader, desc='Processing CSV rows', unit=' rows', disable=quiet):
            if first_column:
                if row:
                    yield row[0].strip()
            else:
                if len(row) >= 2:
                    for field in row[1:]:
                        yield field.strip()


def _extract_line_items(input_file: str, quiet: bool = False) -> Iterable[str]:
    """Yield each line from the file."""
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, desc='Processing lines (line mode)', unit=' lines', disable=quiet):
            yield line.rstrip('\n')


def arrow_mode(
    input_file: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    quiet: bool = False,
) -> None:
    """Wrapper for processing items separated by ' -> '."""
    _process_items(_extract_arrow_items, input_file, output_file, min_length, max_length, process_output, 'Arrow', 'File processed successfully.', quiet)


def backtick_mode(
    input_file: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    quiet: bool = False,
) -> None:
    """Wrapper for extracting text between backticks."""
    _process_items(_extract_backtick_items, input_file, output_file, min_length, max_length, process_output, 'Backtick', 'Strings extracted successfully.', quiet)

def count_mode(
    input_file: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    quiet: bool = False,
) -> None:
    """
    Counts the frequency of each word in the input file and writes the
    sorted results to the output file. Only words with length between
    min_length and max_length are counted.
    The stats are based on the raw count of words versus the filtered words.
    Note: process_output is ignored in count mode.
    """
    try:
        raw_count = 0
        filtered_words = []
        word_counts = Counter()
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc='Counting words', unit=' lines', disable=quiet):
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
        with open(output_file, 'w', encoding='utf-8') as out_file:
            for word, count in sorted_words:
                out_file.write(f"{word}: {count}\n")
        print_processing_stats(raw_count, filtered_words, item_label="word")
        logging.info(
            f"[Count Mode] Word frequencies have been written to '{output_file}'."
        )
    except FileNotFoundError:
        logging.error(f"[Count Mode] Error: Input file not found at '{input_file}'")
    except Exception as e:
        logging.error(f"[Count Mode] An unexpected error occurred while processing '{input_file}': {e}")

def check_mode(
    input_file: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    quiet: bool = False,
) -> None:
    """
    Checks a CSV file of typos and corrections for any words that appear
    as both a typo and a correction anywhere in the file. The CSV is
    assumed to have the typo in the first column and one or more
    corrections in subsequent columns.

    The intersection of all typo words and all correction words is
    written to the output file. Standard length filtering and optional
    output processing (lowercasing, deduping, sorting) are applied.
    """
    try:
        typos = set()
        corrections = set()
        with open(input_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in tqdm(reader, desc='Checking CSV for overlaps', unit=' rows', disable=quiet):
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
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for word in filtered_items:
                outfile.write(word + '\n')
        print_processing_stats(len(duplicates), filtered_items)
        logging.info(
            f"[Check Mode] Found {len(filtered_items)} overlapping words. Output written to '{output_file}'."
        )
    except FileNotFoundError:
        logging.error(f"[Check Mode] Error: Input file not found at '{input_file}'")
    except Exception as e:
        logging.error(f"[Check Mode] An unexpected error occurred while processing '{input_file}': {e}")


def csv_mode(
    input_file: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    first_column: bool = False,
    delimiter: str = ',',
    quiet: bool = False,
) -> None:
    """Wrapper for extracting fields from CSV files."""
    extractor = lambda f, quiet=False: _extract_csv_items(f, first_column, delimiter, quiet=quiet)
    _process_items(extractor, input_file, output_file, min_length, max_length, process_output, 'CSV', 'CSV fields extracted successfully.', quiet)


def line_mode(
    input_file: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    quiet: bool = False,
) -> None:
    """Wrapper for processing raw lines from a file."""
    _process_items(_extract_line_items, input_file, output_file, min_length, max_length, process_output, 'Line', 'Lines processed successfully.', quiet)


def combine_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    quiet: bool = False,
) -> None:
    """Merge cleaned contents from multiple files into one deduplicated list."""

    try:
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

        with open(output_file, 'w', encoding='utf-8') as outfile:
            for item in combined_unique:
                outfile.write(item + '\n')

        print_processing_stats(raw_item_count, combined_unique)
        logging.info(
            "[Combine Mode] Combined %d file(s). Output written to '%s'.",
            len(input_files),
            output_file,
        )
    except FileNotFoundError as e:
        logging.error(f"[Combine Mode] Error: File not found at '{e.filename}'")
    except Exception as e:
        logging.error(f"[Combine Mode] An unexpected error occurred: {e}")

def _add_common_mode_arguments(
    subparser: argparse.ArgumentParser, include_process_output: bool = True
) -> None:
    """Attach shared CLI arguments to a mode-specific subparser."""
    subparser.add_argument(
        '--input',
        type=str,
        default='input.txt',
        help="Path to the input file (default: input.txt)",
    )
    subparser.add_argument(
        '--output',
        type=str,
        default='output.txt',
        help="Path to the output file (default: output.txt)",
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
    input_file: str,
    file2: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    quiet: bool = False,
) -> None:
    """
    Filters words from input_file (list1) that do not appear as substrings of any
    word in file2 (list2).
    Then applies length filtering using min_length and max_length.
    Optionally converts the output to lowercase, sorts it, and removes duplicates.
    Finally, writes the filtered words to output_file and prints statistics.
    """
    try:
        raw_list1, cleaned_list1, _ = _load_and_clean_file(
            input_file,
            min_length,
            max_length,
            apply_length_filter=False,
        )
        _, _, unique_list2 = _load_and_clean_file(
            file2,
            min_length,
            max_length,
            split_whitespace=True,
            apply_length_filter=False,
        )

        # Aho-Corasick automaton for efficient substring matching
        auto = ahocorasick.Automaton()
        for keyword in cleaned_list1:
            auto.add_word(keyword, keyword)
        auto.make_automaton()

        matched_words = set()
        for item in tqdm(unique_list2, desc="Finding matches", disable=quiet):
            for end_index, keyword in auto.iter(item):
                matched_words.add(keyword)

        non_matches = [word for word in cleaned_list1 if word not in matched_words]
        filtered_items = clean_and_filter(non_matches, min_length, max_length)

        if process_output:
            filtered_items = list(set(filtered_items))
            filtered_items.sort()

        with open(output_file, 'w', encoding='utf-8') as f:
            for word in filtered_items:
                f.write(word + '\n')

        print_processing_stats(len(raw_list1), filtered_items)
        logging.info(
            f"[FilterFragments Mode] Filtering complete. Results saved to '{output_file}'."
        )
    except FileNotFoundError as e:
        logging.error(f"[FilterFragments Mode] Error: File not found at '{e.filename}'")
    except Exception as e:
        logging.error(f"[FilterFragments Mode] An unexpected error occurred: {e}")


def set_operation_mode(
    input_file: str,
    file2: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    operation: str,
    quiet: bool = False,
) -> None:
    """Perform set operations (intersection, union, difference) between two files."""
    allowed_operations = {'intersection', 'union', 'difference'}
    if operation not in allowed_operations:
        raise ValueError(
            f"Invalid operation '{operation}'. Must be one of: {', '.join(sorted(allowed_operations))}."
        )

    try:
        raw_items_a, _, unique_a = _load_and_clean_file(
            input_file, min_length, max_length
        )
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

        with open(output_file, 'w', encoding='utf-8') as outfile:
            for item in result_items:
                outfile.write(item + '\n')

        print_processing_stats(len(raw_items_a) + len(raw_items_b), result_items)
        logging.info(
            f"[Set Operation Mode] Completed {operation} between '{input_file}' and "
            f"'{file2}'. Output written to '{output_file}'."
        )
    except FileNotFoundError as e:
        logging.error(f"[Set Operation Mode] Error: File not found at '{e.filename}'")
    except Exception as e:
        logging.error(f"[Set Operation Mode] An unexpected error occurred: {e}")

MODE_DETAILS = {
    "arrow": {
        "summary": "Extract text before ' -> ' from each line of a file.",
        "description": "Useful for processing conversion tables or mappings formatted as 'typo -> correction'.",
        "example": "python multitool.py arrow --input typos.log --output cleaned.txt",
    },
    "combine": {
        "summary": "Merge multiple files into one deduplicated list.",
        "description": "Reads several input files, cleans their contents, and writes a unified sorted list.",
        "example": "python multitool.py combine --input typos1.txt typos2.txt --output all_typos.txt",
    },
    "backtick": {
        "summary": "Extract text between pairs of backticks on each line.",
        "description": "Designed for compiler or linter diagnostics that enclose problematic identifiers in backticks. Includes heuristics for handling diagnostic messages with `error:`, `warning:`, and `note:` prefixes.",
        "example": "python multitool.py backtick --input build.log --output suspects.txt",
    },
    "csv": {
        "summary": "Extract typo or correction columns from a CSV file.",
        "description": "By default skips the first column so you can gather every suggested correction in one list.",
        "example": "python multitool.py csv --input typos.csv --output corrections.txt",
    },
    "line": {
        "summary": "Output each input line as-is (after optional filtering).",
        "description": "A simple pass-through mode that keeps one entry per line from the input file.",
        "example": "python multitool.py line --input raw_words.txt --output filtered.txt",
    },
    "count": {
        "summary": "Count word frequencies within the provided file.",
        "description": "Reports how often each cleaned word appears, sorted by frequency then alphabetically.",
        "example": "python multitool.py count --input typos.log --output counts.txt",
    },
    "filterfragments": {
        "summary": "Remove words that appear as substrings in a comparison file.",
        "description": "Helps filter out generated words that already exist within a dictionary or corpus.",
        "example": "python multitool.py filterfragments --input generated.txt --file2 dictionary.txt --output unique.txt",
    },
    "check": {
        "summary": "Report words that show up as both typos and corrections in a CSV file.",
        "description": "Useful sanity check for typo databases to avoid suggesting the same word as both wrong and right.",
        "example": "python multitool.py check --input typos.csv --output duplicates.txt",
    },
    "set_operation": {
        "summary": "Perform set-based comparisons between two files.",
        "description": "Supports intersection, union, and difference between the cleaned contents of the files.",
        "example": "python multitool.py set_operation --input fileA.txt --file2 fileB.txt --operation intersection --output shared.txt",
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
        modes_to_show = MODE_DETAILS.keys() if values in (None, "all") else [values]
        help_blocks = []
        for mode in modes_to_show:
            details = MODE_DETAILS.get(mode)
            if not details:
                continue
            block = [f"Mode: {mode}", f"  Summary: {details['summary']}"]
            if details.get("description"):
                block.append(f"  Description: {details['description']}")
            if details.get("example"):
                block.append(f"  Example: {details['example']}")
            help_blocks.append("\n".join(block))

        message = "\n\n".join(help_blocks)
        parser.exit(message=f"\n{message}\n\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multipurpose File Processing Tool",
        epilog=dedent(
            """
            Examples:
              python multitool.py --mode-help             # Show a summary of every mode
              python multitool.py --mode-help csv         # Describe the CSV extraction mode
              python multitool.py arrow --input file.txt  # Run a specific mode
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

    combine_parser = subparsers.add_parser(
        'combine',
        help=MODE_DETAILS['combine']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['combine']['description'],
    )
    combine_parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        required=True,
        help='Paths to the input files to merge.',
    )
    combine_parser.add_argument(
        '--output',
        type=str,
        default='output.txt',
        help="Path to the output file (default: output.txt)",
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

    return parser


def main() -> None:
    parser = _build_parser()

    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.min_length < 1:
        logging.error("[Error] --min-length must be a positive integer.")
        sys.exit(1)
    if args.max_length < args.min_length:
        logging.error("[Error] --max-length must be greater than or equal to --min-length.")
        sys.exit(1)

    logging.info(f"Selected Mode: {args.mode}")
    input_paths = args.input if isinstance(args.input, list) else [args.input]
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

    if args.mode in {'filterfragments', 'set_operation'} and file2:
        logging.info(f"File2: {file2}")
    if args.mode == 'set_operation' and operation:
        logging.info(f"Set Operation: {operation}")
    if args.mode == 'csv':
        logging.info(f"First Column Only: {'Yes' if first_column else 'No'}")
        logging.info(f"Delimiter: '{delimiter}'")

    common_kwargs = {
        'input_file': args.input,
        'output_file': args.output,
        'min_length': args.min_length,
        'max_length': args.max_length,
        'process_output': getattr(args, 'process_output', False),
        'quiet': args.quiet,
    }

    handler_map = {
        'arrow': (arrow_mode, dict(common_kwargs)),
        'backtick': (backtick_mode, dict(common_kwargs)),
        'csv': (
            csv_mode,
            {
                **common_kwargs,
                'first_column': first_column,
                'delimiter': delimiter,
            },
        ),
        'line': (line_mode, dict(common_kwargs)),
        'count': (count_mode, dict(common_kwargs)),
        'filterfragments': (
            filter_fragments_mode,
            {**common_kwargs, 'file2': file2},
        ),
        'check': (check_mode, dict(common_kwargs)),
        'set_operation': (
            set_operation_mode,
            {**common_kwargs, 'file2': file2, 'operation': operation},
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
            },
        ),
    }

    handler, handler_args = handler_map[args.mode]
    handler(**handler_args)


if __name__ == "__main__":
    main()
