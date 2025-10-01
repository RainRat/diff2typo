import argparse
import csv
from collections import Counter
import sys
import re
from tqdm import tqdm


def filter_to_letters(text):
    """Return text containing only lowercase a-z characters."""
    return re.sub("[^a-z]", "", text.lower())


def clean_and_filter(items, min_length, max_length):
    """Clean items to letters only and apply length filtering."""
    cleaned = [filter_to_letters(item) for item in items]
    return [c for c in cleaned if min_length <= len(c) <= max_length]


def print_processing_stats(raw_item_count, filtered_items, item_label="item"):
    """Print summary statistics for processed text items."""
    item_label_plural = f"{item_label}s"
    print("Statistics:")
    print(f"  Total {item_label_plural} encountered: {raw_item_count}")
    print(f"  Total {item_label_plural} after filtering: {len(filtered_items)}")
    if filtered_items:
        unique_items = list(dict.fromkeys(filtered_items))
        shortest = min(unique_items, key=len)
        longest = max(unique_items, key=len)
        print(f"  Shortest {item_label}: '{shortest}' (length: {len(shortest)})")
        print(f"  Longest {item_label}: '{longest}' (length: {len(longest)})")
    else:
        print(f"  No {item_label_plural} passed the filtering criteria.")


def _process_items(extractor_func, input_file, output_file, min_length, max_length, process_output, mode_name, success_msg):
    """Generic processing for modes that extract raw string items from a file."""
    try:
        raw_items = list(extractor_func(input_file))
        filtered_items = clean_and_filter(raw_items, min_length, max_length)
        if process_output:
            filtered_items = sorted(set(filtered_items))
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for item in filtered_items:
                outfile.write(item + '\n')
        print_processing_stats(len(raw_items), filtered_items)
        print(f"[{mode_name} Mode] {success_msg} Output written to '{output_file}'.")
    except Exception as e:
        print(f"[{mode_name} Mode] An error occurred: {e}")


def _extract_arrow_items(input_file):
    """Yield text before ' -> ' from each line."""
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, desc='Processing lines (arrow mode)', unit=' lines'):
            if " -> " in line:
                yield line.split(" -> ", 1)[0].strip()


def _extract_backtick_items(input_file):
    """Yield text found between backticks with heuristics for diagnostics."""

    context_markers = ("error:", "warning:", "note:")

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, desc='Processing lines (backtick mode)', unit=' lines'):
            # Split the line on backticks to inspect the surrounding context of
            # each candidate substring. This helps avoid extracting identifiers
            # from file paths when a later pair of backticks contains the actual
            # typo from messages such as "error: `foo` should be `bar`".
            parts = line.split('`')
            selected = None
            if len(parts) >= 3:
                for index in range(1, len(parts)):
                    preceding = parts[index - 1].lower() if index - 1 >= 0 else ""
                    if "error:" in preceding:
                        selected = parts[index].strip()
                        break

            if selected is None:
                start_index = line.find('`')
                end_index = line.find('`', start_index + 1) if start_index != -1 else -1
                if start_index != -1 and end_index != -1:
                    selected = line[start_index + 1:end_index].strip()

            if selected:
                yield selected


def _extract_csv_items(input_file, first_column):
    """Yield fields from CSV rows based on column selection."""
    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in tqdm(reader, desc='Processing CSV rows', unit=' rows'):
            if first_column:
                if row:
                    yield row[0].strip()
            else:
                if len(row) >= 2:
                    for field in row[1:]:
                        yield field.strip()


def _extract_line_items(input_file):
    """Yield each line from the file."""
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, desc='Processing lines (line mode)', unit=' lines'):
            yield line.rstrip('\n')


def arrow_mode(input_file, output_file, min_length, max_length, process_output):
    """Wrapper for processing items separated by ' -> '."""
    _process_items(_extract_arrow_items, input_file, output_file, min_length, max_length, process_output, 'Arrow', 'File processed successfully.')


def backtick_mode(input_file, output_file, min_length, max_length, process_output):
    """Wrapper for extracting text between backticks."""
    _process_items(_extract_backtick_items, input_file, output_file, min_length, max_length, process_output, 'Backtick', 'Strings extracted successfully.')

def count_mode(input_file, output_file, min_length, max_length, process_output):
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
            for line in tqdm(file, desc='Counting words', unit=' lines'):
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
        print(f"[Count Mode] Word frequencies have been written to '{output_file}'.")
    except Exception as e:
        print(f"[Count Mode] An error occurred: {e}")

def check_mode(input_file, output_file, min_length, max_length, process_output):
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
            for row in tqdm(reader, desc='Checking CSV for overlaps', unit=' rows'):
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
        print(f"[Check Mode] Found {len(filtered_items)} overlapping words. Output written to '{output_file}'.")
    except Exception as e:
        print(f"[Check Mode] An error occurred: {e}")


def csv_mode(input_file, output_file, min_length, max_length, process_output, first_column=False):
    """Wrapper for extracting fields from CSV files."""
    extractor = lambda f: _extract_csv_items(f, first_column)
    _process_items(extractor, input_file, output_file, min_length, max_length, process_output, 'CSV', 'CSV fields extracted successfully.')


def line_mode(input_file, output_file, min_length, max_length, process_output):
    """Wrapper for processing raw lines from a file."""
    _process_items(_extract_line_items, input_file, output_file, min_length, max_length, process_output, 'Line', 'Lines processed successfully.')

def _add_common_mode_arguments(subparser, include_process_output=True):
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


def filter_fragments_mode(input_file, file2, output_file, min_length, max_length, process_output):
    """
    Filters words from input_file (list1) that do not appear as substrings of any
    word in file2 (list2).
    Then applies length filtering using min_length and max_length.
    Optionally converts the output to lowercase, sorts it, and removes duplicates.
    Finally, writes the filtered words to output_file and prints statistics.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            list1 = [filter_to_letters(line.strip()) for line in f]

        comparison_words = []
        with open(file2, 'r', encoding='utf-8') as f:
            for line in f:
                for raw_word in line.split():
                    cleaned = filter_to_letters(raw_word)
                    if cleaned:
                        comparison_words.append(cleaned)

        # Remove duplicates to reduce redundant substring checks while keeping
        # all unique words for comparison.
        comparison_words = list(dict.fromkeys(comparison_words))

        non_matches = []
        for word in tqdm(list1, desc='Filtering words (substring match)', unit=' word'):
            if not word:
                continue
            if any(word in comparison for comparison in comparison_words if len(comparison) >= len(word)):
                continue
            non_matches.append(word)

        filtered_items = clean_and_filter(non_matches, min_length, max_length)

        if process_output:
            filtered_items = list(set(filtered_items))
            filtered_items.sort()

        with open(output_file, 'w', encoding='utf-8') as f:
            for word in filtered_items:
                f.write(word + '\n')

        print_processing_stats(len(list1), filtered_items)
        print(f"[FilterFragments Mode] Filtering complete. Results saved to '{output_file}'.")
    except Exception as e:
        print(f"[FilterFragments Mode] An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Multipurpose File Processing Tool",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='mode', required=True, metavar='mode')

    arrow_parser = subparsers.add_parser(
        'arrow',
        help="Extract text before ' -> ' from each line.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _add_common_mode_arguments(arrow_parser)

    backtick_parser = subparsers.add_parser(
        'backtick',
        help='Extract text between the first pair of backticks on each line.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _add_common_mode_arguments(backtick_parser)

    csv_parser = subparsers.add_parser(
        'csv',
        help='Extract fields from a CSV file.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _add_common_mode_arguments(csv_parser)
    csv_parser.add_argument(
        '--first-column',
        action='store_true',
        help='Extract the first column instead of subsequent columns.',
    )

    line_parser = subparsers.add_parser(
        'line',
        help='Output each line as-is, subject to filtering.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _add_common_mode_arguments(line_parser)

    count_parser = subparsers.add_parser(
        'count',
        help='Count word frequencies.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _add_common_mode_arguments(count_parser, include_process_output=False)

    filter_parser = subparsers.add_parser(
        'filterfragments',
        help='Filter words that also appear in another file.',
        formatter_class=argparse.RawTextHelpFormatter,
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
        help='Report words that are both typos and corrections in a CSV.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _add_common_mode_arguments(check_parser)

    args = parser.parse_args()

    if hasattr(args, 'min_length') and args.min_length < 1:
        print("[Error] --min-length must be a positive integer.")
        sys.exit(1)
    if hasattr(args, 'max_length') and args.max_length < args.min_length:
        print("[Error] --max-length must be greater than or equal to --min-length.")
        sys.exit(1)

    print(f"Selected Mode: {args.mode}")
    if hasattr(args, 'input'):
        print(f"Input File: {args.input}")
    if hasattr(args, 'output'):
        print(f"Output File: {args.output}")

    if hasattr(args, 'min_length'):
        print(f"Minimum String Length: {args.min_length}")
        print(f"Maximum String Length: {args.max_length}")

    if args.mode != 'count' and hasattr(args, 'process_output'):
        print(f"Process Output: {'Enabled' if args.process_output else 'Disabled'}")

    if args.mode == 'filterfragments':
        print(f"File2: {args.file2}")
    if args.mode == 'csv':
        print(f"First Column Only: {'Yes' if args.first_column else 'No'}")

    if args.mode == 'arrow':
        arrow_mode(args.input, args.output, args.min_length, args.max_length, args.process_output)
    elif args.mode == 'backtick':
        backtick_mode(args.input, args.output, args.min_length, args.max_length, args.process_output)
    elif args.mode == 'csv':
        csv_mode(args.input, args.output, args.min_length, args.max_length, args.process_output, args.first_column)
    elif args.mode == 'line':
        line_mode(args.input, args.output, args.min_length, args.max_length, args.process_output)
    elif args.mode == 'count':
        count_mode(args.input, args.output, args.min_length, args.max_length, args.process_output)
    elif args.mode == 'filterfragments':
        filter_fragments_mode(args.input, args.file2, args.output, args.min_length, args.max_length, args.process_output)
    elif args.mode == 'check':
        check_mode(args.input, args.output, args.min_length, args.max_length, args.process_output)


if __name__ == "__main__":
    main()
