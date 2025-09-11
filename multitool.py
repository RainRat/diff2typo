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

def print_stats(raw_items, filtered_items):
    print("Statistics:")
    print(f"  Total items encountered: {len(raw_items)}")
    print(f"  Total items after filtering: {len(filtered_items)}")
    if filtered_items:
        shortest = min(filtered_items, key=len)
        longest = max(filtered_items, key=len)
        print(f"  Shortest item: '{shortest}' (length: {len(shortest)})")
        print(f"  Longest item: '{longest}' (length: {len(longest)})")
    else:
        print("  No items passed the filtering criteria.")

def print_stats_count(raw_count, filtered_words):
    print("Statistics:")
    print(f"  Total words encountered: {raw_count}")
    print(f"  Total words after filtering: {len(filtered_words)}")
    if filtered_words:
        unique_words = set(filtered_words)
        shortest = min(unique_words, key=len)
        longest = max(unique_words, key=len)
        print(f"  Shortest word: '{shortest}' (length: {len(shortest)})")
        print(f"  Longest word: '{longest}' (length: {len(longest)})")
    else:
        print("  No words passed the filtering criteria.")

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
        print_stats(raw_items, filtered_items)
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
    """Yield text found between the first two backticks in each line."""
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, desc='Processing lines (backtick mode)', unit=' lines'):
            start_index = line.find('`')
            end_index = line.find('`', start_index + 1) if start_index != -1 else -1
            if start_index != -1 and end_index != -1:
                yield line[start_index + 1:end_index].strip()


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
        print_stats_count(raw_count, filtered_words)
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
        print_stats(duplicates, filtered_items)
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

def filter_fragments_mode(input_file, file2, output_file, min_length, max_length, process_output):
    """
    Filters words from input_file (list1) that are NOT substrings of any word in file2 (list2).
    Then applies length filtering using min_length and max_length.
    Optionally converts the output to lowercase, sorts it, and removes duplicates.
    Finally, writes the filtered words to output_file and prints statistics.
    """
    try:
        # Read words from input_file (list1)
        with open(input_file, 'r') as f:
            list1 = [filter_to_letters(line.strip()) for line in f]

        # Read words from file2 (list2) as a single string to allow
        # efficient substring checks. Reading the entire file and
        # performing one `in` lookup per word is drastically faster
        # than checking against each line individually.
        with open(file2, 'r', encoding='utf-8') as f:
            list2_content = f.read()

        # Filter words from list1 that are NOT substrings of the content of file2
        non_substrings = [
            word for word in tqdm(list1, desc='Filtering words (not substrings)', unit=' word')
            if word and word not in list2_content
        ]

        # Further filter by length and clean
        filtered_items = clean_and_filter(non_substrings, min_length, max_length)

        # Optionally process the output: dedupe and sort.
        if process_output:
            filtered_items = list(set(filtered_items))
            filtered_items.sort()

        # Write the filtered words to the output file
        with open(output_file, 'w') as f:
            for word in filtered_items:
                f.write(word + '\n')

        # Print statistics based on the original list1 and the final filtered list.
        print_stats(list1, filtered_items)
        print(f"[FilterFragments Mode] Filtering complete. Results saved to '{output_file}'.")
    except Exception as e:
        print(f"[FilterFragments Mode] An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Multipurpose File Processing Tool",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Example usage:
  python multitool.py --mode arrow --input input1.txt --output output1.txt
  python multitool.py --mode backtick --input input2.txt --output output2.txt
  python multitool.py --mode count --input input3.txt --output output3.txt
  python multitool.py --mode csv --input input4.csv --output output4.txt
  python multitool.py --mode line --input input5.txt --output output5.txt --min-length 5 --max-length 50 --process-output
  python multitool.py --mode filterfragments --input list1.txt --file2 list2.txt --output filtered.txt --min-length 3 --max-length 20 --process-output
  python multitool.py --mode check --input typos.csv --output duplicates.txt
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['arrow', 'backtick', 'count', 'csv', 'line', 'filterfragments', 'check'],
        help="Mode of operation:\n"
             "  arrow           - Extract text before ' -> '\n"
             "  backtick        - Extract strings between backticks\n"
             "  count           - Count word frequencies\n"
             "  csv             - Extract fields from CSV\n"
             "  line            - Output each line as-is, subject to length filtering\n"
             "  filterfragments - Filter words from file1 that are not substrings of any word in file2\n"
             "  check           - List words appearing as both typo and correction in a CSV"
    )

    parser.add_argument(
        '--input',
        type=str,
        default='input.txt',
        help="Path to the input file (default: input.txt)"
    )

    parser.add_argument(
        '--file2',
        type=str,
        help="Path to the second file (required for filterfragments mode)"
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output.txt',
        help="Path to the output file (default: output.txt)"
    )

    parser.add_argument(
        '--min-length',
        type=int,
        default=3,
        help="Minimum string length to process (default: 3)"
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=1000,
        help="Maximum string length to process (default: 1000)"
    )

    parser.add_argument(
        '--process-output',
        action='store_true',
        help="If set, converts output to lowercase, sorts it, and removes duplicates (applicable to all modes except 'count')."
    )

    parser.add_argument(
        '--first-column',
        action='store_true',
        help="In csv mode, extract the first column instead of the second onward."
    )

    args = parser.parse_args()

    min_length = args.min_length
    max_length = args.max_length

    if min_length < 1:
        print("[Error] --min-length must be a positive integer.")
        sys.exit(1)
    if max_length < min_length:
        print("[Error] --max-length must be greater than or equal to --min-length.")
        sys.exit(1)

    selected_mode = args.mode
    input_file = args.input
    output_file = args.output
    process_output = args.process_output
    first_column = args.first_column

    print(f"Selected Mode: {selected_mode}")
    print(f"Input File: {input_file}")
    print(f"Output File: {output_file}")

    if selected_mode in ['arrow', 'backtick', 'csv', 'line', 'count', 'filterfragments', 'check']:
        print(f"Minimum String Length: {min_length}")
        print(f"Maximum String Length: {max_length}")
        if selected_mode not in ['count']:
            print(f"Process Output: {'Enabled' if process_output else 'Disabled'}")
    if selected_mode == 'filterfragments':
        # For filterfragments, ensure that file2 is provided.
        if not args.file2:
            print("[Error] --file2 is required for filterfragments mode.")
            sys.exit(1)
        print(f"File2: {args.file2}")

    # Dispatch modes.
    mode_functions = {
        'arrow': arrow_mode,
        'backtick': backtick_mode,
        'count': count_mode,
        'csv': csv_mode,
        'line': line_mode,
        'check': check_mode
    }

    if selected_mode == 'filterfragments':
        filter_fragments_mode(input_file, args.file2, output_file, min_length, max_length, process_output)
    else:
        if selected_mode == 'csv':
            csv_mode(input_file, output_file, min_length, max_length, process_output, first_column)
        else:
            mode_functions[selected_mode](input_file, output_file, min_length, max_length, process_output)

if __name__ == "__main__":
    main()
