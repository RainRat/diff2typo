import argparse
import csv
from collections import Counter
import sys
from tqdm import tqdm

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

def arrow_mode(input_file, output_file, min_length, max_length, process_output):
    """
    Processes each line by extracting text before " -> " if present.
    Gathers raw items, then filters them based on length.
    Optionally converts to lowercase, sorts, and dedupes the output.
    """
    try:
        raw_items = []
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in tqdm(infile, desc='Processing lines (arrow mode)', unit=' lines'):
                if " -> " in line:
                    extracted = line.split(" -> ", 1)[0].strip()
                    raw_items.append(extracted)
        filtered_items = [item for item in raw_items if min_length <= len(item) <= max_length]
        if process_output:
            filtered_items = [s.lower() for s in filtered_items]
            filtered_items = list(set(filtered_items))
            filtered_items.sort()
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for item in filtered_items:
                outfile.write(item + '\n')
        print_stats(raw_items, filtered_items)
        print(f"[Arrow Mode] File processed successfully. Output written to '{output_file}'.")
    except Exception as e:
        print(f"[Arrow Mode] An error occurred: {e}")

def backtick_mode(input_file, output_file, min_length, max_length, process_output):
    """
    Extracts strings between the first two backticks (`) in each line.
    Gathers raw items, then filters them based on length.
    Optionally converts to lowercase, sorts, and dedupes the output.
    """
    try:
        raw_items = []
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in tqdm(infile, desc='Processing lines (backtick mode)', unit=' lines'):
                start_index = line.find('`')
                end_index = line.find('`', start_index + 1) if start_index != -1 else -1
                if start_index != -1 and end_index != -1:
                    extracted = line[start_index + 1:end_index].strip()
                    raw_items.append(extracted)
        filtered_items = [item for item in raw_items if min_length <= len(item) <= max_length]
        if process_output:
            filtered_items = [s.lower() for s in filtered_items]
            filtered_items = list(set(filtered_items))
            filtered_items.sort()
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for item in filtered_items:
                outfile.write(item + '\n')
        print_stats(raw_items, filtered_items)
        print(f"[Backtick Mode] Strings extracted successfully. Output written to '{output_file}'.")
    except Exception as e:
        print(f"[Backtick Mode] An error occurred: {e}")

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
                filtered = [word.lower() for word in words if min_length <= len(word) <= max_length]
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

def csv_mode(input_file, output_file, min_length, max_length, process_output, first_column=False):
    """
    Reads a CSV file and writes fields from the second column onward for each
    row to the output file by default. With ``first_column`` set to ``True``,
    extracts only the first column. Gathers raw fields, then filters them based
    on length. Optionally converts to lowercase, sorts, and dedupes the output.
    """
    try:
        raw_items = []
        with open(input_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in tqdm(reader, desc='Processing CSV rows', unit=' rows'):
                if first_column:
                    if row:
                        raw_items.append(row[0].strip())
                else:
                    if len(row) >= 2:
                        for field in row[1:]:
                            raw_items.append(field.strip())
        filtered_items = [item for item in raw_items if min_length <= len(item) <= max_length]
        if process_output:
            filtered_items = [s.lower() for s in filtered_items]
            filtered_items = list(set(filtered_items))
            filtered_items.sort()
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for item in filtered_items:
                outfile.write(item + '\n')
        print_stats(raw_items, filtered_items)
        print(f"[CSV Mode] CSV fields extracted successfully. Output written to '{output_file}'.")
    except Exception as e:
        print(f"[CSV Mode] An error occurred: {e}")

def line_mode(input_file, output_file, min_length, max_length, process_output):
    """
    Processes each line as-is, gathering raw lines then filtering them based on
    length. Optionally converts to lowercase, sorts, and dedupes the output.
    """
    try:
        raw_items = []
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in tqdm(infile, desc='Processing lines (line mode)', unit=' lines'):
                raw_items.append(line.rstrip('\n'))
        filtered_items = [item for item in raw_items if min_length <= len(item) <= max_length]
        if process_output:
            filtered_items = [s.lower() for s in filtered_items]
            filtered_items = list(set(filtered_items))
            filtered_items.sort()
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for item in filtered_items:
                outfile.write(item + '\n')
        print_stats(raw_items, filtered_items)
        print(f"[Line Mode] Lines processed successfully. Output written to '{output_file}'.")
    except Exception as e:
        print(f"[Line Mode] An error occurred: {e}")

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
            list1 = [line.strip() for line in f]

        # Read words from file2 (list2)
        with open(file2, 'r') as f:
            list2 = [line.strip() for line in f]

        # Filter words from list1 that are NOT substrings of any word in list2
        non_substrings = [
            word for word in tqdm(list1, desc='Filtering words (not substrings)', unit=' word')
            if all(word not in target for target in list2)
        ]

        # Further filter by length
        filtered_items = [word for word in non_substrings if min_length <= len(word) <= max_length]

        # Optionally process the output: convert to lowercase, dedupe, and sort.
        if process_output:
            filtered_items = [s.lower() for s in filtered_items]
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
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['arrow', 'backtick', 'count', 'csv', 'line', 'filterfragments'],
        help="Mode of operation:\n"
             "  arrow           - Extract text before ' -> '\n"
             "  backtick        - Extract strings between backticks\n"
             "  count           - Count word frequencies\n"
             "  csv             - Extract fields from CSV\n"
             "  line            - Output each line as-is, subject to length filtering\n"
             "  filterfragments - Filter words from file1 that are not substrings of any word in file2"
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

    if selected_mode in ['arrow', 'backtick', 'csv', 'line', 'count', 'filterfragments']:
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
        'line': line_mode
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
