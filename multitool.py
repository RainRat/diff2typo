import argparse
import csv
from collections import Counter
import sys

def arrow_mode(input_file, output_file, min_length, max_length, process_output):
    """
    Processes each line by extracting text before " -> " if present.
    Only strings with length between min_length and max_length are kept.
    Optionally converts to lowercase, sorts, and dedupes the output.
    """
    try:
        extracted_strings = []
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                index = line.find(" -> ")
                if index != -1:
                    extracted_string = line[:index].strip()
                    if len(extracted_string) >= min_length and len(extracted_string) <= max_length:
                        extracted_strings.append(extracted_string)
                else:
                    continue
        
        if process_output:
            extracted_strings = [s.lower() for s in extracted_strings]
            extracted_strings = list(set(extracted_strings))
            extracted_strings.sort()
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for string in extracted_strings:
                outfile.write(string + '\n')
        
        print(f"[Arrow Mode] File processed successfully. Output written to '{output_file}'.")
    except Exception as e:
        print(f"[Arrow Mode] An error occurred: {e}")

def backtick_mode(input_file, output_file, min_length, max_length, process_output):
    """
    Extracts strings between the first two backticks (`) in each line.
    Only strings with length between min_length and max_length are kept.
    Optionally converts to lowercase, sorts, and dedupes the output.
    """
    try:
        extracted_strings = []
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                start_index = line.find('`')
                end_index = line.find('`', start_index + 1) if start_index != -1 else -1

                if start_index != -1 and end_index != -1:
                    extracted_string = line[start_index + 1:end_index].strip()
                    if len(extracted_string) >= min_length and len(extracted_string) <= max_length:
                        extracted_strings.append(extracted_string)
        
        if process_output:
            extracted_strings = [s.lower() for s in extracted_strings]
            extracted_strings = list(set(extracted_strings))
            extracted_strings.sort()
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for string in extracted_strings:
                outfile.write(string + '\n')
        
        print(f"[Backtick Mode] Strings extracted successfully. Output written to '{output_file}'.")
    except Exception as e:
        print(f"[Backtick Mode] An error occurred: {e}")

def count_mode(input_file, output_file, min_length, process_output):
    """
    Counts the frequency of each word in the input file and writes the
    sorted results to the output file. Only words with length >= min_length are counted.
    Note: process_output is ignored in count mode.
    """
    try:
        word_counts = Counter()
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                words = [word.strip().lower() for word in line.split()]
                filtered_words = [word for word in words if (len(word) >= min_length and len(word) <= max_length)]
                word_counts.update(filtered_words)

        sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))

        with open(output_file, 'w', encoding='utf-8') as out_file:
            for word, count in sorted_words:
                out_file.write(f"{word}: {count}\n")

        print(f"[Count Mode] Word frequencies have been written to '{output_file}'.")
    except Exception as e:
        print(f"[Count Mode] An error occurred: {e}")

def csv_mode(input_file, output_file, min_length, max_length, process_output):
    """
    Reads a CSV file and writes fields from the second column onward for
    each row to the output file. Only fields with length between min_length and max_length are kept.
    Optionally converts to lowercase, sorts, and dedupes the output.
    """
    try:
        extracted_fields = []
        with open(input_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 2:
                    for field in row[1:]:
                        field = field.strip()
                        if len(field) >= min_length and len(field) <= max_length:
                            extracted_fields.append(field)
        
        if process_output:
            extracted_fields = [f.lower() for f in extracted_fields]
            extracted_fields = list(set(extracted_fields))
            extracted_fields.sort()
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for field in extracted_fields:
                outfile.write(field + '\n')
        
        print(f"[CSV Mode] CSV fields extracted successfully. Output written to '{output_file}'.")
    except Exception as e:
        print(f"[CSV Mode] An error occurred: {e}")

def line_mode(input_file, output_file, min_length, max_length, process_output):
    """
    Processes each line as-is, outputting only those lines whose length is
    between min_length and max_length.
    Optionally converts to lowercase, sorts, and dedupes the output.
    """
    try:
        lines_to_output = []
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                # Remove the trailing newline character.
                line = line.rstrip('\n')
                if len(line) >= min_length and len(line) <= max_length:
                    lines_to_output.append(line)
        
        if process_output:
            lines_to_output = [s.lower() for s in lines_to_output]
            lines_to_output = list(set(lines_to_output))
            lines_to_output.sort()
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in lines_to_output:
                outfile.write(line + '\n')
        
        print(f"[Line Mode] Lines processed successfully. Output written to '{output_file}'.")
    except Exception as e:
        print(f"[Line Mode] An error occurred: {e}")

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
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['arrow', 'backtick', 'count', 'csv', 'line'],
        help="Mode of operation:\n"
             "  arrow    - Extract text before ' -> '\n"
             "  backtick - Extract strings between backticks\n"
             "  count    - Count word frequencies\n"
             "  csv      - Extract fields from CSV\n"
             "  line     - Output each line as-is, subject to length filtering"
    )

    parser.add_argument(
        '--input',
        type=str,
        default='input.txt',
        help="Path to the input file (default: input.txt)"
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

    print(f"Selected Mode: {selected_mode}")
    print(f"Input File: {input_file}")
    print(f"Output File: {output_file}")
    print(f"Minimum String Length: {min_length}")
    print(f"Maximum String Length: {max_length}")
    if selected_mode != 'count':
        print(f"Process Output: {'Enabled' if process_output else 'Disabled'}")

    # Define a dictionary mapping modes to their functions.
    mode_functions = {
        'arrow': arrow_mode,
        'backtick': backtick_mode,
        'csv': csv_mode,
        'line': line_mode
    }

    if selected_mode == 'count':
        # For count mode, max_length and process_output are ignored.
        count_mode(input_file, output_file, min_length, process_output)
    else:
        mode_functions[selected_mode](input_file, output_file, min_length, max_length, process_output)

if __name__ == "__main__":
    main()
