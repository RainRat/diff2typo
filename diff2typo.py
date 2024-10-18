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

Usage:
    python diff2typo.py \
        --input_file=diff.txt \
        --output_file=typos.txt \
        --output_format=list \
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

def extract_backticks(input_text):
    """
    Extracts all backtick-enclosed strings from the input text.

    Args:
        input_text (str): The text to extract backtick-enclosed strings from.

    Returns:
        list: A list of extracted strings without the backticks.
    """
    output = []
    lines = input_text.split('\n')
    for line in lines:
        start_index = line.find('`')
        while start_index != -1:
            end_index = line.find('`', start_index + 1)
            if end_index != -1:
                extracted_string = line[start_index + 1:end_index]
                if len(extracted_string) > 1:
                    output.append(extracted_string)
                start_index = line.find('`', end_index + 1)
            else:
                break
    return output

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
            print(f"Loaded {len(allowed_words)} allowed words from '{allowed_file}'.")
        except Exception as e:
            print(f"Error reading allowed words file '{allowed_file}': {e}")
    else:
        print(f"Allowed words file '{allowed_file}' not found. Skipping allowed word filtering.")
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

def read_dictionary(dictionary_file):
    """
    Reads a dictionary file and returns a set of valid words.
    Automatically detects the format:
        - If a line has two or more words, it assumes the first is a typo and the rest are valid words.
        - If a line has one word, it treats it as a valid word.

    Args:
        dictionary_file (str): Path to the dictionary file.

    Returns:
        set: A set of valid words in lowercase.
    """
    valid_words = set()
    if not os.path.exists(dictionary_file):
        print(f"Dictionary file '{dictionary_file}' not found. Skipping dictionary filtering.")
        return valid_words

    try:
        with open(dictionary_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if len(row) >= 2:
                    # Assume the second word is the valid word
                    # Join all words after the first in case there are more than two
                    valid_word = ' '.join(row[1:]).strip().lower()
                    if valid_word:
                        valid_words.add(valid_word)
                elif len(row) == 1:
                    # Single word, treat as valid
                    valid_word = row[0].strip().lower()
                    if valid_word:
                        valid_words.add(valid_word)
        print(f"Loaded {len(valid_words)} valid words from the dictionary.")
    except Exception as e:
        print(f"Error reading dictionary file '{dictionary_file}': {e}")
        sys.exit(1)

    return valid_words

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
    
    for line in lines:
        if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
            continue
        if line.startswith('-'):
            removals.append(line[1:].strip())
        elif line.startswith('+'):
            additions.append(line[1:].strip())
        else:
            # When encountering a context line or any other line, process the collected removals and additions
            if removals and additions:
                # Ensure that the number of removals and additions are the same
                if len(removals) == len(additions):
                    for before, after in zip(removals, additions):
                        before_words = split_into_subwords(before)
                        after_words = split_into_subwords(after)
                        if len(before_words) == len(after_words):
                            for i in range(len(before_words)):
                                if before_words[i] != after_words[i]:
                                    # Ensure adjacent words are unchanged to avoid incorrect pairings
                                    if (i == 0 or before_words[i - 1] == after_words[i - 1]) and \
                                       (i == len(before_words) - 1 or before_words[i + 1] == after_words[i + 1]):
                                        if len(before_words[i]) >= min_length and len(after_words[i]) >= min_length:
                                            if not any(char.isdigit() for char in after_words[i]):
                                                typos.append(f"{before_words[i]} -> {after_words[i]}")
                else:
                    print("Warning: Number of removals and additions do not match. Skipping these changes.")
                # Reset the lists after processing
                removals = []
                additions = []

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
            formatted.append(typo)
    return formatted

def main():
    """
    Main function to orchestrate the typo extraction and filtering process.
    """
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Process a git diff to identify typos for the `typos` utility.")
    parser.add_argument('--input_file', type=str, default='diff.txt', help='Path to the input git diff file.')
    parser.add_argument('--output_file', type=str, default='typos.txt', help='Path to the output typos file.')
    parser.add_argument('--output_format', type=str, choices=['arrow', 'csv', 'table', 'list'], default='arrow',
                        help='Format of the output typos. Choices are: arrow (typo -> correction), csv (typo,correction), table (typo = "correction"), list (typo). Default is arrow.')
    parser.add_argument('--typos_tool_path', type=str, default='typos', help='Path to the typos tool executable.')
    parser.add_argument('--allowed_file', type=str, default='allowed.csv', help='CSV file with allowed words to exclude from typos.')
    parser.add_argument('--min_length', type=int, default=2, help='Minimum length of differing substrings to consider as typos.')
    parser.add_argument('--dictionary_file', type=str, default='words.csv', help='Path to the dictionary file for filtering valid words.')
    args = parser.parse_args()

    temp_file = 'typos_temp.txt'

    print("Starting typo extraction process...")

    # Read and load the dictionary
    valid_words = set()
    if args.dictionary_file:
        valid_words = read_dictionary(args.dictionary_file)

    # Read the diff file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            diff_text = f.read()
        print(f"Successfully read input diff file '{args.input_file}'.")
    except UnicodeDecodeError:
        with open(args.input_file, 'r', encoding='latin-1') as f:
            diff_text = f.read()
        print(f"Successfully read input diff file '{args.input_file}' with 'latin-1' encoding.")
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found. Exiting.")
        sys.exit(1)

    # Find typos in the diff
    print("Identifying potential typos from the diff...")
    typos = find_typos(diff_text, min_length=args.min_length)
    typos = lowercase_sort_dedup(typos)
    print(f"Identified {len(typos)} potential typos.")

    # Save typos to a temporary file for processing with the typos tool
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            for typo in typos:
                f.write(f"{typo}\n")
        print(f"Saved typos to temporary file '{temp_file}'.")
    except Exception as e:
        print(f"Error writing to temporary file '{temp_file}': {e}")
        sys.exit(1)

    # Run the typos tool to filter out already known typos
    if os.path.exists(args.typos_tool_path) or os.path.exists(f"{args.typos_tool_path}.exe"):
        # Determine the correct executable based on the operating system
        if os.name == 'nt' and not args.typos_tool_path.lower().endswith('.exe'):
            typos_executable = f"{args.typos_tool_path}.exe"
        else:
            typos_executable = args.typos_tool_path

        print(f"Running typos tool at '{typos_executable}' to filter known typos...")
        command = [typos_executable, '--format', 'brief', temp_file]
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            already_known_typos = extract_backticks(result.stdout)
            filtered_lines = [line for line in typos if line.split(' -> ')[0] not in already_known_typos]
            print(f"Filtered out {len(already_known_typos)} already-known typos.")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Typos tool returned a non-zero exit status. Skipping known typo filtering.")
            print("Standard Output:", e.stdout)
            print("Standard Error:", e.stderr)
            filtered_lines = typos
        except FileNotFoundError:
            print(f"Warning: Typos tool '{typos_executable}' not found. Skipping known typo filtering.")
            filtered_lines = typos
    else:
        print(f"Warning: Typos tool '{args.typos_tool_path}' not found. Skipping known typo filtering.")
        filtered_lines = typos

    # Read allowed words to exclude from typos
    allowed_words = read_allowed_words(args.allowed_file)

    # Filter out allowed words
    if allowed_words:
        initial_count = len(filtered_lines)
        filtered_lines = [typo for typo in filtered_lines if typo.split(' -> ')[0].lower() not in allowed_words]
        excluded_count = initial_count - len(filtered_lines)
        print(f"Excluded {excluded_count} typos based on allowed words.")
    else:
        print("No allowed words to filter.")

    # Apply dictionary filtering if valid_words is provided
    if valid_words:
        initial_count = len(filtered_lines)
        filtered_lines = [typo for typo in filtered_lines if typo.split(' -> ')[0].lower() not in valid_words]
        filtered_count = initial_count - len(filtered_lines)
        print(f"Excluded {filtered_count} typos where the 'before' word is a valid dictionary word.")

    # Format typos based on the selected output format
    print(f"Formatting typos in '{args.output_format}' format...")
    formatted_typos = format_typos(filtered_lines, args.output_format)

    # Write the final typos to the output file
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for typo in formatted_typos:
                f.write(f"{typo}\n")
        print(f"Successfully wrote {len(formatted_typos)} typos to '{args.output_file}'.")
    except Exception as e:
        print(f"Error writing to output file '{args.output_file}': {e}")
        sys.exit(1)

    print(f"Typos have been written to '{args.output_file}' in '{args.output_format}' format.")

if __name__ == "__main__":
    main()
