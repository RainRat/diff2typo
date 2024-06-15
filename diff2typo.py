import re
import subprocess
import argparse

def extract_backticks(input_text):
    output = []
    lines = input_text.split('\n')
    for line in lines:
        start_index = line.find('`')
        end_index = line.find('`', start_index + 1)
        if start_index != -1 and end_index != -1:
            extracted_string = line[start_index + 1:end_index]
            if len(extracted_string) > 1:
                output.append(extracted_string)
    return output

def find_typos(diff_text):
    typos = []
    lines = diff_text.split('\n')
    before_change = ''
    after_change = ''
    
    word_pattern = re.compile(r'\b\w+\b')
    
    for line in lines:
        if line.startswith('---') or line.startswith('+++'):
            continue
        if line.startswith('-'):
            before_change = line[1:].strip()
        elif line.startswith('+'):
            after_change = line[1:].strip()
            before_words = word_pattern.findall(before_change)
            after_words = word_pattern.findall(after_change)
            # Check if the lines differ by one word and filter out single-character words
            # Will only extract one typo per line but prevents getting confused by complex fixes
            if len(before_words) == len(after_words):
                diff_count = sum(1 for a, b in zip(before_words, after_words) if a != b)
                if diff_count == 1:
                    typo_word = [(a, b) for a, b in zip(before_words, after_words) if a != b][0]
                    if len(typo_word[0]) > 1 and len(typo_word[1]) > 1:
                        typos.append(f"{typo_word[0]} -> {typo_word[1]}")
            before_change = ''
            after_change = ''
    return typos

def lowercase_sort_dedup(input_list):
    lines = [line.lower() for line in input_list]
    unique_lines = list(set(lines))
    unique_lines.sort()
    return unique_lines

def main():
    # Convert `git diff` to data file for `typos` tool, to ensure typo does not happen again.
    # 1. Input diff.txt, convert to raw corrections
    # 2. Lowercase, sort, dedupe
    # 3. Save to temp file.
    # 4. Run "typos --format brief [input] > [output]"
    # 5. Extract the list of already-known typos between backticks
    # 6. Remove the already-known typos

    parser = argparse.ArgumentParser(description="Process diff file to identify typos.")
    parser.add_argument('--input_file', type=str, default='diff.txt', help='The input diff file.')
    parser.add_argument('--output_file', type=str, default='typos.txt', help='The output typos file.')
    parser.add_argument('--typos_tool_path', type=str, default='typos', help='Path to the typos tool.')
    args = parser.parse_args()

    temp_file = 'typos_temp.txt'

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            diff_text = f.read()
    except UnicodeDecodeError:
        with open(args.input_file, 'r', encoding='latin-1') as f:
            diff_text = f.read()
    
    typos = find_typos(diff_text)
    typos = lowercase_sort_dedup(typos)

    with open(temp_file, 'w', encoding='utf-8') as f:
        for typo in typos:
            f.write(f"{typo}\n")

    command = [args.typos_tool_path, '--format', 'brief', temp_file]
    result = subprocess.run(command, capture_output=True, text=True)
    already_known_typos = extract_backticks(result.stdout)
    filtered_lines = [line for line in typos if line.split(' -> ')[0] not in already_known_typos]

    with open(args.output_file, 'w') as f:
        for typo in filtered_lines:
            f.write(f"{typo}\n")

    print(f"Typos have been written to {args.output_file}")

if __name__ == "__main__":
    main()
