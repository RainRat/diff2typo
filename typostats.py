from collections import defaultdict
import json
import sys
import logging

def is_one_letter_replacement(typo, correction, allow_two_char=False):
    """
    Check if 'typo' differs from 'correction' by exactly one "letter replacement".
    If allow_two_char is True, also check if 'typo' can be formed by replacing a single
    character in 'correction' with two characters in 'typo'.

    Returns:
      (correct_char, typo_char_or_chars) if such a single replacement (one-to-one or one-to-two) is found,
      otherwise None.

    correct_char: a single letter from the correction (correct spelling)
    typo_char_or_chars: one or two letters from the typo (incorrect spelling)
    """

    # Same length scenario: one-to-one replacement
    if len(typo) == len(correction):
        differences = []
        for t_char, c_char in zip(typo, correction):
            if t_char != c_char:
                # t_char is from typo, c_char is from correction
                # We want to store (correct_char, typo_char)
                differences.append((c_char, t_char))
                if len(differences) > 1:
                    return None  # More than one difference

        if len(differences) == 1:
            return differences[0]
        return None

    # One-to-two replacement scenario allowed only if difference in length is 1
    if allow_two_char and len(typo) == len(correction) + 1:
        # We want to find a position i where:
        # correction[i] is replaced by two chars in typo at position i and i+1.
        # i.e., correction[i] -> typo[i:i+2]
        # and the rest matches appropriately.
        for i in range(len(correction)):
            # Verify matching up to i
            if typo[:i] != correction[:i]:
                break
            # Now check if we can replace correction[i] with typo[i:i+2]
            # Then correction[i+1:] must match typo[i+2:]
            if i+1 <= len(correction) and typo[i:i+2] and correction[i+1:] == typo[i+2:]:
                # correction[i] replaced by typo[i:i+2]
                return (correction[i], typo[i:i+2])

    return None

def process_typos(lines, allow_two_char):
    replacement_counts = defaultdict(int)
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split(',')
        typo = parts[0].strip()
        corrections = [corr.strip() for corr in parts[1:]]

        # Filter out non-ASCII words
        if not all(ord(c) < 128 for c in typo):
            continue

        for correction in corrections:
            if not all(ord(c) < 128 for c in correction):
                continue
            # Now we have: `typo` (incorrect word), `correction` (correct word)
            # Check replacements
            replacement = is_one_letter_replacement(typo, correction, allow_two_char=allow_two_char)
            if replacement:
                # replacement is (correct_char, typo_char)
                replacement_counts[replacement] += 1
    return replacement_counts


def generate_report(replacement_counts, output_file=None, min_occurrences=1, sort_by='count', output_format='arrow'):
    """
    Generate a report.

    If output_format='yaml', print in specified YAML-like format:
    <correct_char>:
      - <typo_char_or_chars>
      - ...

    If output_format='json', emit a machine-readable document with the schema:
    {
        "replacements": [
            {"correct": "<correct>", "typo": "<typo>", "count": <count>},
            ...
        ]
    }
    """
    # Filter
    filtered = {k: v for k,v in replacement_counts.items() if v >= min_occurrences}

    # Sort
    if sort_by == 'count':
        sorted_replacements = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    elif sort_by == 'typo':
        # k is (correct_char, typo_char), sort by typo_char then correct_char
        sorted_replacements = sorted(filtered.items(), key=lambda x: (x[0][1], x[0][0]))
    elif sort_by == 'correct':
        # sort by correct_char then typo_char
        sorted_replacements = sorted(filtered.items(), key=lambda x: (x[0][0], x[0][1]))
    else:
        logging.warning(f"Invalid sort option: '{sort_by}'. Defaulting to 'count'.")
        sorted_replacements = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

    if output_format == 'arrow':
        # arrow
        report_lines = ["Most Frequent Letter Replacements in Typos:\n"]
        for (correct_char, typo_char), count in sorted_replacements:
            report_lines.append(f"{correct_char} -> {typo_char}: {count}")
        report_content = "\n".join(report_lines)
    elif output_format == 'json':
        replacements = [
            {
                "correct": correct_char,
                "typo": typo_char,
                "count": count,
            }
            for (correct_char, typo_char), count in sorted_replacements
        ]
        report_content = json.dumps({"replacements": replacements}, indent=2)
    else:
        # YAML-like
        # Group by correct_char
        grouping = defaultdict(set)
        for (correct_char, typo_char), count in sorted_replacements:
            grouping[correct_char].add(typo_char)

        lines = []
        for correct_char in sorted(grouping.keys()):
            lines.append(f"  {correct_char}:")
            for t_char in sorted(grouping[correct_char]):
                lines.append(f'  - "{t_char}"')
        report_content = "\n".join(lines)

    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logging.info(f"Report successfully written to '{output_file}'.")
        except Exception as e:
            logging.error(f"Failed to write report to '{output_file}'. Error: {e}")
    else:
        print(report_content)


def detect_encoding(file_path):
    """
    Attempts to detect the encoding of the given file using chardet.
    """
    try:
        import chardet
    except ImportError:
        logging.warning("chardet not installed. Install via 'pip install chardet'.")
        return None

    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        if encoding and confidence > 0.5:
            logging.info(f"Detected encoding: {encoding} (confidence {confidence:.2f})")
            return encoding
        else:
            logging.warning("Failed to reliably detect encoding.")
            return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze typo corrections and report frequent replacements.")
    parser.add_argument('input_file', help="Path to the input file.")
    parser.add_argument('-o', '--output', help="Path to the output file (optional).")
    parser.add_argument('-m', '--min', type=int, default=1, help="Minimum occurrences.")
    parser.add_argument('-s', '--sort', choices=['count', 'typo', 'correct'], default='count', help="Sorting criterion.")
    parser.add_argument(
        '-f',
        '--format',
        choices=['arrow', 'yaml', 'json'],
        default='arrow',
        help=(
            "Output format. 'json' emits {\"replacements\": [{\"correct\", \"typo\", \"count\"}, ...]}"
        ),
    )
    parser.add_argument('-2', '--allow_two_char', action='store_true', help="Allow one-to-two letter replacements.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    file_path = args.input_file
    output_file = args.output
    min_occurrences = args.min
    sort_by = args.sort
    output_format = args.format
    allow_two_char = args.allow_two_char

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        logging.warning("UTF-8 decoding failed. Trying latin1...")
        try:
            with open(file_path, 'r', encoding='latin1') as f:
                lines = f.readlines()
        except UnicodeDecodeError as e:
            logging.warning("latin1 decoding also failed.")
            enc = detect_encoding(file_path)
            if enc:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        lines = f.readlines()
                except UnicodeDecodeError as e2:
                    logging.error(f"Failed with detected encoding {enc}.")
                    sys.exit(1)
            else:
                sys.exit(1)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        sys.exit(1)

    counts = process_typos(lines, allow_two_char=allow_two_char)
    generate_report(counts, output_file=output_file, min_occurrences=min_occurrences, sort_by=sort_by, output_format=output_format)


if __name__ == "__main__":
    main()
