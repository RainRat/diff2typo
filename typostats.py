from collections import defaultdict
import json
import sys
import logging
import csv
import io
from typing import Iterable

try:
    import chardet  # type: ignore

    _CHARDET_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    chardet = None
    _CHARDET_AVAILABLE = False

def is_one_letter_replacement(
    typo: str, correction: str, allow_two_char: bool = False
) -> list[tuple[str, str]]:
    """
    Check if 'typo' differs from 'correction' by one or more "letter replacements".

    If allow_two_char is True, also check if 'typo' can be formed by replacing a single
    character in 'correction' with two characters in 'typo'.

    Returns:
      A list of (correction_char, typo_char_or_chars) tuples for each found replacement.
      The first value comes from the expected spelling (``correction``) and the second
      value comes from the observed typo. Returns an empty list if no replacements are
      found.
    """

    # Same length scenario: one-to-one replacement
    if len(typo) == len(correction):
        differences = []
        for t_char, c_char in zip(typo, correction):
            if t_char != c_char:
                # t_char is from typo, c_char is from correction
                # We want to store (correct_char, typo_char)
                differences.append((c_char, t_char))

        if len(differences) == 1:
            return differences
        return []

    # One-to-two replacement scenario allowed only if difference in length is 1
    if allow_two_char and len(typo) == len(correction) + 1:
        # First, check for the specific case of a character being doubled, as this is a common typo.
        for i in range(len(correction)):
            if typo == correction[:i] + correction[i] * 2 + correction[i+1:]:
                return [(correction[i], correction[i] * 2)]

        # If no doubling is found, check for a generic one-to-two replacement.
        # Find all positions i where correction[i] is replaced by typo[i:i+2].
        replacements = []
        for i in range(len(correction)):
            # To be a replacement of correction[i] with typo[i:i+2],
            # the prefix correction[:i] must match typo[:i], and
            # the suffix correction[i+1:] must match typo[i+2:].
            if typo[:i] == correction[:i] and typo[i+2:] == correction[i+1:]:
                replacements.append((correction[i], typo[i:i+2]))
        return replacements

    return []

def process_typos(lines: Iterable[str], allow_two_char: bool) -> dict[tuple[str, str], int]:
    replacement_counts = defaultdict(int)
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if " -> " in line:
            parts = line.split(" -> ", 1)
            typo = parts[0].strip()
            # Arrow format usually implies single correction per line: typo -> correction
            corrections = [parts[1].strip()]
        else:
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
            replacements = is_one_letter_replacement(typo, correction, allow_two_char=allow_two_char)
            for replacement in replacements:
                # replacement is (correct_char, typo_char)
                replacement_counts[replacement] += 1
    return replacement_counts


def generate_report(
    replacement_counts: dict[tuple[str, str], int],
    output_file: str | None = None,
    min_occurrences: int = 1,
    sort_by: str = 'count',
    output_format: str = 'arrow',
) -> None:
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
    elif output_format == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['correct_char', 'typo_char', 'count'])
        for (correct_char, typo_char), count in sorted_replacements:
            writer.writerow([correct_char, typo_char, count])
        report_content = output.getvalue()
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
        sys.stdout.write(report_content)
        logging.info("%s", report_content.rstrip("\n"))


def detect_encoding(file_path: str) -> str | None:
    """
    Attempts to detect the encoding of the given file using chardet.
    """
    if not _CHARDET_AVAILABLE:
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


def load_lines_from_file(file_path: str) -> list[str] | None:
    """
    Loads lines from a file, attempting to detect the encoding if UTF-8 fails.
    Falls back to Latin-1 if detection fails or is unavailable.
    """
    if file_path == '-':
        logging.info("Reading from stdin...")
        return sys.stdin.readlines()

    lines = None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        logging.warning(f"UTF-8 decoding failed for {file_path}. Attempting detection...")
        lines = None

        # Try to detect encoding
        enc = detect_encoding(file_path)
        if enc:
            try:
                logging.info(f"Using detected encoding: {enc}")
                with open(file_path, 'r', encoding=enc) as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                logging.warning(f"Detected encoding {enc} failed.")

        # Fallback to latin1 if detection failed or wasn't possible
        if lines is None:
            logging.warning("Fallback to latin1...")
            try:
                with open(file_path, 'r', encoding='latin1') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                # Should practically never happen for latin1
                logging.error("Final fallback to latin1 failed.")
                return None
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None

    return lines


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Find common patterns in your typos. This tool analyzes your list of corrections and tells you which keys you hit by mistake most often."
    )
    parser.add_argument(
        'input_files',
        nargs='*',
        help="One or more files containing typo corrections. If empty, it reads from standard input.",
    )
    parser.add_argument('-o', '--output', help="Save the report to this file instead of printing it.")
    parser.add_argument('-m', '--min', type=int, default=1, help="Only show patterns that appear at least this many times.")
    parser.add_argument(
        '-s', '--sort',
        choices=['count', 'typo', 'correct'],
        default='count',
        help="How to sort the results: 'count' (most frequent first), 'typo' (alphabetical by typo), or 'correct' (alphabetical by fix)."
    )
    parser.add_argument(
        '-f',
        '--format',
        choices=['arrow', 'yaml', 'json', 'csv'],
        default='arrow',
        help="The format of the report."
    )
    parser.add_argument(
        '-2', '--allow_two_char',
        action='store_true',
        help="Look for cases where one letter is replaced by two (like 'm' becoming 'rn')."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    input_files = args.input_files
    output_file = args.output
    min_occurrences = args.min
    sort_by = args.sort
    output_format = args.format
    allow_two_char = args.allow_two_char

    if not input_files:
        input_files = ['-']

    all_counts = defaultdict(int)

    for file_path in input_files:
        lines = load_lines_from_file(file_path)

        if lines:
            file_counts = process_typos(lines, allow_two_char=allow_two_char)
            for k, v in file_counts.items():
                all_counts[k] += v

    generate_report(all_counts, output_file=output_file, min_occurrences=min_occurrences, sort_by=sort_by, output_format=output_format)


if __name__ == "__main__":
    main()
