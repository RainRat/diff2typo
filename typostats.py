from collections import defaultdict
import json
import sys
import logging
import csv
import io
from typing import Iterable

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    _TQDM_AVAILABLE = False

try:
    import chardet  # type: ignore

    _CHARDET_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    chardet = None
    _CHARDET_AVAILABLE = False

def is_transposition(typo: str, correction: str) -> list[tuple[str, str]]:
    """
    Check if 'typo' is formed by swapping two adjacent characters in 'correction'.

    Returns:
      A list containing a single (correction_chars, typo_chars) tuple if a
      transposition is found, otherwise an empty list.
    """
    if len(typo) != len(correction):
        return []

    differences = []
    for i in range(len(typo)):
        if typo[i] != correction[i]:
            differences.append(i)

    if len(differences) == 2 and differences[1] == differences[0] + 1:
        i, j = differences
        if typo[i] == correction[j] and typo[j] == correction[i]:
            # Found a transposition
            return [(correction[i:j+1], typo[i:j+1])]

    return []


def is_one_letter_replacement(
    typo: str, correction: str, allow_two_char: bool = False
) -> list[tuple[str, str]]:
    """
    Check if 'typo' differs from 'correction' by one or more "letter replacements".

    If allow_two_char is True, also check if 'typo' can be formed by replacing a single
    character in 'correction' with two characters in 'typo', or two characters in
    'correction' with a single character in 'typo'.

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
        # Find all positions i where correction[i] is replaced by typo[i:i+2].
        # We use a set to avoid counting identical interpretations (e.g. for doubled letters) multiple times.
        replacements = set()
        for i in range(len(correction)):
            # To be a replacement of correction[i] with typo[i:i+2],
            # the prefix correction[:i] must match typo[:i], and
            # the suffix correction[i+1:] must match typo[i+2:].
            if typo[:i] == correction[:i] and typo[i+2:] == correction[i+1:]:
                replacements.add((correction[i], typo[i:i+2]))
        return sorted(list(replacements))

    # Two-to-one replacement scenario (e.g. 'ph' -> 'f')
    if allow_two_char and len(typo) == len(correction) - 1:
        replacements = set()
        for i in range(len(typo)):
            # To be a replacement of correction[i:i+2] with typo[i],
            # the prefix correction[:i] must match typo[:i], and
            # the suffix correction[i+2:] must match typo[i+1:].
            if correction[:i] == typo[:i] and correction[i+2:] == typo[i+1:]:
                replacements.add((correction[i:i+2], typo[i]))
        return sorted(list(replacements))

    return []

def process_typos(
    lines: Iterable[str], allow_two_char: bool, allow_transposition: bool = False
) -> dict[tuple[str, str], int]:
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
        elif " = " in line:
            parts = line.split(" = ", 1)
            typo = parts[0].strip()
            correction = parts[1].strip().strip('"')
            corrections = [correction]
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
            # Check for transpositions first if enabled, as they are a specific pattern
            replacements = []
            if allow_transposition:
                replacements = is_transposition(typo, correction)

            # If no transposition found, check for one-letter replacements
            if not replacements:
                replacements = is_one_letter_replacement(
                    typo, correction, allow_two_char=allow_two_char
                )

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
    limit: int | None = None,
    quiet: bool = False,
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

    Args:
        replacement_counts: Dictionary mapping (correct, typo) to frequency.
        output_file: Path to write the report to (optional).
        min_occurrences: Minimum count to include in the report.
        sort_by: Criterion to sort results by ('count', 'typo', 'correct').
        output_format: Format of the report ('arrow', 'yaml', 'json', 'csv').
        limit: Maximum number of results to include in the report.
    """
    # Filter
    filtered = {k: v for k, v in replacement_counts.items() if v >= min_occurrences}

    # Sort
    if sort_by == 'typo':
        # k is (correct_char, typo_char), sort by typo_char then correct_char
        sorted_replacements = sorted(filtered.items(), key=lambda x: (x[0][1], x[0][0]))
    elif sort_by == 'correct':
        # sort by correct_char then typo_char
        sorted_replacements = sorted(filtered.items(), key=lambda x: (x[0][0], x[0][1]))
    else:
        # Default to sort by count
        sorted_replacements = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

    if limit:
        sorted_replacements = sorted_replacements[:limit]

    # ANSI Color Codes
    GREEN = "\033[1;32m"
    RED = "\033[1;31m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Disable colors if not running in a terminal or if writing to a file
    if output_file or not sys.stdout.isatty():
        GREEN = RED = RESET = BOLD = ""

    if output_format == 'arrow':
        # arrow
        title = "Most Frequent Letter Replacements in Typos:"

        # Calculate padding for alignment (default to header labels' lengths)
        max_c = max((len(c) for (c, t), count in sorted_replacements), default=7)
        max_c = max(max_c, 7)  # 'CORRECT' is 7
        max_t = max((len(t) for (c, t), count in sorted_replacements), default=4)
        max_t = max(max_t, 4)  # 'TYPO' is 4
        max_n = max((len(str(count)) for (c, t), count in sorted_replacements), default=5)
        max_n = max(max_n, 5)  # 'COUNT' is 5

        # Header row
        header_row = f"{'CORRECT':>{max_c}}    {'TYPO':<{max_t}}   {'COUNT':>{max_n}}"
        divider = "-" * len(header_row)

        if not output_file:
            # Move the human-readable header to stderr to keep stdout clean for piping
            if not quiet:
                sys.stderr.write(f"\n {title}\n\n")
                sys.stderr.write(f" {header_row}\n")
                sys.stderr.write(f" {divider}\n")
                sys.stderr.flush()
            report_lines = []
        else:
            report_lines = [title, "", header_row, divider]

        for (correct_char, typo_char), count in sorted_replacements:
            report_lines.append(
                f" {GREEN}{correct_char:>{max_c}}{RESET} -> {RED}{typo_char:<{max_t}}{RESET} : {BOLD}{count:>{max_n}}{RESET}"
            )
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
                if not report_content.endswith('\n'):
                    f.write('\n')
            logging.info(f"Report successfully written to '{output_file}'.")
        except Exception as e:
            logging.error(f"Failed to write report to '{output_file}'. Error: {e}")
    else:
        sys.stdout.write(report_content)
        if not report_content.endswith('\n'):
            sys.stdout.write('\n')


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

    parser = argparse.ArgumentParser(description="Analyze typo corrections and report frequent replacements.")

    # Input/Output Group
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        'input_files',
        nargs='*',
        help="Path to the input file(s). Reads from stdin if empty or '-'.",
    )
    io_group.add_argument('-o', '--output', help="Path to the output file (optional).")
    io_group.add_argument(
        '-f',
        '--format',
        choices=['arrow', 'yaml', 'json', 'csv'],
        default='arrow',
        help=(
            "Output format (default: arrow). 'json' emits {\"replacements\": [{\"correct\", \"typo\", \"count\"}, ...]}"
        ),
    )
    io_group.add_argument('-q', '--quiet', action='store_true', help="Suppress informational log output.")

    # Analysis Options Group
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument('-m', '--min', type=int, default=1, help="Minimum occurrences (default: 1).")
    analysis_group.add_argument('-s', '--sort', choices=['count', 'typo', 'correct'], default='count', help="Sorting criterion (default: count).")
    analysis_group.add_argument(
        '-2',
        '--allow-two-char',
        dest='allow_two_char',
        action='store_true',
        help="Allow multi-character letter replacements (e.g., 'm' to 'rn' or 'ph' to 'f').",
    )
    # Hidden alias for backward compatibility
    parser.add_argument('--allow_two_char', action='store_true', help=argparse.SUPPRESS)

    analysis_group.add_argument(
        '-t',
        '--transposition',
        action='store_true',
        help="Detect transpositions of adjacent characters (e.g., 'teh' to 'the').",
    )
    analysis_group.add_argument(
        '-n',
        '--limit',
        type=int,
        help="Limit the report to the top N most frequent replacements.",
    )
    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    input_files = args.input_files
    output_file = args.output
    min_occurrences = args.min
    sort_by = args.sort
    output_format = args.format
    allow_two_char = args.allow_two_char
    allow_transposition = args.transposition
    limit = args.limit

    if not input_files:
        input_files = ['-']

    all_counts = defaultdict(int)

    for file_path in input_files:
        lines = load_lines_from_file(file_path)

        if lines:
            if not args.quiet and _TQDM_AVAILABLE:
                lines = tqdm(lines, desc=f"Processing {file_path}", unit="lines", leave=False)

            file_counts = process_typos(
                lines, allow_two_char=allow_two_char, allow_transposition=allow_transposition
            )
            for k, v in file_counts.items():
                all_counts[k] += v

    generate_report(
        all_counts,
        output_file=output_file,
        min_occurrences=min_occurrences,
        sort_by=sort_by,
        output_format=output_format,
        limit=limit,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
