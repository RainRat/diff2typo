from collections import defaultdict
import json
import sys
import logging
import csv
import io
import os
import copy
from typing import Iterable, Sequence, Mapping, Set, List, Any
from types import SimpleNamespace

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


# ANSI Color Codes
BLUE = "\033[1;34m"
GREEN = "\033[1;32m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Disable colors if not running in a terminal or if NO_COLOR is set
if not sys.stdout.isatty() or os.environ.get('NO_COLOR'):
    BLUE = GREEN = RED = YELLOW = RESET = BOLD = ""


class MinimalFormatter(logging.Formatter):
    """A logging formatter that removes prefixes for INFO level messages."""

    LEVEL_COLORS = {
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            return record.getMessage()

        levelname = record.levelname
        # Colorize the level name if stderr is a terminal and color is available
        if sys.stderr.isatty() and levelname:
            color = self.LEVEL_COLORS.get(record.levelno)
            if color:
                levelname = f"{color}{levelname}{RESET}"

        return f"{levelname}: {record.getMessage()}"


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


def get_adjacent_keys(include_diagonals: bool = True) -> dict[str, set[str]]:
    """
    Returns a dictionary of adjacent keys on a QWERTY keyboard.
    Can include diagonally adjacent keys based on the 'include_diagonals' flag.

    Args:
        include_diagonals (bool): Whether to include diagonally adjacent keys.

    Returns:
        dict: A mapping from each character to its adjacent characters.
    """
    keyboard = [
        'qwertyuiop',
        'asdfghjkl',
        'zxcvbnm',
    ]

    # Map each character to its (row, column) coordinate for quick lookup
    coords: dict[str, tuple[int, int]] = {}
    for r, row in enumerate(keyboard):
        for c, ch in enumerate(row):
            coords[ch] = (r, c)

    adjacent: dict[str, set[str]] = {ch: set() for ch in coords}

    for ch, (r, c) in coords.items():
        # Examine neighbouring positions within a 1-key radius
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue  # Skip the key itself

                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= len(keyboard):
                    continue
                if nc < 0 or nc >= len(keyboard[nr]):
                    continue

                # Exclude diagonal keys if requested
                if not include_diagonals and dr != 0 and dc != 0:
                    continue

                adjacent_char = keyboard[nr][nc]
                adjacent[ch].add(adjacent_char)

    return adjacent


def is_one_letter_replacement(
    typo: str,
    correction: str,
    allow_1to2: bool = False,
    allow_2to1: bool = False,
    include_deletions: bool = False,
    **kwargs,
) -> list[tuple[str, str]]:
    """
    Check if 'typo' differs from 'correction' by one or more "letter replacements".

    If allow_1to2 is True, check if 'typo' can be formed by replacing a single
    character in 'correction' with two characters in 'typo'.
    If allow_2to1 is True, check if 'typo' can be formed by replacing two characters
    in 'correction' with a single character in 'typo'.

    Returns:
      A list of (correction_char, typo_char_or_chars) tuples for each found replacement.
      The first value comes from the expected spelling (``correction``) and the second
      value comes from the observed typo. Returns an empty list if no replacements are
      found.
    """

    allow_two_char = kwargs.get('allow_two_char', False)
    if allow_two_char:
        allow_1to2 = True
        allow_2to1 = True

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
    if allow_1to2 and len(typo) == len(correction) + 1:
        # Find all positions i where correction[i] is replaced by typo[i:i+2].
        # We use a set to avoid counting identical interpretations (e.g. for doubled letters) multiple times.
        replacements = set()
        for i in range(len(correction)):
            # To be a replacement of correction[i] with typo[i:i+2],
            # the prefix correction[:i] must match typo[:i], and
            # the suffix correction[i+1:] must match typo[i+2:].
            if typo[:i] == correction[:i] and typo[i+2:] == correction[i+1:]:
                repl_correction = correction[i]
                repl_typo = typo[i:i+2]

                # Filter out insertions unless requested
                if not include_deletions:
                    # It's an insertion if the correction character is one of the two typo characters
                    if repl_correction in repl_typo:
                        continue

                replacements.add((repl_correction, repl_typo))
        return sorted(replacements)

    # Two-to-one replacement scenario (e.g. 'ph' -> 'f')
    if allow_2to1 and len(typo) == len(correction) - 1:
        replacements = set()
        for i in range(len(typo)):
            # To be a replacement of correction[i:i+2] with typo[i],
            # the prefix correction[:i] must match typo[:i], and
            # the suffix correction[i+2:] must match typo[i+1:].
            if correction[:i] == typo[:i] and correction[i+2:] == typo[i+1:]:
                repl_correction = correction[i:i+2]
                repl_typo = typo[i]

                # Filter out deletions unless requested
                if not include_deletions:
                    # It's a deletion if the typo character is one of the two correction characters
                    if repl_typo in repl_correction:
                        continue

                replacements.add((repl_correction, repl_typo))
        return sorted(replacements)

    return []


def print_processing_stats(
    raw_item_count: int,
    filtered_item_count: int,
    item_label: str = "item",
) -> None:
    """Print summary statistics for processed text items with visual hierarchy."""
    item_label_plural = f"{item_label}s"

    # Use stderr-specific color check for logging output
    c_bold = BOLD if sys.stderr.isatty() else ""
    c_reset = RESET if sys.stderr.isatty() else ""
    c_yellow = YELLOW if sys.stderr.isatty() else ""
    c_green = GREEN if sys.stderr.isatty() else ""

    padding = "  "
    logging.info(f"\n{padding}{c_bold}ANALYSIS STATISTICS{c_reset}")
    logging.info(f"{padding}{c_bold}───────────────────────────────────────────────────────{c_reset}")
    logging.info(
        f"  {c_bold}{'Total ' + item_label_plural + ' encountered:':<35}{c_reset} {c_yellow}{raw_item_count}{c_reset}"
    )
    logging.info(
        f"  {c_bold}{'Total ' + item_label_plural + ' after filtering:':<35}{c_reset} {c_green}{filtered_item_count}{c_reset}"
    )

    if raw_item_count > 0:
        retention = (filtered_item_count / raw_item_count) * 100
        logging.info(f"  {c_bold}{'Retention rate:':<35}{c_reset} {c_green}{retention:.1f}%{c_reset}")

    if filtered_item_count == 0:
        logging.info(f"  {c_yellow}No {item_label_plural} passed the filtering criteria.{c_reset}")
    logging.info("")


def process_typos(
    lines: Iterable[str],
    allow_1to2: bool = False,
    allow_2to1: bool = False,
    include_deletions: bool = False,
    allow_transposition: bool = False,
    **kwargs,
) -> tuple[dict[tuple[str, str], int], int, int]:
    allow_two_char = kwargs.get('allow_two_char', False)
    if allow_two_char:
        allow_1to2 = True
        allow_2to1 = True

    replacement_counts = defaultdict(int)
    total_lines = 0
    total_pairs = 0
    for line in lines:
        line = line.strip()
        total_lines += 1
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
        elif ": " in line:
            parts = line.split(": ", 1)
            typo = parts[0].strip()
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
            total_pairs += 1
            # Now we have: `typo` (incorrect word), `correction` (correct word)
            # Check for transpositions first if enabled, as they are a specific pattern
            replacements = []
            if allow_transposition:
                replacements = is_transposition(typo, correction)

            # If no transposition found, check for one-letter replacements
            if not replacements:
                replacements = is_one_letter_replacement(
                    typo,
                    correction,
                    allow_1to2=allow_1to2,
                    allow_2to1=allow_2to1,
                    include_deletions=include_deletions,
                )

            for replacement in replacements:
                # replacement is (correct_char, typo_char)
                replacement_counts[replacement] += 1
    return replacement_counts, total_lines, total_pairs


def generate_report(
    replacement_counts: dict[tuple[str, str], int],
    output_file: str | None = None,
    min_occurrences: int = 1,
    sort_by: str = 'count',
    output_format: str = 'arrow',
    limit: int | None = None,
    quiet: bool = False,
    keyboard: bool = False,
    **kwargs,
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
    unique_total = len(replacement_counts)
    unique_filtered = len(filtered)

    total_typos = sum(replacement_counts.values())

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

    if limit is not None:
        sorted_replacements = sorted_replacements[:limit]

    # Color support detection
    # main output colors (used for the report data)
    # These are suppressed if writing to a file or if the main output is not a terminal (piping)
    use_color_stdout = not output_file and sys.stdout.isatty()
    c_out_green = GREEN if use_color_stdout else ""
    c_out_red = RED if use_color_stdout else ""
    c_out_yellow = YELLOW if use_color_stdout else ""
    c_out_bold = BOLD if use_color_stdout else ""
    c_out_reset = RESET if use_color_stdout else ""

    # standard error colors (used for human-readable headers)
    # If output_file is set, we avoid colors in headers that might be written to the file
    use_color_summary = not output_file and sys.stderr.isatty()
    c_err_bold = BOLD if use_color_summary else ""
    c_err_yellow = YELLOW if use_color_summary else ""
    c_err_green = GREEN if use_color_summary else ""
    c_err_reset = RESET if use_color_summary else ""

    if output_format == 'arrow':
        # arrow
        padding = "  "
        title = f"{padding}LETTER REPLACEMENTS"
        label_width = 35

        enabled_features = []
        if keyboard: enabled_features.append("keyboard")
        if kwargs.get('allow_transposition'): enabled_features.append("transposition")
        if kwargs.get('allow_1to2'): enabled_features.append("1-to-2")
        if kwargs.get('allow_2to1'): enabled_features.append("2-to-1")
        if kwargs.get('include_deletions'): enabled_features.append("deletions/insertions")

        features_str = ""
        if enabled_features:
            features_str = f"  {c_err_bold}{'Enabled features:':<{label_width}}{c_err_reset} {', '.join(enabled_features)}"

        keyboard_summary = ""
        adjacent_map = {}
        if keyboard:
            adjacent_map = get_adjacent_keys(include_diagonals=True)
            total_single_char = 0
            adjacent_count = 0
            for (c, t), count in replacement_counts.items():
                if len(c) == 1 and len(t) == 1:
                    total_single_char += count
                    if t.lower() in adjacent_map.get(c.lower(), set()):
                        adjacent_count += count

            if total_single_char > 0:
                percent = (adjacent_count / total_single_char) * 100
                keyboard_summary = f"  {c_err_bold}{'Keyboard Adjacency:':<{label_width}}{c_err_reset} {c_err_yellow}{adjacent_count}{c_err_reset}/{total_single_char} ({c_err_green}{percent:.1f}%{c_err_reset})"

        transposition_summary = ""
        if kwargs.get('allow_transposition'):
            trans_count = 0
            for (c, t), count in replacement_counts.items():
                if len(c) == 2 and len(t) == 2 and c == t[::-1]:
                    trans_count += count
            if trans_count > 0:
                percent = (trans_count / total_typos) * 100 if total_typos > 0 else 0
                transposition_summary = f"  {c_err_bold}{'Transpositions:':<{label_width}}{c_err_reset} {c_err_yellow}{trans_count}{c_err_reset}/{total_typos} ({c_err_green}{percent:.1f}%{c_err_reset})"

        analysis_summary = f"  {c_err_bold}{'Total replacements analyzed:':<{label_width}}{c_err_reset} {c_err_yellow}{total_typos}{c_err_reset}"
        unique_patterns_summary = f"  {c_err_bold}{'Unique patterns identified:':<{label_width}}{c_err_reset} {unique_total}"

        criteria_summary = ""
        if unique_filtered != unique_total:
            criteria_summary = f"  {c_err_bold}{'Patterns matching criteria:':<{label_width}}{c_err_reset} {c_err_green}{unique_filtered}{c_err_reset}"

        display_summary = ""
        if unique_filtered > len(sorted_replacements):
            display_summary = f"  {c_err_bold}{'Showing patterns:':<{label_width}}{c_err_reset} {c_err_green}{len(sorted_replacements)}{c_err_reset} of {unique_filtered}"

        # Calculate padding for alignment (default to header labels' lengths)
        max_c = max((len(c) for (c, t), count in sorted_replacements), default=7)
        max_c = max(max_c, 7)  # 'CORRECT' is 7
        max_t = max((len(t) for (c, t), count in sorted_replacements), default=4)
        max_t = max(max_t, 4)  # 'TYPO' is 4
        max_n = max((len(str(count)) for (c, t), count in sorted_replacements), default=5)
        max_n = max(max_n, 5)  # 'COUNT' is 5
        max_p = 6  # Width for percentage (e.g., "100.0%")

        # Header row and divider with consistent padding and vertical separators
        padding = "  "
        header_row = (
            f"{padding}{c_out_bold}{'CORRECT':>{max_c}}{c_out_reset} │ "
            f"{c_out_bold}{'TYPO':<{max_t}}{c_out_reset} │ "
            f"{c_out_bold}{'COUNT':>{max_n}}{c_out_reset} │ "
            f"{c_out_bold}{'%':>{max_p}}{c_out_reset}"
        )
        # 3 chars for each " │ " (total 3 * 3 = 9)
        visible_header_len = max_c + max_t + max_n + max_p + 9

        show_attr = any([keyboard, kwargs.get('allow_transposition'), kwargs.get('allow_1to2'), kwargs.get('allow_2to1')])

        if show_attr:
            header_row += f" │ {c_out_bold}{'ATTR':<4}{c_out_reset}"
            visible_header_len += 7

        # Add Visual column header
        max_bar = 15
        header_row += f" │ {c_out_bold}{'VISUAL':<{max_bar}}{c_out_reset}"
        visible_header_len += 3 + max_bar

        divider = f"{padding}{c_out_bold}{'─' * visible_header_len}{c_out_reset}"

        if not output_file:
            # Move the human-readable header to standard error to keep the main output clean for piping
            if not quiet:
                sys.stderr.write(f"\n{c_err_bold}{title}{c_err_reset}\n")
                sys.stderr.write(f"{padding}{c_err_bold}───────────────────────────────────────────────────────{c_err_reset}\n")
                sys.stderr.write(f"{analysis_summary}\n")
                sys.stderr.write(f"{unique_patterns_summary}\n")
                if criteria_summary:
                    sys.stderr.write(f"{criteria_summary}\n")
                if features_str:
                    sys.stderr.write(f"{features_str}\n")
                if keyboard_summary:
                    sys.stderr.write(f"{keyboard_summary}\n")
                if transposition_summary:
                    sys.stderr.write(f"{transposition_summary}\n")
                if sorted_replacements:
                    if display_summary:
                        sys.stderr.write(f"{display_summary}\n")
                    sys.stderr.write(f"\n{header_row}\n")
                    sys.stderr.write(f"{divider}\n")
                sys.stderr.flush()
            report_lines = []
        else:
            report_lines = [title, f"{padding}───────────────────────────────────────────────────────", analysis_summary, unique_patterns_summary]
            if criteria_summary:
                report_lines.append(criteria_summary)
            if features_str:
                report_lines.append(features_str)
            if keyboard_summary:
                report_lines.append(keyboard_summary)
            if transposition_summary:
                report_lines.append(transposition_summary)
            if sorted_replacements:
                if display_summary:
                    report_lines.append(display_summary)
                report_lines.extend(["", header_row, divider])

        if not sorted_replacements:
            no_results = f"{padding}No replacements found matching the criteria."
            if not output_file:
                if not quiet:
                    sys.stderr.write(f"{no_results}\n")
            else:
                report_lines.append(no_results)

        for (correct_char, typo_char), count in sorted_replacements:
            percent = (count / total_typos * 100) if total_typos > 0 else 0

            # Create a high-resolution visual bar
            total_blocks = (percent * max_bar) / 100
            full_blocks = int(total_blocks)
            fraction = total_blocks - full_blocks
            blocks = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
            frac_idx = int(fraction * 8)

            bar = "█" * full_blocks
            if full_blocks < max_bar:
                bar += blocks[frac_idx]
                bar += " " * (max_bar - full_blocks - 1)

            row = (
                f"{padding}{c_out_green}{correct_char:>{max_c}}{c_out_reset} │ "
                f"{c_out_red}{typo_char:<{max_t}}{c_out_reset} │ "
                f"{c_out_yellow}{count:>{max_n}}{c_out_reset} │ "
                f"{c_out_green}{percent:>5.1f}%{c_out_reset}"
            )

            if show_attr:
                marker = "    "
                if len(correct_char) == 1 and len(typo_char) == 1:
                    if keyboard and typo_char.lower() in adjacent_map.get(correct_char.lower(), set()):
                        marker = f"{c_out_bold}[K]{c_out_reset} "
                elif len(correct_char) == 2 and len(typo_char) == 2 and correct_char == typo_char[::-1]:
                    marker = f"{c_out_bold}[T]{c_out_reset} "
                elif len(correct_char) != len(typo_char):
                    marker = f"{c_out_bold}[M]{c_out_reset} "
                row += f" │ {marker}"

            row += f" │ {bar}"
            report_lines.append(row)
        report_content = "\n".join(report_lines)
    elif output_format == 'json':
        adjacent_map = {}
        if keyboard:
            adjacent_map = get_adjacent_keys(include_diagonals=True)

        replacements = []
        for (correct_char, typo_char), count in sorted_replacements:
            item = {
                "correct": correct_char,
                "typo": typo_char,
                "count": count,
            }
            if keyboard:
                is_adjacent = False
                if len(correct_char) == 1 and len(typo_char) == 1:
                    if typo_char.lower() in adjacent_map.get(correct_char.lower(), set()):
                        is_adjacent = True
                item["is_adjacent"] = is_adjacent
            replacements.append(item)

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
        logging.info("Reading from standard input...")
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
            with open(file_path, 'r', encoding='latin1') as f:
                lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None

    return lines


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=f"{BOLD}Find common patterns in your typos. This tool analyzes your list of corrections and tells you which keys you hit by mistake most often.{RESET}",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"""{BLUE}Examples:{RESET}
  {GREEN}python typostats.py typos.txt -t{RESET}
  {GREEN}python typostats.py typos.txt --1to2 --2to1{RESET}
  {GREEN}python typostats.py typos.txt -k -n 20{RESET}
  {GREEN}python typostats.py typos.txt -a{RESET}
""",
    )

    # Input/Output Group
    io_group = parser.add_argument_group(f"{BLUE}INPUT/OUTPUT OPTIONS{RESET}")
    io_group.add_argument(
        'input_files',
        nargs='*',
        help="One or more files containing typo corrections. If empty, it reads from standard input.",
    )
    io_group.add_argument('-o', '--output', help="Save the report to this file instead of printing it.")
    io_group.add_argument(
        '-f',
        '--format',
        choices=['arrow', 'yaml', 'json', 'csv'],
        metavar='FMT',
        default='arrow',
        help="The format of the report (default: arrow).",
    )
    io_group.add_argument('-q', '--quiet', action='store_true', help="Suppress informational log output.")

    # Analysis Options Group
    analysis_group = parser.add_argument_group(f"{BLUE}ANALYSIS OPTIONS{RESET}")
    analysis_group.add_argument('-m', '--min', type=int, default=1, help="Only show patterns that appear at least this many times.")
    analysis_group.add_argument(
        '-s', '--sort',
        choices=['count', 'typo', 'correct'],
        default='count',
        help="How to sort the results: 'count' (most frequent first), 'typo' (alphabetical by typo), or 'correct' (alphabetical by fix)."
    )
    analysis_group.add_argument(
        '-a',
        '--all',
        action='store_true',
        help="Enable all analysis options: transposition, keyboard, 2-to-1, 1-to-2, and deletions.",
    )
    analysis_group.add_argument(
        '-2',
        '--allow-two-char',
        dest='allow_two_char',
        action='store_true',
        help="Shortcut for --1to2 and --2to1. Allow multi-character letter replacements.",
    )
    # Hidden alias for backward compatibility
    parser.add_argument('--allow_two_char', action='store_true', help=argparse.SUPPRESS)

    analysis_group.add_argument(
        '--1to2',
        dest='allow_1to2',
        action='store_true',
        help="Allow single-to-double character replacements (e.g., 'm' to 'rn').",
    )
    analysis_group.add_argument(
        '--2to1',
        dest='allow_2to1',
        action='store_true',
        help="Allow double-to-single character replacements (e.g., 'ph' to 'f').",
    )
    analysis_group.add_argument(
        '--include-deletions',
        action='store_true',
        help="Include 2-to-1 deletions (e.g., 'or' to 'o') and 1-to-2 insertions (e.g., 'a' to 'aa').",
    )

    analysis_group.add_argument(
        '-t',
        '--transposition',
        action='store_true',
        help="Detect transpositions of adjacent characters (e.g., 'teh' to 'the').",
    )
    analysis_group.add_argument(
        '-k',
        '--keyboard',
        action='store_true',
        help="Identify typos caused by hitting keys next to each other on the keyboard.",
    )
    analysis_group.add_argument(
        '-n',
        '-L',
        '--limit',
        type=int,
        help="Limit the report to the top N most frequent replacements.",
    )
    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    # Use a custom handler and formatter to keep output clean
    handler = logging.StreamHandler()
    handler.setFormatter(MinimalFormatter('%(levelname)s: %(message)s'))
    logging.basicConfig(level=log_level, handlers=[handler])

    input_files = args.input_files
    output_file = args.output
    min_occurrences = args.min
    sort_by = args.sort
    output_format = args.format
    allow_1to2 = args.allow_1to2
    allow_2to1 = args.allow_2to1
    include_deletions = args.include_deletions
    allow_transposition = args.transposition
    keyboard = args.keyboard

    if args.all:
        allow_1to2 = True
        allow_2to1 = True
        include_deletions = True
        allow_transposition = True
        keyboard = True

    if args.allow_two_char:
        allow_1to2 = True
        allow_2to1 = True
    limit = args.limit

    if not input_files:
        input_files = ['-']

    all_counts = defaultdict(int)
    total_lines_all = 0
    total_pairs_all = 0

    for file_path in input_files:
        lines = load_lines_from_file(file_path)

        if lines:
            if not args.quiet and _TQDM_AVAILABLE:
                lines = tqdm(lines, desc=f"Processing {file_path}", unit="lines", leave=False)

            file_counts, lines_count, pairs_count = process_typos(
                lines,
                allow_1to2=allow_1to2,
                allow_2to1=allow_2to1,
                include_deletions=include_deletions,
                allow_transposition=allow_transposition,
            )
            for k, v in file_counts.items():
                all_counts[k] += v
            total_lines_all += lines_count
            total_pairs_all += pairs_count

    if not args.quiet:
        print_processing_stats(total_pairs_all, sum(all_counts.values()), item_label="replacement")

    generate_report(
        all_counts,
        output_file=output_file,
        min_occurrences=min_occurrences,
        sort_by=sort_by,
        output_format=output_format,
        limit=limit,
        quiet=args.quiet,
        keyboard=keyboard,
        allow_transposition=allow_transposition,
        allow_1to2=allow_1to2,
        allow_2to1=allow_2to1,
        include_deletions=include_deletions,
    )


if __name__ == "__main__":
    main()
