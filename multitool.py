import os
import argparse
import csv
import glob
from collections import Counter, defaultdict, deque
import random
import contextlib
import sys
import re
import time
from textwrap import dedent
from typing import Any, Callable, Iterable, List, Sequence, Tuple, TextIO
from tqdm import tqdm
import logging
import json

try:
    import ahocorasick
    _AHOCORASICK_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _AHOCORASICK_AVAILABLE = False

try:
    import chardet  # type: ignore

    _CHARDET_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    chardet = None
    _CHARDET_AVAILABLE = False

# Cache for standard input to allow multiple passes
_STDIN_CACHE: List[str] | None = None

# ANSI Color Codes
BLUE = "\033[1;34m"
GREEN = "\033[1;32m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Disable colors if not running in a terminal or if NO_COLOR is set
# We check the main output and error output as help goes to the main output and logging/stats to error output
if not sys.stdout.isatty() or os.environ.get('NO_COLOR'):
    BLUE = GREEN = RED = YELLOW = RESET = BOLD = ""
# Note: we use the main output's status for the global constants, but individual
# functions might still check the error output if they specifically log to it.


def _parse_markdown_table_row(line: str) -> List[str] | None:
    """
    Parses a single line as a Markdown table row.
    Returns a list of cell contents if it's a valid data row, otherwise None.
    """
    content = line.strip()
    if not (content.startswith('|') and content.count('|') >= 2):
        return None

    parts = [p.strip() for p in content.split('|')]
    # Filter out empty parts from edges if they exist
    if parts and not parts[0]:
        parts = parts[1:]
    if parts and not parts[-1]:
        parts = parts[:-1]

    if len(parts) < 2:
        return None

    # Skip divider lines like | --- | --- |
    if all(re.match(r'^:?-+:?$', p) for p in parts):
        return None

    # Skip header line if it contains generic labels
    if parts[0].lower() in ('typo', 'left', 'word 1', 'item') and \
       parts[1].lower() in ('correction', 'right', 'word 2', 'count', 'corrections'):
        return None

    return parts


def filter_to_letters(text: str) -> str:
    """Return text containing only lowercase a-z characters."""
    return re.sub("[^a-z]", "", text.lower())


def _apply_smart_case(original: str, replacement: str) -> str:
    """
    Applies the casing of the original string to the replacement string.
    Supports ALL-CAPS, Title Case (Capitalized), and lowercase.
    """
    if not original:
        return replacement
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        # Capitalize first letter, keep rest as provided in replacement
        # (Allows preservation of CamelCase in the replacement itself)
        return replacement[:1].upper() + replacement[1:]
    return replacement.lower()


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the number of character changes needed to turn one string into another."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if not s2:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def get_adjacent_keys(include_diagonals: bool = True) -> dict[str, set[str]]:
    """
    Returns a dictionary of adjacent keys on a QWERTY keyboard.
    """
    keyboard = [
        'qwertyuiop',
        'asdfghjkl',
        'zxcvbnm',
    ]

    coords: dict[str, tuple[int, int]] = {}
    for r, row in enumerate(keyboard):
        for c, ch in enumerate(row):
            coords[ch] = (r, c)

    adjacent: dict[str, set[str]] = {ch: set() for ch in coords}

    for ch, (r, c) in coords.items():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue

                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= len(keyboard):
                    continue
                if nc < 0 or nc >= len(keyboard[nr]):
                    continue

                if not include_diagonals and dr != 0 and dc != 0:
                    continue

                adjacent_char = keyboard[nr][nc]
                adjacent[ch].add(adjacent_char)

    return adjacent


def classify_typo(typo: str, correction: str, adj_keys: dict[str, set[str]]) -> str:
    """
    Categorizes a typo based on its relationship to the correction.
    Returns a code: [K] Keyboard, [T] Transposition, [D] Deletion, [I] Insertion, [R] Replacement, [M] Multiple letters.
    """
    if not typo or not correction:
        return "[?]"

    t_len, c_len = len(typo), len(correction)

    # 1. Transposition [T]
    if t_len == c_len:
        diffs = [i for i in range(t_len) if typo[i] != correction[i]]
        if len(diffs) == 2 and diffs[1] == diffs[0] + 1:
            i, j = diffs
            if typo[i] == correction[j] and typo[j] == correction[i]:
                return "[T]"

    # 2. Deletion [D] - Typo is shorter (a character was removed)
    if t_len == c_len - 1:
        for i in range(c_len):
            if correction[:i] + correction[i+1:] == typo:
                return "[D]"

    # 3. Insertion [I] - Typo is longer (a character was added)
    if t_len == c_len + 1:
        for i in range(t_len):
            if typo[:i] + typo[i+1:] == correction:
                return "[I]"

    # 4. Replacement [R] or [K] - Same length, one character difference
    if t_len == c_len:
        diffs = [i for i in range(t_len) if typo[i] != correction[i]]
        if len(diffs) == 1:
            idx = diffs[0]
            t_char, c_char = typo[idx].lower(), correction[idx].lower()
            if t_char in adj_keys.get(c_char, set()):
                return "[K]"
            return "[R]"

    # 5. Multiple letters [M]
    if levenshtein_distance(typo, correction) > 1:
        return "[M]"

    return "[?]"


def _smart_split(text: str) -> List[str]:
    """
    Splits text into subwords based on non-alphanumeric characters
    and casing boundaries (CamelCase).
    """
    # Split by non-alphanumeric characters
    parts = re.split(r'[^a-zA-Z0-9]+', text)
    subwords = []
    for part in parts:
        if not part:
            continue
        # Split based on casing (camelCase, PascalCase) and numbers.
        # re.findall with this pattern matches:
        # 1. An optional uppercase letter followed by one or more lowercase letters.
        # 2. One or more uppercase letters (not followed by a lowercase letter).
        # 3. One or more digits.
        split_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+', part)
        subwords.extend(split_parts)
    return subwords


def clean_and_filter(items: Iterable[str], min_length: int, max_length: int, clean: bool = True) -> List[str]:
    """Clean items to letters only (if clean=True) and apply length filtering."""
    if clean:
        items = [filter_to_letters(item) for item in items]
    return [c for c in items if min_length <= len(c) <= max_length]


def detect_encoding(file_path: str) -> str | None:
    """Attempt to detect a file's encoding using chardet if available."""

    if not _CHARDET_AVAILABLE:
        logging.warning("chardet not installed. Install via 'pip install chardet'.")
        return None

    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result.get('encoding')
    confidence = result.get('confidence', 0)
    if encoding and confidence > 0.5:
        logging.info(
            "Detected encoding '%s' for '%s' (confidence %.2f)",
            encoding,
            file_path,
            confidence,
        )
        return encoding

    logging.warning("Failed to reliably detect encoding for '%s'.", file_path)
    return None

def _read_file_lines_robust(path: str, newline: str | None = None) -> List[str]:
    """Read lines from a file with robust encoding fallback (UTF-8 -> Detect -> Latin-1)."""
    global _STDIN_CACHE
    lines = []
    used_encoding = 'utf-8'

    if path == '-':
        if _STDIN_CACHE is not None:
            logging.info("Using cached standard input...")
            return list(_STDIN_CACHE)

        logging.info("Reading from standard input...")
        stream = getattr(sys.stdin, "buffer", sys.stdin)
        data = stream.read()
        if isinstance(data, str):
            lines = data.splitlines(keepends=True)
            used_encoding = sys.stdin.encoding or 'utf-8'
        else:
            try:
                text = data.decode("utf-8")
                used_encoding = 'utf-8'
            except UnicodeDecodeError:
                text = data.decode("latin-1")
                used_encoding = 'latin-1'
            lines = text.splitlines(keepends=True)

        _STDIN_CACHE = lines
    else:
        try:
            with open(path, 'r', encoding='utf-8', newline=newline) as handle:
                lines = handle.readlines()
                used_encoding = 'utf-8'
        except UnicodeDecodeError:
            logging.warning("UTF-8 decoding failed for '%s'. Attempting detection...", path)
            detected_encoding = detect_encoding(path)
            if detected_encoding:
                logging.warning(
                    "Using detected encoding '%s' for '%s'.", detected_encoding, path
                )
                try:
                    with open(path, 'r', encoding=detected_encoding, newline=newline) as handle:
                        lines = handle.readlines()
                    used_encoding = detected_encoding
                except UnicodeDecodeError:
                    logging.warning(
                        "Detected encoding '%s' failed for '%s'. Fallback to latin-1.",
                        detected_encoding,
                        path,
                    )
                    with open(path, 'r', encoding='latin-1', newline=newline) as handle:
                        lines = handle.readlines()
                    used_encoding = 'latin-1'
            else:
                logging.warning("Encoding detection failed. Fallback to latin-1 for '%s'.", path)
                with open(path, 'r', encoding='latin-1', newline=newline) as handle:
                    lines = handle.readlines()
                used_encoding = 'latin-1'

    logging.info("Loaded '%s' using %s encoding.", path, used_encoding)
    return lines


def _load_and_clean_file(
    path: str,
    min_length: int,
    max_length: int,
    *,
    split_whitespace: bool = False,
    apply_length_filter: bool = True,
    clean_items: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    """Load text items from *path* and normalize them for set-style operations."""

    raw_items = []
    cleaned_items = []

    lines = _read_file_lines_robust(path)

    for line in lines:
        line_content = line.strip()
        if not line_content:
            continue

        parts = line_content.split() if split_whitespace else [line_content]
        for part in parts:
            raw_items.append(part)
            if clean_items:
                cleaned = filter_to_letters(part)
                if cleaned:
                    cleaned_items.append(cleaned)
            else:
                if part:
                    cleaned_items.append(part)

    if apply_length_filter:
        cleaned_items = [
            item for item in cleaned_items if min_length <= len(item) <= max_length
        ]

    unique_items = list(dict.fromkeys(cleaned_items))
    return raw_items, cleaned_items, unique_items


def print_processing_stats(
    raw_item_count: int,
    filtered_items: Sequence[Any],
    item_label: str = "item",
    start_time: float | None = None,
) -> None:
    """Print summary statistics for processed text items with visual hierarchy."""
    item_label_plural = f"{item_label}s"

    # Colors for stderr logging
    c_bold = BOLD if sys.stderr.isatty() else ""
    c_yellow = YELLOW if sys.stderr.isatty() else ""
    c_green = GREEN if sys.stderr.isatty() else ""
    c_reset = RESET if sys.stderr.isatty() else ""

    padding = "  "
    label_width = 35

    report = []
    report.append(f"\n{padding}{c_bold}ANALYSIS SUMMARY{c_reset}")
    report.append(f"{padding}{c_bold}───────────────────────────────────────────────────────{c_reset}")
    report.append(
        f"  {c_bold}{'Total ' + item_label_plural + ' encountered:':<{label_width}}{c_reset} {c_yellow}{raw_item_count}{c_reset}"
    )
    report.append(
        f"  {c_bold}{'Total ' + item_label_plural + ' after filtering:':<{label_width}}{c_reset} {c_green}{len(filtered_items)}{c_reset}"
    )

    if raw_item_count > 0:
        retention = (len(filtered_items) / raw_item_count) * 100
        report.append(
            f"  {c_bold}{'Retention rate:':<{label_width}}{c_reset} {c_green}{retention:.1f}%{c_reset}"
        )

    # Unique Items
    unique_count = len(set(filtered_items))
    report.append(
        f"  {c_bold}{'Unique ' + item_label_plural + ':':<{label_width}}{c_reset} {c_green}{unique_count}{c_reset}"
    )

    # Levenshtein distance for paired data
    if (
        filtered_items
        and isinstance(filtered_items[0], tuple)
        and len(filtered_items[0]) == 2
    ):
        distances = [levenshtein_distance(str(p[0]), str(p[1])) for p in filtered_items]
        if distances:
            min_dist = min(distances)
            max_dist = max(distances)
            avg_dist = sum(distances) / len(distances)
            report.append(
                f"  {c_bold}{'Min/Max/Avg changes:':<{label_width}}{c_reset} {min_dist} / {max_dist} / {avg_dist:.1f}"
            )

    if filtered_items:

        def format_item(it: Any) -> str:
            if isinstance(it, tuple) and len(it) == 2:
                return f"{it[0]} -> {it[1]}"
            return str(it)

        shortest = min(filtered_items, key=lambda x: len(format_item(x)))
        longest = max(filtered_items, key=lambda x: len(format_item(x)))

        s_display = format_item(shortest)
        l_display = format_item(longest)

        report.append(
            f"  {c_bold}{'Shortest ' + item_label + ':':<{label_width}}{c_reset} '{s_display}' (length: {len(s_display)})"
        )
        report.append(
            f"  {c_bold}{'Longest ' + item_label + ':':<{label_width}}{c_reset} '{l_display}' (length: {len(l_display)})"
        )
    else:
        report.append(
            f"  {c_yellow}No {item_label_plural} passed the filtering criteria.{c_reset}"
        )

    # Processing Time
    if start_time is not None:
        duration = time.perf_counter() - start_time
        report.append(
            f"  {c_bold}{'Processing time:':<{label_width}}{c_reset} {c_green}{duration:.3f}s{c_reset}"
        )

    report.append("")
    logging.info("\n".join(report))


@contextlib.contextmanager
def smart_open_output(filename: str, encoding: str = 'utf-8', newline: str | None = None) -> Iterable[TextIO]:
    """
    Context manager that yields a file object for writing.
    If filename is '-', yields the main output (the screen).
    Otherwise, opens the file for writing.
    """
    if filename == '-':
        yield sys.stdout
    else:
        with open(filename, 'w', encoding=encoding, newline=newline) as f:
            yield f

def write_output(
    items: Iterable[str],
    output_file: str,
    output_format: str = 'line',
    quiet: bool = False,
    limit: int | None = None
) -> None:
    """
    Writes a collection of strings to the output file in the specified format.

    Args:
        items: Collection of strings to write.
        output_file: Path to the output file or '-' for the main output.
        output_format: Format (line, json, csv, markdown, md-table, yaml).
        quiet: If True, suppress informational output.
        limit: If provided, limit the output to the first N items.
    """
    items_list = list(items)  # Consume generator to know length/content
    if limit is not None:
        items_list = items_list[:limit]

    # Use newline='' for CSV format to ensure correct line endings across platforms
    newline = '' if output_format == 'csv' else None

    with smart_open_output(output_file, newline=newline) as outfile:
        if output_format == 'json':
            json.dump(items_list, outfile, indent=2)
            outfile.write('\n')
        elif output_format == 'csv':
            writer = csv.writer(outfile)
            for item in items_list:
                writer.writerow([item])
        elif output_format == 'markdown':
            for item in items_list:
                outfile.write(f"- {item}\n")
        elif output_format == 'md-table':
            outfile.write("| Item |\n")
            outfile.write("| :--- |\n")
            for item in items_list:
                outfile.write(f"| {item} |\n")
        elif output_format == 'yaml':
            try:
                import yaml
                yaml.dump(items_list, outfile, default_flow_style=False)
            except ImportError:
                for item in items_list:
                    outfile.write(f"- {item}\n")
        else:  # 'line' or fallback
            for item in items_list:
                outfile.write(item + '\n')


def _extract_pairs(input_files: Sequence[str], quiet: bool = False) -> Iterable[Tuple[str, str]]:
    """Yield (left, right) pairs from input files, supporting multiple formats."""
    for input_file in input_files:
        ext = input_file.lower()
        if ext.endswith('.json'):
            content = "".join(_read_file_lines_robust(input_file))
            if content.strip():
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        if 'replacements' in data and isinstance(data['replacements'], list):
                            for item in data['replacements']:
                                if isinstance(item, dict) and 'typo' in item:
                                    correct = item.get('correct', item.get('correction'))
                                    if correct is not None:
                                        yield str(item['typo']), str(correct)
                        else:
                            for k, v in data.items():
                                yield str(k), str(v)
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'typo' in item:
                                correct = item.get('correct', item.get('correction'))
                                if correct is not None:
                                    yield str(item['typo']), str(correct)
                except Exception as e:
                    logging.error(f"Failed to parse JSON in '{input_file}': {e}")
            continue

        if ext.endswith('.yaml') or ext.endswith('.yml'):
            try:
                import yaml
                content = "".join(_read_file_lines_robust(input_file))
                for doc in yaml.safe_load_all(content):
                    if isinstance(doc, dict):
                        for k, v in doc.items():
                            yield str(k), str(v)
                    elif isinstance(doc, list):
                        for item in doc:
                            if isinstance(item, dict):
                                if 'typo' in item:
                                    correct = item.get('correct', item.get('correction'))
                                    if correct is not None:
                                        yield str(item['typo']), str(correct)
                                        continue
                                for k, v in item.items():
                                    yield str(k), str(v)
            except ImportError:
                logging.error("PyYAML not installed.")
            except Exception as e:
                logging.error(f"Failed to parse YAML in '{input_file}': {e}")
            continue

        # Text formats
        lines = _read_file_lines_robust(input_file)
        for line in tqdm(lines, desc=f'Processing {input_file}', unit=' lines', disable=quiet):
            content = line.strip()
            if not content or content.startswith('#'):
                continue

            # Strip Markdown bullet points if present to handle list items consistently
            content = re.sub(r'^\s*[-*+]\s+', '', content)

            table_parts = _parse_markdown_table_row(line)
            if table_parts:
                yield table_parts[0], table_parts[1]
                continue

            if " -> " in content:
                parts = content.split(" -> ", 1)
                yield parts[0].strip(), parts[1].strip()
            elif ' = "' in content:
                parts = content.split(' = "', 1)
                yield parts[0].strip(), parts[1].rsplit('"', 1)[0]
            elif ": " in content:
                parts = content.split(": ", 1)
                yield parts[0].strip(), parts[1].strip()
            else:
                try:
                    reader = csv.reader([content])
                    row = next(reader)
                    if len(row) >= 2:
                        yield row[0].strip(), row[1].strip()
                except (csv.Error, StopIteration):
                    continue


def _write_paired_output(
    pairs: Iterable[Tuple[str, str]],
    output_file: str,
    output_format: str,
    mode_label: str,
    quiet: bool = False,
    limit: int | None = None,
) -> None:
    """
    Writes a collection of paired strings to the output file in the specified format.

    Args:
        pairs: Collection of (left, right) tuples.
        output_file: Path to the output file or '-' for the main output.
        output_format: Format (arrow, table, csv, markdown, md-table, json, yaml).
        mode_label: Label for the current mode (used for headers).
        quiet: If True, suppress informational output.
        limit: If provided, limit the output to the first N pairs.
    """
    pairs_list = list(pairs)
    if limit is not None:
        pairs_list = pairs_list[:limit]

    # Determine newline behavior for CSV
    newline = '' if output_format == 'csv' else None

    # Determine headers for paired data modes (used in md-table and arrow formats)
    left_header = "Left"
    right_header = "Right"
    if mode_label == "Conflict":
        left_header = "Typo"
        right_header = "Corrections"
    elif mode_label in ("Similarity", "Pairs", "Swap", "Zip", "Classify", "FuzzyMatch", "Discovery", "Map"):
        left_header = "Typo"
        right_header = "Correction"
    elif mode_label == "NearDuplicates":
        left_header = "Word 1"
        right_header = "Word 2"
    elif mode_label == "Casing":
        left_header = "Normalized"
        right_header = "Variations"
    elif mode_label == "Repeated":
        left_header = "Repeated Words"
        right_header = "Fix"

    with smart_open_output(output_file, newline=newline) as out_file:
        if output_format == 'json':
            json_data = {left: right for left, right in pairs_list}
            json.dump(json_data, out_file, indent=2)
            out_file.write('\n')
        elif output_format == 'yaml':
            try:
                import yaml
                # Using a dictionary preserves pairs but deduplicates keys.
                # Since pairs_list is already deduplicated if process_output is True,
                # this is generally safe.
                yaml_data = dict(pairs_list)
                yaml.dump(yaml_data, out_file, default_flow_style=False, sort_keys=False)
            except ImportError:
                # Fallback to simple format if PyYAML not available
                for left, right in pairs_list:
                    out_file.write(f"{left}: {right}\n")
        elif output_format == 'csv':
            writer = csv.writer(out_file)
            for left, right in pairs_list:
                writer.writerow([left, right])
        elif output_format == 'table':
            for left, right in pairs_list:
                out_file.write(f'{left} = "{right}"\n')
        elif output_format == 'markdown':
            for left, right in pairs_list:
                out_file.write(f"- {left}: {right}\n")
        elif output_format == 'md-table':
            out_file.write(f"| {left_header} | {right_header} |\n")
            out_file.write("| :--- | :--- |\n")
            for left, right in pairs_list:
                out_file.write(f"| {left} | {right} |\n")
        elif output_format == 'arrow' and (out_file.isatty() or os.environ.get('FORCE_COLOR')):
            # Dynamic column width calculation for aligned table
            max_left = max((len(str(left)) for left, _ in pairs_list), default=len(left_header))
            max_left = max(max_left, len(left_header))
            max_right = max((len(str(right)) for _, right in pairs_list), default=len(right_header))
            max_right = max(max_right, len(right_header))

            # Colors for table
            c_bold = BOLD if out_file.isatty() else ""
            c_blue = BLUE if out_file.isatty() else ""
            c_green = GREEN if out_file.isatty() else ""
            c_reset = RESET if out_file.isatty() else ""

            # Header and divider
            padding = "  "
            header = f"{padding}{c_bold}{c_blue}{left_header:<{max_left}}{c_reset} │ {c_bold}{c_blue}{right_header:<{max_right}}{c_reset}"
            # 3 chars for the separator " │ "
            visible_width = max_left + max_right + 3
            divider = f"{padding}{c_bold}{'─' * visible_width}{c_reset}"

            out_file.write(f"\n{header}\n")
            out_file.write(f"{divider}\n")
            for left, right in pairs_list:
                out_file.write(f"{padding}{c_green}{left:<{max_left}}{c_reset} │ {right}\n")
            out_file.write("\n")
        else:  # 'line' or fallback
            for left, right in pairs_list:
                out_file.write(f"{left} -> {right}\n")

    logging.info(
        f"[{mode_label} Mode] Processed {len(pairs_list)} pairs. Output written to '{output_file}' in {output_format} format."
    )


def _process_items(
    extractor_func: Callable[[str, bool], Iterable[str]],
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    mode_name: str,
    success_msg: str,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """Generic processing for modes that extract raw string items from one or more files."""
    start_time = time.perf_counter()

    raw_items = [
        item for input_file in input_files
        for item in extractor_func(input_file, quiet=quiet)
    ]
    filtered_items = clean_and_filter(raw_items, min_length, max_length, clean=clean_items)

    if process_output:
        # Note: If not cleaning, duplicates might differ by case/whitespace if user wants that.
        # But process_output implies "normalize, sort, dedup".
        # If clean_items is False, we just sort and dedup raw strings.
        filtered_items = sorted(set(filtered_items))

    write_output(filtered_items, output_file, output_format, quiet, limit=limit)

    print_processing_stats(len(raw_items), filtered_items, start_time=start_time)
    logging.info(
        f"[{mode_name} Mode] {success_msg} Output written to '{output_file}'."
    )


def _extract_arrow_items(input_file: str, right_side: bool = False, quiet: bool = False) -> Iterable[str]:
    """Yield text before (or after) ' -> ' from each line."""
    lines = _read_file_lines_robust(input_file)
    for line in tqdm(lines, desc=f'Processing {input_file} (arrow)', unit=' lines', disable=quiet):
        if " -> " in line:
            parts = line.split(" -> ", 1)
            idx = 1 if right_side else 0
            yield parts[idx].strip()


def _extract_table_items(input_file: str, right_side: bool = False, quiet: bool = False) -> Iterable[str]:
    """Yield text before (or after) ' = ' from each line, handling quotes for the value."""
    lines = _read_file_lines_robust(input_file)
    for line in tqdm(lines, desc=f'Processing {input_file} (table)', unit=' lines', disable=quiet):
        if ' = "' in line:
            parts = line.split(' = "', 1)
            if right_side:
                # Value is after ' = "' and ends with a quote. We extract everything up to the last quote.
                if '"' in parts[1]:
                    yield parts[1].rsplit('"', 1)[0]
                else:
                    yield parts[1].strip()
            else:
                yield parts[0].strip()


def _extract_backtick_items(input_file: str, quiet: bool = False) -> Iterable[str]:
    """Yield text found between backticks with heuristics for diagnostics."""

    context_markers = ("error:", "warning:", "note:")

    lines = _read_file_lines_robust(input_file)
    for line in tqdm(lines, desc=f'Processing {input_file} (backtick)', unit=' lines', disable=quiet):
        parts = line.split('`')
        if len(parts) < 3:
            continue

        candidates = []
        prioritized = []
        has_marker = False
        for index in range(1, len(parts), 2):
            item = parts[index].strip()
            if not item:
                continue

            candidates.append(item)
            preceding = parts[index - 1].lower()
            if any(marker in preceding for marker in context_markers):
                has_marker = True

            if has_marker:
                prioritized.append(item)

        if prioritized:
            yield from prioritized
        else:
            yield from candidates


def _traverse_data(data: Any, path_parts: List[str]) -> Iterable[str]:
    """Recursively traverse a nested data structure (list/dict) to extract values."""
    # If it's a list, apply the current path traversal to every item
    if isinstance(data, list):
        for item in data:
            yield from _traverse_data(item, path_parts)
        return

    # If we are at the end of the path, yield the string representation of the data
    if not path_parts:
        if isinstance(data, dict):
            # For a dictionary root, yield the keys (common for typo mappings)
            yield from (str(k) for k in data.keys())
        else:
            yield str(data)
        return

    current_key = path_parts[0]
    if isinstance(data, dict):
        if current_key in data:
            yield from _traverse_data(data[current_key], path_parts[1:])


def _extract_json_items(input_file: str, key_path: str, quiet: bool = False) -> Iterable[str]:
    """Yield values from JSON objects based on a dotted key path."""

    path_parts = key_path.split('.') if key_path else []

    lines = _read_file_lines_robust(input_file)
    content = "".join(lines)
    # Load the entire file content as JSON
    # Note: Standard JSON parsers expect the whole file. Streaming JSON (JSONL) is handled differently.
    # Here we assume standard JSON as output by typostats.py.
    try:
        if not content.strip():
            return
        data = json.loads(content)
        yield from _traverse_data(data, path_parts)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON in '{input_file}': {e}")
        return


def _extract_yaml_items(input_file: str, key_path: str, quiet: bool = False) -> Iterable[str]:
    """Yield values from YAML objects based on a dotted key path."""

    # Lazy import to avoid crashing if PyYAML is not installed and other modes are used
    try:
        import yaml
    except ImportError:
        logging.error("PyYAML is not installed. Install via 'pip install PyYAML' to use YAML mode.")
        sys.exit(1)

    path_parts = key_path.split('.') if key_path else []

    lines = _read_file_lines_robust(input_file)
    content = "".join(lines)
    try:
        # yaml.safe_load_all yields a generator of documents
        for doc in yaml.safe_load_all(content):
            if doc is None:
                continue
            yield from _traverse_data(doc, path_parts)
    except yaml.YAMLError as e:
        logging.error(f"Failed to parse YAML in '{input_file}': {e}")
        return


def _extract_csv_items(
    input_file: str,
    first_column: bool,
    delimiter: str = ',',
    quiet: bool = False,
    columns: List[int] | None = None,
) -> Iterable[str]:
    """Yield fields from CSV rows based on column selection."""
    lines = _read_file_lines_robust(input_file)
    reader = csv.reader(lines, delimiter=delimiter)
    for row in tqdm(reader, desc=f'Processing {input_file} (CSV)', unit=' rows', disable=quiet):
        if columns is not None:
            for idx in columns:
                if 0 <= idx < len(row):
                    yield row[idx].strip()
        elif first_column:
            if row:
                yield row[0].strip()
        else:
            if len(row) >= 2:
                for field in row[1:]:
                    yield field.strip()


def _extract_line_items(input_file: str, quiet: bool = False) -> Iterable[str]:
    """Yield each line from the file."""
    lines = _read_file_lines_robust(input_file)
    for line in tqdm(lines, desc=f'Processing {input_file} (lines)', unit=' lines', disable=quiet):
        yield line.rstrip('\n')


def _yield_words_from_lines(
    lines: Iterable[str],
    delimiter: str | None = None,
    smart: bool = False,
) -> Iterable[str]:
    """Yield individual words from an iterable of lines."""
    for line in lines:
        parts = line.split(delimiter)
        for part in parts:
            if smart:
                yield from _smart_split(part)
            else:
                word = part.strip()
                if word:
                    yield word


def _extract_words_items(
    input_file: str,
    delimiter: str | None = None,
    quiet: bool = False,
    smart: bool = False,
) -> Iterable[str]:
    """Yield individual words from each line, split by delimiter (default whitespace)."""
    lines = _read_file_lines_robust(input_file)
    yield from _yield_words_from_lines(
        tqdm(lines, desc=f'Processing {input_file} (words)', unit=' lines', disable=quiet),
        delimiter=delimiter,
        smart=smart
    )


def _extract_markdown_items(input_file: str, right_side: bool = False, quiet: bool = False) -> Iterable[str]:
    """Yield text from Markdown list items, optionally splitting by ':' or '->'."""
    lines = _read_file_lines_robust(input_file)
    # Match bullet points: - , * , + at the start of line (optional whitespace)
    # We require a space after the marker to distinguish from other symbols (like horizontal rules '---')
    pattern = re.compile(r'^\s*[-*+]\s+(.*)$')

    for line in tqdm(lines, desc=f'Processing {input_file} (markdown)', unit=' lines', disable=quiet):
        match = pattern.match(line)
        if match:
            content = match.group(1).strip()
            if not content:
                continue

            # Check for common separators if we want to support --right
            # This allows extracting from pairs like "- typo: correction"
            separator = None
            if " -> " in content:
                separator = " -> "
            elif ": " in content:
                separator = ": "

            if separator:
                # split(separator, 1) always returns a list of length 2 if separator is found
                parts = content.split(separator, 1)
                idx = 1 if right_side else 0
                yield parts[idx].strip()
            elif not right_side:
                yield content


def _extract_md_table_items(
    input_file: str,
    right_side: bool = False,
    quiet: bool = False,
    columns: List[int] | None = None,
) -> Iterable[str]:
    """Yield text from a specific column in Markdown tables."""
    lines = _read_file_lines_robust(input_file)
    for line in tqdm(lines, desc=f'Processing {input_file} (md-table)', unit=' lines', disable=quiet):
        parts = _parse_markdown_table_row(line)
        if parts:
            if columns is not None:
                for idx in columns:
                    if 0 <= idx < len(parts):
                        yield parts[idx]
            else:
                idx = 1 if right_side else 0
                yield parts[idx]


def _extract_regex_items(input_file: str, pattern: str, quiet: bool = False) -> Iterable[str]:
    """Yield text matching the compiled regex pattern from the file."""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        logging.error(f"Invalid regular expression '{pattern}': {e}")
        sys.exit(1)

    lines = _read_file_lines_robust(input_file)
    for line in tqdm(lines, desc=f'Processing {input_file} (regex)', unit=' lines', disable=quiet):
        matches = regex.findall(line)
        for match in matches:
            if isinstance(match, tuple):
                # If multiple groups, yield them as separate items
                for group in match:
                    yield group
            else:
                yield match


def _extract_repeated_items(
    input_files: Sequence[str],
    delimiter: str | None = None,
    quiet: bool = False,
    smart: bool = False,
    clean_items: bool = True,
    min_length: int = 1,
    max_length: int = 1000,
) -> Iterable[Tuple[str, str]]:
    """Yield pairs of (repeated words, single word) from input files."""

    for input_file in input_files:
        prev_word: str | None = None
        prev_raw: str | None = None
        words_gen = _extract_words_items(input_file, delimiter=delimiter, quiet=quiet, smart=smart)
        for word in words_gen:
            # Word for matching
            match_word = filter_to_letters(word) if clean_items else word
            if not match_word:
                prev_word = None
                prev_raw = None
                continue

            # Check length of the word itself
            if not (min_length <= len(match_word) <= max_length):
                prev_word = None
                prev_raw = None
                continue

            if prev_word is not None and match_word == prev_word:
                # If cleaning is enabled, we use the cleaned version for both.
                # This ensures consistent casing and format in the output.
                if clean_items:
                    yield f"{match_word} {match_word}", match_word
                else:
                    yield f"{prev_raw} {word}", word

            prev_word = match_word
            prev_raw = word


def _extract_ngram_items(
    input_file: str,
    n: int = 2,
    delimiter: str | None = None,
    quiet: bool = False,
    smart: bool = False,
    clean_items: bool = True,
) -> Iterable[str]:
    """Yield sequences of N words joined by spaces."""
    lines = _read_file_lines_robust(input_file)
    words_gen = _yield_words_from_lines(
        tqdm(lines, desc=f'Processing {input_file} (ngrams)', unit=' lines', disable=quiet),
        delimiter=delimiter,
        smart=smart
    )

    window = deque(maxlen=n)
    for word in words_gen:
        if clean_items:
            word = filter_to_letters(word)
            if not word:
                continue

        window.append(word)
        if len(window) == n:
            yield " ".join(window)


def ngrams_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    n: int = 2,
    delimiter: str | None = None,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
    smart: bool = False,
) -> None:
    """Wrapper for extracting N-grams from file(s)."""
    def extractor(f, quiet=False):
        return _extract_ngram_items(
            f, n=n, delimiter=delimiter, quiet=quiet, smart=smart, clean_items=clean_items
        )
    # Pass clean_items=False to _process_items to preserve spaces in n-grams.
    _process_items(
        extractor,
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        'Ngrams',
        f'{n}-grams extracted successfully.',
        output_format,
        quiet,
        clean_items=False,
        limit=limit,
    )


def arrow_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    right_side: bool = False,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """Wrapper for processing items separated by ' -> '."""
    def extractor(f, quiet=False):
        return _extract_arrow_items(f, right_side=right_side, quiet=quiet)
    _process_items(
        extractor,
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        'Arrow',
        'File(s) processed successfully.',
        output_format,
        quiet,
        clean_items=clean_items,
        limit=limit,
    )


def table_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    right_side: bool = False,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """Wrapper for processing items in 'key = \"value\"' format."""
    def extractor(f, quiet=False):
        return _extract_table_items(f, right_side=right_side, quiet=quiet)
    _process_items(
        extractor,
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        'Table',
        'Table fields extracted successfully.',
        output_format,
        quiet,
        clean_items=clean_items,
        limit=limit,
    )


def markdown_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    right_side: bool = False,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """Wrapper for processing items from Markdown bulleted lists."""
    def extractor(f, quiet=False):
        return _extract_markdown_items(f, right_side=right_side, quiet=quiet)
    _process_items(
        extractor,
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        'Markdown',
        'Markdown list items extracted successfully.',
        output_format,
        quiet,
        clean_items=clean_items,
        limit=limit,
    )


def md_table_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    right_side: bool = False,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
    columns: List[int] | None = None,
) -> None:
    """Wrapper for processing items from Markdown tables."""
    def extractor(f, quiet=False):
        return _extract_md_table_items(
            f, right_side=right_side, quiet=quiet, columns=columns
        )
    _process_items(
        extractor,
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        'MDTable',
        'Markdown table items extracted successfully.',
        output_format,
        quiet,
        clean_items=clean_items,
        limit=limit,
    )


def backtick_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """Wrapper for extracting text between backticks."""
    _process_items(
        _extract_backtick_items,
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        'Backtick',
        'Strings extracted successfully.',
        output_format,
        quiet,
        clean_items=clean_items,
        limit=limit,
    )


def json_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    key: str,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """Wrapper for extracting fields from JSON files."""
    def extractor(f, quiet=False):
        return _extract_json_items(f, key, quiet=quiet)
    _process_items(
        extractor,
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        'JSON',
        'JSON values extracted successfully.',
        output_format,
        quiet,
        clean_items=clean_items,
        limit=limit,
    )


def yaml_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    key: str,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """Wrapper for extracting fields from YAML files."""
    def extractor(f, quiet=False):
        return _extract_yaml_items(f, key, quiet=quiet)
    _process_items(
        extractor,
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        'YAML',
        'YAML values extracted successfully.',
        output_format,
        quiet,
        clean_items=clean_items,
        limit=limit,
    )


def count_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    min_count: int = 1,
    max_count: int | None = None,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
    delimiter: str | None = None,
    smart: bool = False,
    pairs: bool = False,
) -> None:
    """
    Counts the frequency of each word or pair in the input file(s) and writes the
    sorted results to the output file. Only items with length between
    min_length and max_length are counted.
    The stats are based on the raw count of items versus the filtered items.
    Note: process_output is ignored in count mode.
    """
    raw_count = 0
    filtered_items = []
    item_counts = Counter()

    start_time = time.perf_counter()
    if pairs:
        # Mode for counting typo -> correction pairs
        for left, right in _extract_pairs(input_files, quiet=quiet):
            raw_count += 1
            if clean_items:
                left = filter_to_letters(left)
                right = filter_to_letters(right)
            if not left or not right:
                continue
            if min_length <= len(left) <= max_length and min_length <= len(right) <= max_length:
                filtered_items.append((left, right))
                item_counts.update([(left, right)])
    else:
        # Default mode for counting individual words
        for input_file in input_files:
            # Use the shared extraction logic to support custom delimiters and smart splitting
            words_gen = _extract_words_items(input_file, delimiter=delimiter, quiet=quiet, smart=smart)
            for word in words_gen:
                raw_count += 1
                # Filter and clean the word
                filtered = clean_and_filter([word], min_length, max_length, clean=clean_items)
                if filtered:
                    filtered_items.extend(filtered)
                    item_counts.update(filtered)

    sorted_words = sorted(item_counts.items(), key=lambda x: (-x[1], x[0]))

    # Apply frequency filtering
    final_results = []
    for item, count in sorted_words:
        if count < min_count:
            continue
        if max_count is not None and count > max_count:
            continue
        final_results.append((item, count))

    if limit is not None:
        final_results = final_results[:limit]

    # Determine newline behavior for CSV
    newline = '' if output_format == 'csv' else None

    with smart_open_output(output_file, newline=newline) as out_file:
        if output_format == 'json':
            if pairs:
                json_data = [{"typo": item[0], "correction": item[1], "count": count} for item, count in final_results]
            else:
                json_data = [{"item": item, "count": count} for item, count in final_results]
            json.dump(json_data, out_file, indent=2)
            out_file.write('\n')
        elif output_format == 'csv':
            writer = csv.writer(out_file)
            if pairs:
                writer.writerow(["typo", "correction", "count"])
                for item, count in final_results:
                    writer.writerow([item[0], item[1], count])
            else:
                for item, count in final_results:
                    writer.writerow([item, count])
        elif output_format == 'markdown':
            for item, count in final_results:
                label = f"{item[0]} -> {item[1]}" if pairs else item
                out_file.write(f"- {label}: {count}\n")
        elif output_format == 'md-table':
            if pairs:
                out_file.write("| Typo | Correction | Count |\n")
                out_file.write("| :--- | :--- | :--- |\n")
                for item, count in final_results:
                    out_file.write(f"| {item[0]} | {item[1]} | {count} |\n")
            else:
                out_file.write("| Item | Count |\n")
                out_file.write("| :--- | :--- |\n")
                for item, count in final_results:
                    out_file.write(f"| {item} | {count} |\n")
        elif output_format == 'arrow':
            # Rich visual report for arrow format
            total_count = sum(item_counts.values())

            # Format item labels for visualization
            if pairs:
                labels = [f"{item[0]} -> {item[1]}" for item, _ in final_results]
                item_header = "TYPO -> CORRECTION"
            else:
                labels = [str(item) for item, _ in final_results]
                item_header = "ITEM"

            # Find max width for columns
            max_item = max((len(label) for label in labels), default=len(item_header))
            max_item = max(max_item, len(item_header))
            max_count_len = max((len(str(count)) for item, count in final_results), default=5)
            max_count_len = max(max_count_len, 5)  # 'COUNT'
            max_pct = 6  # "100.0%"
            max_bar = 20

            # Colors for output
            # main output colors (used for the report data)
            # These are suppressed if writing to a file or if the main output is not a terminal (piping)
            use_color_stdout = out_file.isatty()
            c_out_green = GREEN if use_color_stdout else ""
            c_out_red = RED if use_color_stdout else ""
            c_out_yellow = YELLOW if use_color_stdout else ""
            c_out_bold = BOLD if use_color_stdout else ""
            c_out_reset = RESET if use_color_stdout else ""

            # standard error colors (used for human-readable headers)
            # If output_file is set to a real file, we avoid colors in headers that might be written to the file
            use_color_err = (output_file == '-') and sys.stderr.isatty()
            c_err_bold = BOLD if use_color_err else ""
            c_err_yellow = YELLOW if use_color_err else ""
            c_err_green = GREEN if use_color_err else ""
            c_err_reset = RESET if use_color_err else ""

            # Header and divider
            padding = "  "

            # Dashboard Summary (Consolidated)
            label_width = 35
            summary_buffer = []
            summary_buffer.append(f"\n{padding}{c_err_bold}ANALYSIS SUMMARY{c_err_reset}")
            summary_buffer.append(f"{padding}{c_err_bold}───────────────────────────────────────────────────────{c_err_reset}")

            item_label = "pair" if pairs else "word"
            item_label_plural = f"{item_label}s"

            summary_buffer.append(f"  {c_err_bold}{'Total ' + item_label_plural + ' encountered:':<{label_width}}{c_err_reset} {c_err_yellow}{raw_count}{c_err_reset}")
            summary_buffer.append(f"  {c_err_bold}{'Total ' + item_label_plural + ' after filtering:':<{label_width}}{c_err_reset} {c_err_green}{len(filtered_items)}{c_err_reset}")

            if raw_count > 0:
                retention = (len(filtered_items) / raw_count) * 100
                summary_buffer.append(f"  {c_err_bold}{'Retention rate:':<{label_width}}{c_err_reset} {c_err_green}{retention:.1f}%{c_err_reset}")

            unique_count = len(item_counts)
            summary_buffer.append(f"  {c_err_bold}{'Unique ' + item_label_plural + ':':<{label_width}}{c_err_reset} {c_err_green}{unique_count}{c_err_reset}")

            if pairs and filtered_items:
                distances = [levenshtein_distance(p[0], p[1]) for p in filtered_items]
                if distances:
                    min_dist = min(distances)
                    max_dist = max(distances)
                    avg_dist = sum(distances) / len(distances)
                    summary_buffer.append(f"  {c_err_bold}{'Min/Max/Avg changes:':<{label_width}}{c_err_reset} {min_dist} / {max_dist} / {avg_dist:.1f}")

            if filtered_items:
                def format_item_local(it: Any) -> str:
                    if isinstance(it, tuple) and len(it) == 2:
                        return f"{it[0]} -> {it[1]}"
                    return str(it)

                shortest = min(filtered_items, key=lambda x: len(format_item_local(x)))
                longest = max(filtered_items, key=lambda x: len(format_item_local(x)))

                s_display = format_item_local(shortest)
                l_display = format_item_local(longest)

                summary_buffer.append(f"  {c_err_bold}{'Shortest ' + item_label + ':':<{label_width}}{c_err_reset} '{s_display}' (length: {len(s_display)})")
                summary_buffer.append(f"  {c_err_bold}{'Longest ' + item_label + ':':<{label_width}}{c_err_reset} '{l_display}' (length: {len(l_display)})")

            duration = time.perf_counter() - start_time
            summary_buffer.append(
                f"  {c_err_bold}{'Processing time:':<{label_width}}{c_err_reset} {c_err_green}{duration:.3f}s{c_err_reset}"
            )

            # Determine colors for headers (might go to stdout or stderr)
            if output_file == '-' and not quiet:
                # Headers go to stderr
                c_head_bold = c_err_bold
                c_head_reset = c_err_reset
            else:
                # Headers go to out_file (stdout or real file)
                c_head_bold = c_out_bold
                c_head_reset = c_out_reset

            header = (
                f"{padding}{c_head_bold}{item_header:<{max_item}}{c_head_reset} │ "
                f"{c_head_bold}{'COUNT':>{max_count_len}}{c_head_reset} │ "
                f"{c_head_bold}{'%':>{max_pct}}{c_head_reset} │ "
                f"{c_head_bold}{'VISUAL':<{max_bar}}{c_head_reset}"
            )
            # sum(column_widths) + 3 * len(' │ ') = sum + 9
            visible_header_len = max_item + max_count_len + max_pct + max_bar + 9
            divider = f"{padding}{c_head_bold}{'─' * visible_header_len}{c_head_reset}"

            # Write summary and headers to either stderr (if piping) or the output stream
            output_summary = "\n".join(summary_buffer) + "\n"
            output_header_block = f"\n{header}\n{divider}\n"

            if output_file == '-':
                if not quiet:
                    sys.stderr.write(output_summary)
                    sys.stderr.write(output_header_block)
                    sys.stderr.flush()
            else:
                out_file.write(output_summary)
                out_file.write(output_header_block)

            for i, (item, count) in enumerate(final_results):
                percent = (count / total_count * 100) if total_count > 0 else 0
                label = labels[i]

                # High-res visual bar
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
                    f"{padding}{c_out_green}{label:<{max_item}}{c_out_reset} │ "
                    f"{c_out_yellow}{count:>{max_count_len}}{c_out_reset} │ "
                    f"{c_out_green}{percent:>5.1f}%{c_out_reset} │ "
                    f"{c_out_red}{bar}{c_out_reset}"
                )
                out_file.write(f"{row}\n")
            out_file.write("\n")
        else:  # 'line' or fallback
            for item, count in final_results:
                label = f"{item[0]} -> {item[1]}" if pairs else item
                out_file.write(f"{label}: {count}\n")

    if output_format != 'arrow':
        print_processing_stats(
            raw_count,
            filtered_items,
            item_label="pair" if pairs else "word",
            start_time=start_time,
        )
        logging.info(
            f"[Count Mode] Word frequencies ({len(final_results)} items) have been written to '{output_file}' in {output_format} format."
        )


def classify_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    show_dist: bool = False,
    output_format: str = 'arrow',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """
    Categorizes typo corrections based on their error type.
    """
    start_time = time.perf_counter()
    raw_pairs = _extract_pairs(input_files, quiet=quiet)
    adj_keys = get_adjacent_keys()

    results = []
    raw_count = 0
    for left, right in raw_pairs:
        raw_count += 1
        # Clean if requested
        if clean_items:
            left_clean = filter_to_letters(left)
            right_clean = filter_to_letters(right)
        else:
            left_clean = left
            right_clean = right

        # Skip if either side is empty after cleaning
        if not left_clean or not right_clean:
            continue

        # Apply length filtering
        if min_length <= len(left_clean) <= max_length and min_length <= len(right_clean) <= max_length:
            label = classify_typo(left_clean, right_clean, adj_keys)
            if show_dist:
                dist = levenshtein_distance(left_clean, right_clean)
                label = f"{label} (dist: {dist})"
            results.append((left, f"{right} {label}"))

    if process_output:
        results = sorted(set(results))

    _write_paired_output(
        results,
        output_file,
        output_format,
        "Classify",
        quiet,
        limit=limit
    )

    # Use actual result pairs for stats to enable distance calculation
    stats_items = []
    for typo, correction_with_label in results:
        # correction_with_label is like "correction [T]" or "correction [T] (dist: 1)"
        # We need the base correction for distance stats
        base_correction = correction_with_label.split(' [')[0]
        stats_items.append((typo, base_correction))

    print_processing_stats(
        raw_count, stats_items, item_label="classified-typo", start_time=start_time
    )


def stats_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    include_pairs: bool = False,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """
    Calculates and displays statistics for items or paired data.
    """
    start_time = time.perf_counter()
    # 1. Collect Items
    raw_item_count = 0
    all_items = []
    for input_file in input_files:
        lines = _read_file_lines_robust(input_file)
        for line in lines:
            line_content = line.strip()
            if not line_content:
                continue
            parts = line_content.split()
            raw_item_count += len(parts)
            all_items.extend(parts)

    filtered_items = clean_and_filter(all_items, min_length, max_length, clean=clean_items)
    unique_items = list(dict.fromkeys(filtered_items))
    unique_count = len(unique_items)

    stats = {
        "items": {
            "total_encountered": raw_item_count,
            "total_filtered": len(filtered_items),
            "unique_count": unique_count,
        }
    }

    if filtered_items:
        lengths = [len(i) for i in filtered_items]
        stats["items"]["min_length"] = min(lengths)
        stats["items"]["max_length"] = max(lengths)
        stats["items"]["avg_length"] = sum(lengths) / len(lengths)
        stats["items"]["shortest"] = min(unique_items, key=len)
        stats["items"]["longest"] = max(unique_items, key=len)

    # 2. Collect Pairs if requested
    if include_pairs:
        raw_pairs = list(_extract_pairs(input_files, quiet=quiet))
        filtered_pairs = []
        for left, right in raw_pairs:
            if clean_items:
                left = filter_to_letters(left)
                right = filter_to_letters(right)
            if not left and not right:
                continue
            if min_length <= len(left) <= max_length and min_length <= len(right) <= max_length:
                filtered_pairs.append((left, right))

        unique_pairs = set(filtered_pairs)
        typos = [p[0] for p in filtered_pairs]
        corrections = [p[1] for p in filtered_pairs]
        unique_typos = set(typos)
        unique_corrections = set(corrections)

        # Conflicts: 1 typo -> multiple unique corrections
        typo_to_corr = defaultdict(set)
        for t, c in filtered_pairs:
            typo_to_corr[t].add(c)
        conflicts = [t for t, cs in typo_to_corr.items() if len(cs) > 1]

        # Overlaps: word is both a typo and a correction
        overlaps = unique_typos & unique_corrections

        # Character changes
        distances = [levenshtein_distance(p[0], p[1]) for p in filtered_pairs]

        stats["pairs"] = {
            "total_extracted": len(raw_pairs),
            "total_filtered": len(filtered_pairs),
            "unique_pairs": len(unique_pairs),
            "unique_typos": len(unique_typos),
            "unique_corrections": len(unique_corrections),
            "conflicts": len(conflicts),
            "overlaps": len(overlaps),
        }

        if distances:
            stats["pairs"]["min_dist"] = min(distances)
            stats["pairs"]["max_dist"] = max(distances)
            stats["pairs"]["avg_dist"] = sum(distances) / len(distances)

    # 3. Output
    if output_format == 'json':
        with smart_open_output(output_file) as f:
            json.dump(stats, f, indent=2)
            f.write('\n')
    elif output_format == 'yaml':
        with smart_open_output(output_file) as f:
            try:
                import yaml
                yaml.dump(stats, f, default_flow_style=False)
            except ImportError:
                # Basic fallback
                f.write("items:\n")
                for k, v in stats["items"].items():
                    f.write(f"  {k}: {v}\n")
                if "pairs" in stats:
                    f.write("pairs:\n")
                    for k, v in stats["pairs"].items():
                        f.write(f"  {k}: {v}\n")
    elif output_format in ('markdown', 'md-table'):
        with smart_open_output(output_file) as f:
            f.write("### ANALYSIS SUMMARY\n\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            f.write(f"| Total items encountered | {stats['items']['total_encountered']} |\n")
            f.write(f"| Total items after filtering | {stats['items']['total_filtered']} |\n")
            f.write(f"| Unique items | {stats['items']['unique_count']} |\n")
            if "min_length" in stats["items"]:
                f.write(f"| Min length | {stats['items']['min_length']} |\n")
                f.write(f"| Max length | {stats['items']['max_length']} |\n")
                f.write(f"| Avg length | {stats['items']['avg_length']:.1f} |\n")

            if "pairs" in stats:
                f.write("\n### PAIRED DATA STATISTICS\n\n")
                f.write("| Metric | Value |\n")
                f.write("| :--- | :--- |\n")
                f.write(f"| Total pairs extracted | {stats['pairs']['total_extracted']} |\n")
                f.write(f"| Total pairs after filtering | {stats['pairs']['total_filtered']} |\n")
                f.write(f"| Unique pairs | {stats['pairs']['unique_pairs']} |\n")
                f.write(f"| Unique typos / corrections | {stats['pairs']['unique_typos']} / {stats['pairs']['unique_corrections']} |\n")
                f.write(f"| Conflicts (1 typo -> N corr) | {stats['pairs']['conflicts']} |\n")
                f.write(f"| Overlaps (typo == correction) | {stats['pairs']['overlaps']} |\n")
                if "min_dist" in stats["pairs"]:
                    f.write(f"| Min character changes | {stats['pairs']['min_dist']} |\n")
                    f.write(f"| Max character changes | {stats['pairs']['max_dist']} |\n")
                    f.write(f"| Avg character changes | {stats['pairs']['avg_dist']:.1f} |\n")
    else:
        # Human readable text
        with smart_open_output(output_file) as f:
            # Colors for output stream
            c_bold = BOLD if f.isatty() else ""
            c_green = GREEN if f.isatty() else ""
            c_yellow = YELLOW if f.isatty() else ""
            c_reset = RESET if f.isatty() else ""

            report = []
            padding = "  "
            report.append(f"\n{padding}{c_bold}ANALYSIS SUMMARY{c_reset}")
            report.append(f"{padding}{c_bold}───────────────────────────────────────────────────────{c_reset}")

            label_width = 35
            report.append(f"  {c_bold}{'Total items encountered:':<{label_width}}{c_reset} {c_yellow}{stats['items']['total_encountered']}{c_reset}")
            report.append(f"  {c_bold}{'Total items after filtering:':<{label_width}}{c_reset} {c_green}{stats['items']['total_filtered']}{c_reset}")

            if stats['items']['total_encountered'] > 0:
                retention = (stats['items']['total_filtered'] / stats['items']['total_encountered']) * 100
                report.append(f"  {c_bold}{'Retention rate:':<{label_width}}{c_reset} {c_green}{retention:.1f}%{c_reset}")

            report.append(f"  {c_bold}{'Unique items:':<{label_width}}{c_reset} {stats['items']['unique_count']}")

            if "min_length" in stats["items"]:
                report.append(f"  {c_bold}{'Min/Max/Avg length:':<{label_width}}{c_reset} {stats['items']['min_length']} / {stats['items']['max_length']} / {stats['items']['avg_length']:.1f}")
                shortest = stats["items"]["shortest"]
                longest = stats["items"]["longest"]
                report.append(f"  {c_bold}{'Shortest item:':<{label_width}}{c_reset} '{shortest}' (length: {len(shortest)})")
                report.append(f"  {c_bold}{'Longest item:':<{label_width}}{c_reset} '{longest}' (length: {len(longest)})")

            duration = time.perf_counter() - start_time
            report.append(
                f"  {c_bold}{'Processing time:':<{label_width}}{c_reset} {c_green}{duration:.3f}s{c_reset}"
            )

            if "pairs" in stats:
                report.append(f"\n{padding}{c_bold}PAIRED DATA STATISTICS{c_reset}")
                report.append(f"{padding}{c_bold}───────────────────────────────────────────────────────{c_reset}")
                report.append(f"  {c_bold}{'Total pairs extracted:':<{label_width}}{c_reset} {c_yellow}{stats['pairs']['total_extracted']}{c_reset}")
                report.append(f"  {c_bold}{'Total pairs after filtering:':<{label_width}}{c_reset} {c_green}{stats['pairs']['total_filtered']}{c_reset}")

                if stats['pairs']['total_extracted'] > 0:
                    retention = (stats['pairs']['total_filtered'] / stats['pairs']['total_extracted']) * 100
                    report.append(f"  {c_bold}{'Retention rate:':<{label_width}}{c_reset} {c_green}{retention:.1f}%{c_reset}")

                report.append(f"  {c_bold}{'Unique pairs:':<{label_width}}{c_reset} {stats['pairs']['unique_pairs']}")
                report.append(f"  {c_bold}{'Unique typos / corrections:':<{label_width}}{c_reset} {stats['pairs']['unique_typos']} / {stats['pairs']['unique_corrections']}")
                report.append(f"  {c_bold}{'Conflicts (1 typo -> N corr):':<{label_width}}{c_reset} {stats['pairs']['conflicts']}")
                report.append(f"  {c_bold}{'Overlaps (typo == correction):':<{label_width}}{c_reset} {stats['pairs']['overlaps']}")
                if "min_dist" in stats["pairs"]:
                    report.append(f"  {c_bold}{'Min/Max/Avg changes:':<{label_width}}{c_reset} {stats['pairs']['min_dist']} / {stats['pairs']['max_dist']} / {stats['pairs']['avg_dist']:.1f}")

            report.append("")
            f.write("\n".join(report))

    logging.info(f"[Stats Mode] Analysis complete. Summary written to '{output_file}'.")


def check_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """
    Checks CSV file(s) of typos and corrections for any words that appear
    as both a typo and a correction anywhere in the dataset.
    """
    start_time = time.perf_counter()
    typos = set()
    corrections = set()

    for input_file in input_files:
        lines = _read_file_lines_robust(input_file, newline='')
        reader = csv.reader(lines)
        for row in tqdm(reader, desc=f'Checking {input_file}', unit=' rows', disable=quiet):
            if not row:
                continue
            typos.add(row[0].strip())
            for field in row[1:]:
                corrections.add(field.strip())

    duplicates = list(typos & corrections)
    filtered_items = clean_and_filter(duplicates, min_length, max_length, clean=clean_items)

    if process_output:
        filtered_items = list(set(filtered_items))
    filtered_items.sort()

    write_output(filtered_items, output_file, output_format, quiet, limit=limit)

    print_processing_stats(
        len(duplicates), filtered_items, start_time=start_time
    )
    logging.info(
        f"[Check Mode] Found {len(filtered_items)} overlapping words across {len(input_files)} file(s). Output written to '{output_file}'."
    )


def conflict_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """
    Identifies typos that are associated with more than one unique correction.
    """
    start_time = time.perf_counter()
    raw_pairs = _extract_pairs(input_files, quiet=quiet)
    typo_to_corrections = defaultdict(set)

    for left, right in raw_pairs:
        # Apply cleaning if requested
        if clean_items:
            left = filter_to_letters(left)
            right = filter_to_letters(right)

        # Apply length filtering to both sides to ensure valid data pairs
        if min_length <= len(left) <= max_length and min_length <= len(right) <= max_length:
            typo_to_corrections[left].add(right)

    conflicts = []
    for typo, corrections in typo_to_corrections.items():
        if len(corrections) > 1:
            conflicts.append((typo, ", ".join(sorted(corrections))))

    if process_output:
        conflicts.sort()

    _write_paired_output(
        conflicts,
        output_file,
        output_format,
        "Conflict",
        quiet,
        limit=limit
    )

    print_processing_stats(
        len(conflicts), conflicts, item_label="conflict", start_time=start_time
    )
    logging.info(f"[Conflict Mode] Found {len(conflicts)} typos with conflicting corrections. Output written to '{output_file}'.")


def cycles_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """
    Identifies circular references in typo-correction pairs.
    """
    start_time = time.perf_counter()
    raw_pairs = _extract_pairs(input_files, quiet=quiet)
    adj = defaultdict(set)

    for left, right in raw_pairs:
        if clean_items:
            left = filter_to_letters(left)
            right = filter_to_letters(right)

        if min_length <= len(left) <= max_length and min_length <= len(right) <= max_length:
            adj[left].add(right)

    cycles = []
    visited = set()
    found_normalized_cycles = set()

    for start_node in sorted(adj.keys()):
        if start_node not in visited:
            # Re-initialize path tracking for each new component search
            path_set = set()
            path_list = []

            def walk(node):
                if node in path_set:
                    # Found a cycle! Extract it from the current path.
                    idx = path_list.index(node)
                    cycle_nodes = path_list[idx:]
                    
                    # Normalize the cycle to avoid duplicates (for example, a->b->a and b->a->b)
                    # We use the lexicographically smallest rotation as the representative.
                    min_node = min(cycle_nodes)
                    min_idx = cycle_nodes.index(min_node)
                    normalized = tuple(cycle_nodes[min_idx:] + cycle_nodes[:min_idx])
                    
                    if normalized not in found_normalized_cycles:
                        found_normalized_cycles.add(normalized)
                        # Format as a chain: a -> b -> a
                        chain = " -> ".join(list(normalized) + [normalized[0]])
                        cycles.append((normalized[0], chain))
                    return

                if node in visited:
                    # Already explored this node. In some cases we might want to re-explore 
                    # to find all cycles, but for typo detection, visiting each node once 
                    # in the DFS tree is a good balance between discovery and performance.
                    return

                visited.add(node)
                path_set.add(node)
                path_list.append(node)

                # Sort neighbors to ensure deterministic behavior
                for next_node in sorted(adj.get(node, set())):
                    walk(next_node)

                # Unwind path tracking for the current branch
                path_list.pop()
                path_set.remove(node)

            walk(start_node)

    if process_output:
        cycles.sort()

    _write_paired_output(
        cycles,
        output_file,
        output_format,
        "Cycles",
        quiet,
        limit=limit
    )

    print_processing_stats(
        len(cycles), cycles, item_label="cycle", start_time=start_time
    )
    logging.info(f"[Cycles Mode] Found {len(cycles)} circular dependencies. Output written to '{output_file}'.")


def similarity_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    min_dist: int = 0,
    max_dist: int | None = None,
    show_dist: bool = False,
    output_format: str = 'arrow',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """
    Filters paired data based on the number of character changes between words.
    """
    start_time = time.perf_counter()
    raw_pairs = _extract_pairs(input_files, quiet=quiet)

    filtered_results = []
    for left, right in raw_pairs:
        if clean_items:
            left = filter_to_letters(left)
            right = filter_to_letters(right)

        if not left or not right:
            continue

        # Apply length filtering
        if not (min_length <= len(left) <= max_length and min_length <= len(right) <= max_length):
            continue

        dist = levenshtein_distance(left, right)

        if dist < min_dist:
            continue
        if max_dist is not None and dist > max_dist:
            continue

        if show_dist:
            # Append number of changes to the right side for display
            filtered_results.append((left, f"{right} (changes: {dist})"))
        else:
            filtered_results.append((left, right))

    if process_output:
        filtered_results = sorted(set(filtered_results))

    _write_paired_output(
        filtered_results,
        output_file,
        output_format,
        "Similarity",
        quiet,
        limit=limit
    )

    stats_items = []
    for left, right_with_dist in filtered_results:
        # Strip "(changes: N)" from right side
        base_right = right_with_dist.rsplit(' (changes: ', 1)[0]
        stats_items.append((left, base_right))

    print_processing_stats(
        len(filtered_results), stats_items, item_label="similar-pair", start_time=start_time
    )


def near_duplicates_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    min_dist: int = 1,
    max_dist: int = 1,
    show_dist: bool = False,
    output_format: str = 'arrow',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """
    Finds pairs of words in a single list that are similar to each other.
    """
    start_time = time.perf_counter()
    raw_item_count = 0
    all_unique_items = []

    for file_path in input_files:
        raw, _, unique = _load_and_clean_file(
            file_path,
            min_length,
            max_length,
            clean_items=clean_items,
        )
        raw_item_count += len(raw)
        all_unique_items.extend(unique)

    # Re-deduplicate across all input files
    unique_items = sorted(set(all_unique_items))
    # Sort by length for optimized comparison
    unique_items.sort(key=len)

    results = []
    num_items = len(unique_items)

    for i in tqdm(range(num_items), desc="Finding near-duplicates", unit="word", disable=quiet):
        word_i = unique_items[i]
        len_i = len(word_i)

        for j in range(i + 1, num_items):
            word_j = unique_items[j]
            len_j = len(word_j)

            # Optimization: words are sorted by length, so we can stop if length difference is too large
            if len_j - len_i > max_dist:
                break

            dist = levenshtein_distance(word_i, word_j)

            if min_dist <= dist <= max_dist:
                if show_dist:
                    results.append((word_i, f"{word_j} (changes: {dist})"))
                else:
                    results.append((word_i, word_j))

    if process_output:
        results.sort()

    _write_paired_output(
        results,
        output_file,
        output_format,
        "NearDuplicates",
        quiet,
        limit=limit
    )

    stats_items = []
    for left, right_with_dist in results:
        base_right = right_with_dist.rsplit(' (changes: ', 1)[0]
        stats_items.append((left, base_right))

    print_processing_stats(
        raw_item_count, stats_items, item_label="near-duplicate", start_time=start_time
    )


def fuzzymatch_mode(
    input_files: Sequence[str],
    file2: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    min_dist: int = 1,
    max_dist: int = 1,
    show_dist: bool = False,
    output_format: str = 'arrow',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """
    Finds pairs of words between two lists that are similar to each other.
    """
    start_time = time.perf_counter()
    raw_item_count = 0
    list1_unique = []

    for file_path in input_files:
        raw, _, unique = _load_and_clean_file(
            file_path,
            min_length,
            max_length,
            clean_items=clean_items,
        )
        raw_item_count += len(raw)
        list1_unique.extend(unique)

    list1_unique = sorted(set(list1_unique))

    raw_items_b, _, list2_unique = _load_and_clean_file(
        file2,
        min_length,
        max_length,
        clean_items=clean_items,
    )
    raw_item_count += len(raw_items_b)

    # Sort list2 by length for optimized comparison
    list2_unique = sorted(set(list2_unique), key=len)

    results = []

    for word_i in tqdm(list1_unique, desc="Fuzzy matching", disable=quiet):
        len_i = len(word_i)

        for word_j in list2_unique:
            len_j = len(word_j)

            # Optimization: stop if length difference is too large
            if len_j < len_i - max_dist:
                continue
            if len_j > len_i + max_dist:
                break

            dist = levenshtein_distance(word_i, word_j)

            if min_dist <= dist <= max_dist:
                if show_dist:
                    results.append((word_i, f"{word_j} (changes: {dist})"))
                else:
                    results.append((word_i, word_j))

    if process_output:
        results.sort()

    _write_paired_output(
        results,
        output_file,
        output_format,
        "FuzzyMatch",
        quiet,
        limit=limit
    )

    stats_items = []
    for left, right_with_dist in results:
        # Strip "(changes: N)" from right side
        base_right = right_with_dist.rsplit(' (changes: ', 1)[0]
        stats_items.append((left, base_right))

    print_processing_stats(
        raw_item_count, stats_items, item_label="fuzzy-match", start_time=start_time
    )


def casing_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    delimiter: str | None = None,
    smart: bool = False,
    output_format: str = 'arrow',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """
    Identifies words that appear with inconsistent capitalization.
    """
    start_time = time.perf_counter()
    raw_item_count = 0
    # Map normalized word -> set of original forms
    normalized_to_original = defaultdict(set)

    for input_file in input_files:
        words = _extract_words_items(input_file, delimiter=delimiter, quiet=quiet, smart=smart)
        for word in words:
            raw_item_count += 1
            # Normalize for grouping
            norm = filter_to_letters(word) if clean_items else word.lower()
            if not norm:
                continue

            # Apply length filtering
            if min_length <= len(norm) <= max_length:
                normalized_to_original[norm].add(word)

    conflicts = []
    for norm, originals in normalized_to_original.items():
        if len(originals) > 1:
            conflicts.append((norm, ", ".join(sorted(originals))))

    if process_output:
        conflicts.sort()

    _write_paired_output(
        conflicts,
        output_file,
        output_format,
        "Casing",
        quiet,
        limit=limit
    )

    print_processing_stats(
        raw_item_count,
        [c[0] for c in conflicts],
        item_label="casing-conflict",
        start_time=start_time,
    )


def diff_mode(
    input_files: Sequence[str],
    file2: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    pairs: bool = False,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """
    Identifies added, removed, and changed items between two files or lists.
    """
    start_time = time.perf_counter()
    if pairs:
        # Load pairs from both sources
        left_pairs = dict(_extract_pairs(input_files, quiet=quiet))
        right_pairs = dict(_extract_pairs([file2], quiet=quiet))

        if clean_items:
            left_pairs = {filter_to_letters(k): filter_to_letters(v) for k, v in left_pairs.items()}
            right_pairs = {filter_to_letters(k): filter_to_letters(v) for k, v in right_pairs.items()}

        # Filter by length
        left_pairs = {k: v for k, v in left_pairs.items()
                      if min_length <= len(k) <= max_length and min_length <= len(v) <= max_length}
        right_pairs = {k: v for k, v in right_pairs.items()
                       if min_length <= len(k) <= max_length and min_length <= len(v) <= max_length}

        left_keys = set(left_pairs.keys())
        right_keys = set(right_pairs.keys())

        added_keys = right_keys - left_keys
        removed_keys = left_keys - right_keys
        common_keys = left_keys & right_keys

        changed = []
        for k in sorted(common_keys):
            if left_pairs[k] != right_pairs[k]:
                changed.append((k, f"{left_pairs[k]} -> {right_pairs[k]}"))

        added = sorted([(k, right_pairs[k]) for k in added_keys])
        removed = sorted([(k, left_pairs[k]) for k in removed_keys])

        # Prepare combined output
        results = []
        for k, v in removed:
            results.append(f"- {k} -> {v}")
        for k, v in added:
            results.append(f"+ {k} -> {v}")
        for k, v in changed:
            results.append(f"~ {k}: {v}")

    else:
        # Load items from both sources
        _, _, left_items = _load_and_clean_file(
            input_files[0] if input_files else '-',
            min_length,
            max_length,
            clean_items=clean_items
        )
        # Handle multiple input files by merging them for the "left" side
        if len(input_files) > 1:
            for f in input_files[1:]:
                _, _, extra = _load_and_clean_file(f, min_length, max_length, clean_items=clean_items)
                left_items.extend(extra)
            left_items = sorted(set(left_items))

        _, _, right_items = _load_and_clean_file(
            file2,
            min_length,
            max_length,
            clean_items=clean_items
        )

        left_set = set(left_items)
        right_set = set(right_items)

        added = sorted(right_set - left_set)
        removed = sorted(left_set - right_set)

        results = [f"- {item}" for item in removed] + [f"+ {item}" for item in added]

    if limit is not None:
        results = results[:limit]

    # Handle output
    with smart_open_output(output_file) as out:
        if output_format == 'json':
            # Count how many items from each category were kept after the limit
            rem_limit = sum(1 for r in results if r.startswith('- '))
            add_limit = sum(1 for r in results if r.startswith('+ '))
            if pairs:
                chg_limit = sum(1 for r in results if r.startswith('~ '))
                diff_data = {
                    "added": {k: v for k, v in added[:add_limit]},
                    "removed": {k: v for k, v in removed[:rem_limit]},
                    "changed": {k: v for k, v in changed[:chg_limit]}
                }
            else:
                diff_data = {
                    "added": added[:add_limit],
                    "removed": removed[:rem_limit]
                }
            json.dump(diff_data, out, indent=2)
            out.write('\n')
        else:
            # Terminal/Line output with colors
            c_red = RED if out.isatty() else ""
            c_green = GREEN if out.isatty() else ""
            c_yellow = YELLOW if out.isatty() else ""
            c_reset = RESET if out.isatty() else ""

            for line in results:
                if line.startswith('+'):
                    out.write(f"{c_green}{line}{c_reset}\n")
                elif line.startswith('-'):
                    out.write(f"{c_red}{line}{c_reset}\n")
                elif line.startswith('~'):
                    out.write(f"{c_yellow}{line}{c_reset}\n")

    duration = time.perf_counter() - start_time
    logging.info(
        f"[Diff Mode] Comparison complete. Output written to '{output_file}'. "
        f"Processing time: {duration:.3f}s"
    )


def repeated_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    delimiter: str | None = None,
    smart: bool = False,
    output_format: str = 'arrow',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """
    Identifies consecutive identical words.
    """
    start_time = time.perf_counter()
    results = list(_extract_repeated_items(
        input_files,
        delimiter=delimiter,
        quiet=quiet,
        smart=smart,
        clean_items=clean_items,
        min_length=min_length,
        max_length=max_length
    ))

    if process_output:
        results = sorted(set(results))

    _write_paired_output(
        results,
        output_file,
        output_format,
        "Repeated",
        quiet,
        limit=limit
    )

    print_processing_stats(
        len(results), results, item_label="repeated-word", start_time=start_time
    )


def discovery_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    rare_max: int = 1,
    freq_min: int = 5,
    min_dist: int = 1,
    max_dist: int = 1,
    show_dist: bool = False,
    output_format: str = 'arrow',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """
    Identifies potential typos by comparing rare words to frequent words.
    """
    start_time = time.perf_counter()
    word_counts = Counter()
    raw_item_count = 0

    for input_file in input_files:
        lines = _read_file_lines_robust(input_file)
        for line in tqdm(lines, desc=f'Analyzing frequencies in {input_file}', unit=' lines', disable=quiet):
            words = line.split()
            raw_item_count += len(words)
            filtered = clean_and_filter(words, min_length, max_length, clean=clean_items)
            word_counts.update(filtered)

    # Identify rare and frequent words
    rare_words = sorted([word for word, count in word_counts.items() if count <= rare_max])
    frequent_words = sorted([word for word, count in word_counts.items() if count >= freq_min], key=len)

    results = []
    for rare in tqdm(rare_words, desc="Finding likely corrections", unit="word", disable=quiet):
        len_rare = len(rare)
        for freq in frequent_words:
            len_freq = len(freq)
            # Optimization: words are sorted by length, so we can stop if length difference is too large
            if len_freq < len_rare - max_dist:
                continue
            if len_freq > len_rare + max_dist:
                break

            dist = levenshtein_distance(rare, freq)
            if min_dist <= dist <= max_dist:
                if show_dist:
                    results.append((rare, f"{freq} (changes: {dist})"))
                else:
                    results.append((rare, freq))

    if process_output:
        results.sort()

    _write_paired_output(
        results,
        output_file,
        output_format,
        "Discovery",
        quiet,
        limit=limit
    )

    stats_items = []
    for left, right_with_dist in results:
        # Strip "(changes: N)" from right side
        base_right = right_with_dist.rsplit(' (changes: ', 1)[0]
        stats_items.append((left, base_right))

    print_processing_stats(
        raw_item_count, stats_items, item_label="discovered-typo", start_time=start_time
    )


def csv_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    first_column: bool = False,
    delimiter: str = ',',
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
    columns: List[int] | None = None,
) -> None:
    """Wrapper for extracting fields from CSV files."""
    def extractor(f, quiet=False):
        return _extract_csv_items(
            f, first_column, delimiter, quiet=quiet, columns=columns
        )
    _process_items(
        extractor,
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        'CSV',
        'CSV fields extracted successfully.',
        output_format,
        quiet,
        clean_items=clean_items,
        limit=limit,
    )


def line_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """Wrapper for processing raw lines from file(s)."""
    _process_items(
        _extract_line_items,
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        'Line',
        'Lines processed successfully.',
        output_format,
        quiet,
        clean_items=clean_items,
        limit=limit,
    )


def words_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    delimiter: str | None = None,
    smart: bool = False,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """Wrapper for extracting individual words from file(s)."""
    def extractor(f, quiet=False):
        return _extract_words_items(f, delimiter=delimiter, quiet=quiet, smart=smart)
    _process_items(
        extractor,
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        'Words',
        'Words extracted successfully.',
        output_format,
        quiet,
        clean_items=clean_items,
        limit=limit,
    )


def combine_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """Merge cleaned contents from multiple files into one deduplicated list."""
    start_time = time.perf_counter()
    raw_item_count = 0
    combined_unique: list[str] = []

    for file_path in input_files:
        raw_items, cleaned_items, unique_items = _load_and_clean_file(
            file_path,
            min_length,
            max_length,
            clean_items=clean_items,
        )
        raw_item_count += len(raw_items)
        combined_unique.extend(unique_items)

    combined_unique = sorted(dict.fromkeys(combined_unique))

    write_output(combined_unique, output_file, output_format, quiet, limit=limit)

    print_processing_stats(raw_item_count, combined_unique, start_time=start_time)
    logging.info(
        "[Combine Mode] Combined %d file(s). Output written to '%s'.",
        len(input_files),
        output_file,
    )


def unique_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """Deduplicate items while preserving their first appearance in the input files."""
    start_time = time.perf_counter()
    raw_item_count = 0
    combined_unique: list[str] = []

    for file_path in input_files:
        raw_items, cleaned_items, unique_items = _load_and_clean_file(
            file_path,
            min_length,
            max_length,
            clean_items=clean_items,
        )
        raw_item_count += len(raw_items)
        combined_unique.extend(unique_items)

    # Use dict.fromkeys() to deduplicate while preserving order of first occurrence
    final_items = list(dict.fromkeys(combined_unique))

    if process_output:
        # If the user explicitly requested -P, we still sort alphabetically.
        # But by default unique mode is order-preserving.
        final_items.sort()

    write_output(final_items, output_file, output_format, quiet, limit=limit)

    print_processing_stats(raw_item_count, final_items, start_time=start_time)
    logging.info(
        "[Unique Mode] Deduplicated %d file(s). Output written to '%s'.",
        len(input_files),
        output_file,
    )


def zip_mode(
    input_files: Sequence[str],
    file2: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    output_format: str = 'arrow',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """Combines items from input_files and file2 line-by-line into a paired format."""
    start_time = time.perf_counter()

    def get_cleaned_lines(path: str) -> List[str]:
        lines = _read_file_lines_robust(path)
        cleaned = []
        for line in lines:
            item = line.strip()
            if clean_items:
                item = filter_to_letters(item)
            cleaned.append(item)
        return cleaned

    # Merge all input files for the left side
    left_items = []
    for f in input_files:
        left_items.extend(get_cleaned_lines(f))

    # Read file2 for the right side
    right_items = get_cleaned_lines(file2)

    raw_pairs = list(zip(left_items, right_items))

    # Filter pairs
    filtered_pairs = []
    for left, right in raw_pairs:
        # Skip if either side is empty after cleaning
        if not left or not right:
            continue
        # Apply length filtering to BOTH sides to ensure they meet criteria
        if min_length <= len(left) <= max_length and min_length <= len(right) <= max_length:
            filtered_pairs.append((left, right))

    if process_output:
        # Deduplicate while preserving order if not sorting? No, sorted(set()) sorts.
        filtered_pairs = sorted(set(filtered_pairs))

    _write_paired_output(
        filtered_pairs,
        output_file,
        output_format,
        "Zip",
        quiet,
        limit=limit
    )

    print_processing_stats(
        len(raw_pairs), filtered_pairs, item_label="zipped-pair", start_time=start_time
    )


def pairs_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    output_format: str = 'arrow',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """Processes paired data from input file(s)."""
    start_time = time.perf_counter()

    raw_pairs = _extract_pairs(input_files, quiet=quiet)

    filtered_pairs = []
    raw_count = 0
    for left, right in raw_pairs:
        raw_count += 1
        # Clean if requested
        if clean_items:
            left = filter_to_letters(left)
            right = filter_to_letters(right)

        # Skip if either side is empty after cleaning
        if not left or not right:
            continue

        # Apply length filtering to both sides to ensure valid data pairs
        if min_length <= len(left) <= max_length and min_length <= len(right) <= max_length:
            filtered_pairs.append((left, right))

    if process_output:
        filtered_pairs = sorted(set(filtered_pairs))

    _write_paired_output(
        filtered_pairs,
        output_file,
        output_format,
        "Pairs",
        quiet,
        limit=limit
    )

    print_processing_stats(
        raw_count, filtered_pairs, item_label="pair", start_time=start_time
    )


def swap_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    output_format: str = 'arrow',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """Reverses the order of pairs in the input file(s)."""
    start_time = time.perf_counter()

    raw_pairs = _extract_pairs(input_files, quiet=quiet)

    filtered_pairs = []
    raw_count = 0
    for left, right in raw_pairs:
        raw_count += 1
        # Swap
        new_left, new_right = right, left

        # Clean if requested
        if clean_items:
            new_left = filter_to_letters(new_left)
            new_right = filter_to_letters(new_right)

        # Skip if either side is empty after cleaning
        if not new_left or not new_right:
            continue

        # Apply length filtering
        if min_length <= len(new_left) <= max_length and min_length <= len(new_right) <= max_length:
            filtered_pairs.append((new_left, new_right))

    if process_output:
        filtered_pairs = sorted(set(filtered_pairs))

    _write_paired_output(
        filtered_pairs,
        output_file,
        output_format,
        "Swap",
        quiet,
        limit=limit
    )

    print_processing_stats(
        raw_count, filtered_pairs, item_label="swapped-pair", start_time=start_time
    )


def sample_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    sample_count: int | None = None,
    sample_percent: float | None = None,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """Randomly sample lines from the input file(s)."""
    start_time = time.perf_counter()

    # Extract raw items first
    raw_items = [
        item for input_file in input_files
        for item in _extract_line_items(input_file, quiet=quiet)
    ]

    if not raw_items:
        logging.warning("Input is empty or no lines found.")
        # Create empty output using write_output to ensure consistent formatting (for example empty JSON list)
        write_output([], output_file, output_format, quiet)
        return

    # Clean and filter BEFORE sampling to ensure the requested count is accurate relative to valid items
    cleaned_items = clean_and_filter(raw_items, min_length, max_length, clean=clean_items)

    total_valid_items = len(cleaned_items)

    if sample_count is not None:
        k = min(sample_count, total_valid_items)
    elif sample_percent is not None:
        k = int(total_valid_items * (sample_percent / 100.0))
        k = max(0, min(k, total_valid_items))
    else:
        k = total_valid_items

    sampled_items = random.sample(cleaned_items, k)

    if process_output:
        sampled_items = sorted(set(sampled_items))

    write_output(sampled_items, output_file, output_format, quiet, limit=limit)

    print_processing_stats(len(raw_items), sampled_items, start_time=start_time)
    logging.info(
        f"[Sample Mode] Sampled {k}/{total_valid_items} valid lines from {len(input_files)} file(s). Output written to '{output_file}'."
    )


def regex_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    pattern: str,
    output_format: str = 'line',
    quiet: bool = False,
    limit: int | None = None,
) -> None:
    """Wrapper for extracting text matching a regex pattern."""
    # Regex mode skips the default 'clean_and_filter' (to lower, letters only)
    # because users often want exact matches (for example Emails, URLs, IDs).
    # Users can still use --process-output to sort/dedup, but we don't force lowercase/clean.
    def extractor(f, quiet=False):
        return _extract_regex_items(f, pattern, quiet=quiet)
    _process_items(
        extractor,
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        'Regex',
        'Regex matches extracted successfully.',
        output_format,
        quiet,
        clean_items=False,
        limit=limit,
    )


def _load_mapping_file(path: str, quiet: bool = False) -> dict[str, str]:
    """Load a mapping file into a dictionary, supporting Arrow, CSV, Table, JSON, and YAML."""
    # Use the shared _extract_pairs helper to handle all supported formats consistently
    return dict(_extract_pairs([path], quiet=quiet))


def map_mode(
    input_files: Sequence[str],
    mapping_file: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    drop_missing: bool = False,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
    pairs: bool = False,
    smart_case: bool = False,
) -> None:
    """
    Transforms items based on a mapping file.
    """
    start_time = time.perf_counter()
    # Load mapping
    raw_mapping = _load_mapping_file(mapping_file, quiet=quiet)

    # If clean_items is True, we should also clean the mapping keys to match input
    if clean_items:
        cleaned_mapping = {}
        for k, v in raw_mapping.items():
            cleaned_k = filter_to_letters(k)
            # We assume value is the replacement and should be used as is?
            # Or should value also be cleaned?
            # If I map "TeH" -> "The", and input is "teh".
            # Cleaned input: "teh". Cleaned key: "teh". Value: "The".
            # Output: "The".
            # If output is also cleaned later (for example by process_output)?
            # But here we are producing the item.
            # Let's keep value as is, but clean key.
            if cleaned_k:
                cleaned_mapping[cleaned_k] = v
        mapping = cleaned_mapping
    else:
        mapping = raw_mapping

    raw_item_count = 0
    results = []

    for input_file in input_files:
        # We manually iterate to keep raw and cleaned synchronized for smart casing
        lines = _read_file_lines_robust(input_file)
        for line in lines:
            line_content = line.strip()
            if not line_content:
                continue

            # Default behavior of map_mode is processing the whole line item.
            parts = [line_content]
            for part in parts:
                raw_item_count += 1
                match_key = filter_to_letters(part) if clean_items else part

                if match_key in mapping:
                    transformed = mapping[match_key]
                    if smart_case:
                        transformed = _apply_smart_case(part, transformed)

                    # Re-apply length filtering to the result of the mapping
                    if transformed and min_length <= len(transformed) <= max_length:
                        results.append((part, transformed) if pairs else transformed)
                elif not drop_missing:
                    if part and min_length <= len(part) <= max_length:
                        results.append((part, part) if pairs else part)

    if process_output:
        results = sorted(set(results))

    if pairs:
        _write_paired_output(results, output_file, output_format, "Map", quiet, limit=limit)
    else:
        write_output(results, output_file, output_format, quiet, limit=limit)

    # For stats, if pairs, use the transformed side
    stats_items = [r[1] if isinstance(r, tuple) else r for r in results]
    print_processing_stats(
        raw_item_count, stats_items, item_label="item", start_time=start_time
    )
    logging.info(
        f"[Map Mode] Transformed items using '{mapping_file}'. Output written to '{output_file}'."
    )


def scrub_mode(
    input_files: Sequence[str],
    mapping_file: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
    in_place: str | None = None,
    dry_run: bool = False,
    smart_case: bool = False,
) -> None:
    """
    Performs replacements of typos in text files based on a mapping file.
    Supports in-place modification and dry-run preview.
    """
    start_time = time.perf_counter()
    # Load mapping
    raw_mapping = _load_mapping_file(mapping_file, quiet=quiet)

    # Clean the mapping keys for matching if clean_items is True
    if clean_items:
        mapping = {filter_to_letters(k): v for k, v in raw_mapping.items() if filter_to_letters(k)}
    else:
        mapping = raw_mapping

    total_replacements = 0
    # Pattern for splitting lines into words and non-words (delimiters)
    # This ensures we preserve whitespace and punctuation exactly.
    pattern = re.compile(r'([a-zA-Z0-9]+)')

    # If in_place, we process each file individually.
    # Otherwise, we accumulate and write to output_file.
    accumulated_lines = []

    for input_file in input_files:
        if input_file == '-' and in_place is not None:
            logging.warning("In-place modification requested for standard input; ignoring.")

        file_lines = _read_file_lines_robust(input_file)
        modified_lines = []
        file_replacements = 0

        for line in tqdm(file_lines, desc=f"Scrubbing {input_file}", unit=" lines", disable=quiet):
            # Split into words and non-words
            parts = pattern.split(line)
            new_parts = []
            for part in parts:
                if not part:
                    continue

                if pattern.match(part):
                    # It's a word candidate.
                    # If the whole word matches a typo (after optional cleaning)
                    match_key = filter_to_letters(part) if clean_items else part

                    if match_key in mapping:
                        replacement = mapping[match_key]
                        if smart_case:
                            replacement = _apply_smart_case(part, replacement)
                        new_parts.append(replacement)
                        file_replacements += 1
                    else:
                        # Try subword replacement if the whole word didn't match.
                        sub_parts = _smart_split(part)
                        new_sub_parts = []
                        for sp in sub_parts:
                            sm_key = filter_to_letters(sp) if clean_items else sp
                            if sm_key in mapping:
                                replacement = mapping[sm_key]
                                if smart_case:
                                    replacement = _apply_smart_case(sp, replacement)
                                new_sub_parts.append(replacement)
                                file_replacements += 1
                            else:
                                new_sub_parts.append(sp)
                        new_parts.append("".join(new_sub_parts))
                else:
                    # It's a delimiter (punctuation, whitespace)
                    new_parts.append(part)

            modified_lines.append("".join(new_parts))

        total_replacements += file_replacements

        if in_place is not None and input_file != '-':
            if file_replacements > 0:
                if dry_run:
                    logging.warning(f"[Dry Run] Would make {file_replacements} replacement(s) in '{input_file}'.")
                else:
                    # Backup if extension is provided
                    if in_place:
                        backup_path = input_file + in_place
                        try:
                            import shutil
                            shutil.copy2(input_file, backup_path)
                            logging.info(f"Created backup of '{input_file}' at '{backup_path}'.")
                        except Exception as e:
                            logging.error(f"Failed to create backup of '{input_file}': {e}")
                            sys.exit(1)

                    # Write in-place
                    try:
                        with open(input_file, 'w', encoding='utf-8') as f:
                            for line in modified_lines:
                                f.write(line)
                                if not line.endswith('\n'):
                                    f.write('\n')
                        logging.info(f"Updated '{input_file}' in-place ({file_replacements} replacement(s)).")
                    except Exception as e:
                        logging.error(f"Failed to write to '{input_file}': {e}")
                        sys.exit(1)
            else:
                logging.info(f"No changes needed for '{input_file}'.")
        else:
            accumulated_lines.extend(modified_lines)

    if in_place is None:
        if limit is not None:
            accumulated_lines = accumulated_lines[:limit]

        duration = time.perf_counter() - start_time
        if dry_run:
            logging.warning(f"[Dry Run] Total replacements that would be made: {total_replacements}. Processing time: {duration:.3f}s")
        else:
            with smart_open_output(output_file) as out:
                for line in accumulated_lines:
                    out.write(line)
                    if not line.endswith('\n'):
                        out.write('\n')

            logging.info(
                f"[Scrub Mode] Completed scrubbing {len(input_files)} file(s) using '{mapping_file}'. "
                f"Made {total_replacements} replacements. Output written to '{output_file}'. "
                f"Processing time: {duration:.3f}s"
            )
    elif dry_run:
        logging.warning(f"[Dry Run] Total replacements that would be made across all files: {total_replacements}")


def highlight_mode(
    input_files: Sequence[str],
    mapping_file: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
    smart: bool = False,
) -> None:
    """
    Highlights words from a mapping file or list within the input text files.
    """
    start_time = time.perf_counter()
    # Load mapping or list
    if mapping_file.lower().endswith(('.json', '.csv', '.yaml', '.yml', '.toml')):
        raw_mapping = _load_mapping_file(mapping_file, quiet=quiet)
    else:
        # Treat as a simple list of words if not a common mapping format
        lines = _read_file_lines_robust(mapping_file)
        raw_mapping = {line.strip(): "" for line in lines if line.strip()}

    # Clean the mapping keys for matching if clean_items is True
    if clean_items:
        mapping = {filter_to_letters(k): True for k in raw_mapping.keys() if filter_to_letters(k)}
    else:
        mapping = {k: True for k in raw_mapping.keys() if k}

    total_highlights = 0
    pattern = re.compile(r'([a-zA-Z0-9]+)')

    accumulated_lines = []

    for input_file in input_files:
        file_lines = _read_file_lines_robust(input_file)
        highlighted_lines = []
        file_highlights = 0

        for line in tqdm(file_lines, desc=f"Highlighting {input_file}", unit=" lines", disable=quiet):
            parts = pattern.split(line)
            new_parts = []
            for part in parts:
                if not part:
                    continue

                if pattern.match(part):
                    # It's a word candidate
                    match_key = filter_to_letters(part) if clean_items else part

                    if match_key in mapping:
                        new_parts.append(f"{YELLOW}{part}{RESET}")
                        file_highlights += 1
                    elif smart:
                        # Try subword matching
                        sub_parts = _smart_split(part)
                        new_sub_parts = []
                        for sp in sub_parts:
                            sm_key = filter_to_letters(sp) if clean_items else sp
                            if sm_key in mapping:
                                new_sub_parts.append(f"{YELLOW}{sp}{RESET}")
                                file_highlights += 1
                            else:
                                new_sub_parts.append(sp)
                        new_parts.append("".join(new_sub_parts))
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)

            highlighted_lines.append("".join(new_parts))

        total_highlights += file_highlights
        accumulated_lines.extend(highlighted_lines)

    if limit is not None:
        accumulated_lines = accumulated_lines[:limit]

    with smart_open_output(output_file) as out:
        for line in accumulated_lines:
            out.write(line)
            if not line.endswith('\n'):
                out.write('\n')

    duration = time.perf_counter() - start_time
    logging.info(
        f"[Highlight Mode] Completed highlighting {len(input_files)} file(s) using '{mapping_file}'. "
        f"Found {total_highlights} highlight(s). Output written to '{output_file}'. "
        f"Processing time: {duration:.3f}s"
    )


def _add_common_mode_arguments(
    subparser: argparse.ArgumentParser, include_process_output: bool = True, include_limit: bool = True
) -> None:
    """Attach shared CLI arguments to a mode-specific subparser."""

    # Positional input arguments stay in the default group for prominence
    subparser.add_argument(
        'input_files_pos',
        nargs='*',
        metavar='FILE',
        help="Path(s) to the input file(s). Defaults to standard input ('-') if none provided.",
    )

    # Input/Output Group
    io_group = subparser.add_argument_group(f"{BLUE}INPUT/OUTPUT OPTIONS{RESET}")
    io_group.add_argument(
        '-i', '--input',
        dest='input_files_flag',
        type=str,
        nargs='+',
        metavar='FILE',
        help="Path(s) to the input file(s) (legacy flag, supports multiple).",
    )
    io_group.add_argument(
        '-o', '--output',
        type=str,
        default=argparse.SUPPRESS,
        help="Where to save the results. Use '-' to print to the screen (default: the screen).",
    )
    io_group.add_argument(
        '-f', '--output-format', '--format',
        dest='output_format',
        choices=['line', 'json', 'csv', 'markdown', 'md-table', 'arrow', 'table', 'yaml'],
        metavar='FMT',
        default=argparse.SUPPRESS,
        help="Choose the format for the output (default: line). Choices: line, json, csv, markdown, md-table, arrow, table, yaml.",
    )
    io_group.add_argument(
        '-q', '--quiet',
        action='store_true',
        default=argparse.SUPPRESS,
        help='Suppress progress bars and informational log output.',
    )

    # Processing Configuration Group
    proc_group = subparser.add_argument_group(f"{BLUE}PROCESSING OPTIONS{RESET}")
    proc_group.add_argument(
        '-m', '--min-length',
        type=int,
        default=argparse.SUPPRESS,
        help="Skip items shorter than this (default: 3).",
    )
    proc_group.add_argument(
        '-M', '--max-length',
        type=int,
        default=argparse.SUPPRESS,
        help="Skip items longer than this (default: 1000).",
    )
    proc_group.add_argument(
        '-R', '--raw',
        action='store_true',
        default=argparse.SUPPRESS,
        help="Keep the original text. Do not change it to lowercase or remove punctuation.",
    )
    if include_limit:
        proc_group.add_argument(
            '-L', '--limit',
            type=int,
            default=argparse.SUPPRESS,
            help="Limit the number of items in the output.",
        )
    if include_process_output:
        proc_group.add_argument(
            '-P', '--process-output',
            action='store_true',
            default=argparse.SUPPRESS,
            help="Sort the list and remove duplicates.",
        )
        proc_group.add_argument(
            '--process',
            action='store_true',
            dest='process_output',
            help=argparse.SUPPRESS,
        )
    else:
        subparser.set_defaults(process_output=False)


def filter_fragments_mode(
    input_files: Sequence[str],
    file2: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """
    Filters words from input_files (list1) that do not appear as substrings of any
    word in file2 (list2).
    """
    start_time = time.perf_counter()

    # Load and merge all input files
    all_raw_list1 = []
    all_cleaned_list1 = []

    for input_file in input_files:
        raw, cleaned, _ = _load_and_clean_file(
            input_file,
            min_length,
            max_length,
            apply_length_filter=False,
            clean_items=clean_items,
        )
        all_raw_list1.extend(raw)
        all_cleaned_list1.extend(cleaned)

    _, _, unique_list2 = _load_and_clean_file(
        file2,
        min_length,
        max_length,
        split_whitespace=True,
        apply_length_filter=False,
        clean_items=clean_items,
    )

    # Aho-Corasick automaton for efficient substring matching
    if not _AHOCORASICK_AVAILABLE:
        logging.error("The 'ahocorasick' package is not installed. Install via 'pip install pyahocorasick' to use this mode.")
        sys.exit(1)

    auto = ahocorasick.Automaton()
    for keyword in all_cleaned_list1:
        auto.add_word(keyword, keyword)
    auto.make_automaton()

    matched_words = set()
    # Optimization: Skip iteration if no keywords to search for.
    # Also prevents potential issues with iterating an empty automaton in some library versions.
    if len(all_cleaned_list1) > 0:
        for item in tqdm(unique_list2, desc="Finding matches", disable=quiet):
            for end_index, keyword in auto.iter(item):
                matched_words.add(keyword)

    non_matches = [word for word in all_cleaned_list1 if word not in matched_words]
    # Items were already cleaned/processed during loading; only length filtering is needed now.
    filtered_items = clean_and_filter(non_matches, min_length, max_length, clean=False)

    if process_output:
        filtered_items = list(set(filtered_items))
        filtered_items.sort()

    write_output(filtered_items, output_file, output_format, quiet, limit=limit)

    print_processing_stats(len(all_raw_list1), filtered_items, start_time=start_time)
    logging.info(
        f"[FilterFragments Mode] Filtering complete. Results saved to '{output_file}'."
    )


def set_operation_mode(
    input_files: Sequence[str],
    file2: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    operation: str,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """Perform set operations (intersection, union, difference) between input files (merged) and a second file."""
    start_time = time.perf_counter()
    allowed_operations = {'intersection', 'union', 'difference'}
    if operation not in allowed_operations:
        raise ValueError(
            f"Invalid operation '{operation}'. Must be one of: {', '.join(sorted(allowed_operations))}."
        )

    # Load and merge all input files
    raw_item_count_a = 0
    unique_a_list = []

    for input_file in input_files:
        raw, _, unique = _load_and_clean_file(
            input_file, min_length, max_length, clean_items=clean_items
        )
        raw_item_count_a += len(raw)
        unique_a_list.extend(unique)

    unique_a = list(dict.fromkeys(unique_a_list))

    raw_items_b, _, unique_b = _load_and_clean_file(
        file2, min_length, max_length, clean_items=clean_items
    )

    set_b = set(unique_b)

    if operation == 'intersection':
        result_items = [item for item in unique_a if item in set_b]
    elif operation == 'union':
        result_items = list(dict.fromkeys(unique_a + unique_b))
    else:  # difference
        result_items = [item for item in unique_a if item not in set_b]

    if process_output:
        result_items = sorted(set(result_items))

    write_output(result_items, output_file, output_format, quiet, limit=limit)

    print_processing_stats(
        raw_item_count_a + len(raw_items_b), result_items, start_time=start_time
    )
    logging.info(
        f"[Set Operation Mode] Completed {operation} between {len(input_files)} input file(s) and "
        f"'{file2}'. Output written to '{output_file}'."
    )

MODE_DETAILS = {
    "arrow": {
        "summary": "Extracts text from lines with arrows (->).",
        "description": "Finds text in lines that use arrows (like 'typo -> correction'). It saves the left side by default. Use --right to save the right side instead.",
        "example": "python multitool.py arrow typos.log --right --output corrections.txt",
        "flags": "[--right]",
    },
    "table": {
        "summary": "Extracts text from table-style entries (key = \"value\").",
        "description": "Gets keys or values from entries like 'key = \"value\"'. It saves the key by default. Use --right to save the quoted value instead.",
        "example": "python multitool.py table typos.toml --right -o corrections.txt",
        "flags": "[--right]",
    },
    "combine": {
        "summary": "Merges multiple files into one clean list.",
        "description": "Combines several files into one list. It removes duplicates and sorts the results alphabetically.",
        "example": "python multitool.py combine typos1.txt typos2.txt --output all_typos.txt",
        "flags": "",
    },
    "unique": {
        "summary": "Removes duplicates while keeping the original order.",
        "description": "Removes duplicate items from your list. Unlike 'combine', it preserves the order in which items first appeared in your files.",
        "example": "python multitool.py unique raw_typos.txt --output clean_typos.txt",
        "flags": "",
    },
    "backtick": {
        "summary": "Extracts text found inside backticks.",
        "description": "Finds text inside backticks (like `code`). It prioritizes items near words like 'error' or 'warning' to find the most relevant data.",
        "example": "python multitool.py backtick build.log --output suspects.txt",
        "flags": "",
    },
    "csv": {
        "summary": "Extracts specific columns from a CSV file.",
        "description": "Gets data from CSV files. By default, it extracts every column except the first one. Use --first-column to get only the first column, or --column to pick specific numbers.",
        "example": "python multitool.py csv typos.csv --column 2 -o corrections.txt",
        "flags": "[--first-column] [--column IDX]",
    },
    "markdown": {
        "summary": "Extracts items from Markdown bulleted lists.",
        "description": "Finds text in lines starting with -, *, or +. It can also split items by ':' or '->' to extract one side of a pair (use --right for the second part).",
        "example": "python multitool.py markdown notes.md --output items.txt",
        "flags": "[--right]",
    },
    "md-table": {
        "summary": "Extracts text from Markdown tables.",
        "description": "Finds text in cells of a Markdown table. It saves the first column by default. Use --right to save the second column instead, or --column to pick specific numbers. It automatically skips header and divider rows.",
        "example": "python multitool.py md-table readme.md --column 2 --output corrections.txt",
        "flags": "[--right] [--column IDX]",
    },
    "json": {
        "summary": "Extracts values from a JSON file using an optional key.",
        "description": "Finds values for a specific key in a JSON file. Use dots for nested keys (like 'user.name'). If no key is provided, it extracts items from the root. It automatically handles lists of objects.",
        "example": "python multitool.py json list.json -o items.txt",
        "flags": "[-k KEY]",
    },
    "yaml": {
        "summary": "Extracts values from a YAML file using an optional key.",
        "description": "Finds values for a specific key in a YAML file. Use dots for nested keys (like 'config.items'). If no key is provided, it extracts items from the root. It automatically handles lists.",
        "example": "python multitool.py yaml list.yaml -o items.txt",
        "flags": "[-k KEY]",
    },
    "line": {
        "summary": "Reads a file line by line.",
        "description": "Reads every line from a file, cleans the text, and writes it to the output. Useful for simple cleaning and filtering.",
        "example": "python multitool.py line raw_words.txt --output filtered.txt",
        "flags": "",
    },
    "words": {
        "summary": "Extracts individual words from a file.",
        "description": "Splits a file into individual words using whitespace or a custom delimiter. It's the standard way to get a list of every word used in a document. Use --smart to split by capital letters and symbols.",
        "example": "python multitool.py words report.txt --smart --output wordlist.txt",
        "flags": "[-d DELIMITER] [--smart]",
    },
    "ngrams": {
        "summary": "Extracts sequences of N words (n-grams).",
        "description": "Extracts sequences of N words from a file. This is useful for finding common phrases or context around typos. It supports sliding windows across line boundaries.",
        "example": "python multitool.py ngrams report.txt -n 2 --smart --output phrases.txt",
        "flags": "[-n N] [-d DELIMITER] [--smart]",
    },
    "count": {
        "summary": "Counts how many times each word or pair appears.",
        "description": "Counts frequency and sorts the list from most frequent to least frequent. Use -f arrow for a rich visual report with bar charts. Use --pairs to count word pairs (for example, typo -> correction) instead of single words.",
        "example": "python multitool.py count typos.log -f arrow --smart --pairs",
        "flags": "[--min-count N] [-d DELIM] [--smart] [--pairs]",
    },
    "filterfragments": {
        "summary": "Removes words if they are found inside words in another file.",
        "description": "Removes words from your list if they appear anywhere (even as a fragment) inside words in a second file.",
        "example": "python multitool.py filterfragments list.txt reference.txt --output unique.txt",
        "flags": "[FILE2]",
    },
    "check": {
        "summary": "Finds words that are both typos and corrections.",
        "description": "Checks for words that appear in both the typo and correction columns of a file. Use this to find errors in your typo lists.",
        "example": "python multitool.py check typos.csv --output duplicates.txt",
        "flags": "",
    },
    "set_operation": {
        "summary": "Compares two files using set logic.",
        "description": "Compares two files to find shared lines (intersection), all lines (union), or lines unique to the first file (difference).",
        "example": "python multitool.py set_operation fileA.txt fileB.txt --operation intersection --output shared.txt",
        "flags": "[FILE2] --operation OP",
    },
    "sample": {
        "summary": "Picks a random set of lines from a file.",
        "description": "Selects a random subset of lines. You can choose a specific number of lines (-n) or a percentage (--percent).",
        "example": "python multitool.py sample big_log.txt -n 100 -o sample.txt",
        "flags": "[-n N|--percent P]",
    },
    "regex": {
        "summary": "Finds text that matches a pattern (regular expression).",
        "description": "Finds and extracts all text that matches a Python regular expression pattern.",
        "example": "python multitool.py regex inputs.txt --pattern 'user_\\w+' --output users.txt",
        "flags": "[-r PATTERN]",
    },
    "map": {
        "summary": "Replaces items using a mapping file.",
        "description": "Replaces items in your list with new values from a mapping file. Supports CSV, Arrow, Table, JSON, and YAML mapping formats. Use --smart-case to preserve capitalization and --pairs to see both original and changed words.",
        "example": "python multitool.py map input.txt mapping.csv --smart-case --pairs",
        "flags": "[MAPPING] [--smart-case] [--pairs]",
    },
    "zip": {
        "summary": "Pairs lines from two files together.",
        "description": "Joins two files line-by-line into a paired format like 'typo -> correction'. Useful for creating mapping files from two separate lists.",
        "example": "python multitool.py zip typos.txt corrections.txt --output-format table --output typos.toml",
        "flags": "[FILE2]",
    },
    "swap": {
        "summary": "Reverses the order of elements in paired data.",
        "description": "Flips the left and right elements of pairs (for example, 'typo -> correction' becomes 'correction -> typo'). Supports Arrow, Table, CSV, and Markdown formats.",
        "example": "python multitool.py swap typos.csv --output-format arrow --output flipped.txt",
        "flags": "",
    },
    "pairs": {
        "summary": "Processes and converts paired data.",
        "description": "Reads pairs (like 'typo -> correction') from any supported format and writes them to the specified output format. Useful for cleaning, filtering, and format conversion.",
        "example": "python multitool.py pairs typos.json --output-format csv --output typos.csv",
        "flags": "",
    },
    "conflict": {
        "summary": "Finds typos that have multiple different corrections.",
        "description": "Identifies typos in your paired data that are associated with more than one unique correction. Use this to find inconsistencies in your typo lists.",
        "example": "python multitool.py conflict typos.csv --output-format arrow --output conflicts.txt",
        "flags": "",
    },
    "similarity": {
        "summary": "Filters paired data by the number of changes.",
        "description": "Filters pairs (typo -> correction) based on the number of character changes needed to turn one word into another. Use this to remove extra data or find specific types of typos.",
        "example": "python multitool.py similarity typos.txt --max-dist 2 --show-dist",
        "flags": "[--max-dist N --show-dist]",
    },
    "near_duplicates": {
        "summary": "Finds similar words in a single list.",
        "description": "Identifies pairs of words in your list that are very similar (only a few characters apart). Use this to find potential typos or unintended duplicates in a project.",
        "example": "python multitool.py near_duplicates words.txt --max-dist 1 --show-dist",
        "flags": "[--max-dist N --show-dist]",
    },
    "fuzzymatch": {
        "summary": "Finds similar words between two lists.",
        "description": "Identifies words in your list that are similar to words in a second list (dictionary). Use this to find likely corrections for typos. It defaults to a threshold of 1 character change.",
        "example": "python multitool.py fuzzymatch typos.txt dictionary.txt --max-dist 1 --show-dist",
        "flags": "[FILE2] [--max-dist N --show-dist]",
    },
    "stats": {
        "summary": "Calculates detailed statistics for a typo list.",
        "description": "Provides a detailed overview of your dataset. It reports counts, unique items, length distributions, and (optionally) paired data stats like conflicts, overlaps, and the number of changes between words.",
        "example": "python multitool.py stats typos.csv --pairs --output-format json",
        "flags": "[--pairs]",
    },
    "classify": {
        "summary": "Categorizes typo corrections based on their error type.",
        "description": "Labels typo pairs with error codes like [K] Keyboard, [T] Transposition, [D] Deletion, [I] Insertion, and [M] Multiple letters. Use --show-dist to include the number of character changes.",
        "example": "python multitool.py classify typos.txt --show-dist --output labeled.txt",
        "flags": "[--show-dist]",
    },
    "discovery": {
        "summary": "Discovers potential typos by comparing rare words to frequent words.",
        "description": "Automatically finds potential typos in a text by identifying rare words that are very similar to frequent words. It assumes that frequent words are likely correct and rare variations are likely typos. This is a powerful way to find errors without needing a dictionary.",
        "example": "python multitool.py discovery report.txt --rare-max 2 --freq-min 10 --max-dist 1",
        "flags": "[--rare-max N] [--freq-min N] [--max-dist N]",
    },
    "casing": {
        "summary": "Identifies words with inconsistent capitalization.",
        "description": "Finds words that appear in your files with multiple different casing styles (for example, 'hello', 'Hello', 'HELLO'). This is useful for identifying inconsistent naming or typos that differ only by case.",
        "example": "python multitool.py casing report.txt --smart --output-format arrow",
        "flags": "[-d DELIMITER] [--smart]",
    },
    "cycles": {
        "summary": "Identifies circular references in typo-correction pairs.",
        "description": "Detects cycles in your typo mappings (for example, 'A' maps to 'B' and 'B' maps back to 'A'). Circular references can cause issues during automated scrubbing and represent logic errors in your data.",
        "example": "python multitool.py cycles typos.csv --output-format arrow",
        "flags": "",
    },
    "repeated": {
        "summary": "Finds consecutive identical words.",
        "description": "Identifies doubled words (for example, 'the the') in your text. It outputs the duplicated pair and the suggested fix. Use --smart to handle CamelCase or punctuation.",
        "example": "python multitool.py repeated report.txt --smart --output-format arrow",
        "flags": "[-d DELIMITER] [--smart]",
    },
    "scrub": {
        "summary": "Replaces typos in text files based on a mapping.",
        "description": "Performs in-place replacements of typos in your text files using a mapping file. It tries to preserve the surrounding context (punctuation, whitespace) while fixing errors. It automatically handles compound words like 'CamelCase' and 'snake_case' variables. Supports CSV, Arrow, Table, JSON, and YAML mapping formats.",
        "example": "python multitool.py scrub input.txt --mapping corrections.csv --output fixed.txt",
        "flags": "[MAPPING]",
    },
    "diff": {
        "summary": "Compares two files to find added, removed, or changed items.",
        "description": "Identifies differences between two files or lists. It can track simple word additions/removals or (with --pairs) find changed corrections for existing typos. Color-coded output highlights what's new (+), what's gone (-), and what changed (~).",
        "example": "python multitool.py diff old_typos.csv new_typos.csv --pairs --output-format json",
        "flags": "[FILE2] [--pairs]",
    },
    "highlight": {
        "summary": "Highlights specific words or typos within text files.",
        "description": "Searches for words from a list or mapping and highlights them with color in the output. Useful as a non-destructive preview before using 'scrub'. Supports the same smart word detection as the scrubbing tool.",
        "example": "python multitool.py highlight input.txt --mapping corrections.csv",
        "flags": "[MAPPING] [--smart]",
    },
}


def get_mode_summary_text() -> str:
    """Return a formatted summary table of all available modes as a string."""
    categories = {
        "Extraction": ["arrow", "table", "backtick", "csv", "markdown", "md-table", "json", "yaml", "line", "words", "ngrams", "regex"],
        "Manipulation": ["combine", "unique", "diff", "highlight", "filterfragments", "set_operation", "sample", "map", "zip", "swap", "pairs", "scrub"],
        "Analysis": ["count", "check", "conflict", "cycles", "similarity", "near_duplicates", "fuzzymatch", "stats", "classify", "discovery", "casing", "repeated"],
    }

    lines = []
    lines.append(f"{BOLD}Available Modes:{RESET}")

    max_mode_len = max(len(m) for m in MODE_DETAILS.keys())
    width = max_mode_len + 4

    # Table Header
    header_mode = f"{'Mode':<{width}}"
    header_summary = f"{'Summary':<55}"
    header_flags = "Quick Start / Primary Flags"
    lines.append(f"\n    {BOLD}{header_mode} {header_summary} {header_flags}{RESET}")
    # Separator matches combined column widths + spacing
    total_header_width = width + 55 + len(header_flags) + 2
    lines.append(f"    {'-' * total_header_width}")

    for category, modes in categories.items():
        lines.append(f"\n  {BLUE}{category.upper()}{RESET}")
        lines.append(f"  {BLUE}{'─' * 55}{RESET}")
        for mode in modes:
            if mode in MODE_DETAILS:
                details = MODE_DETAILS[mode]
                summary = details['summary']
                flags = details.get('flags', '')
                lines.append(f"    {GREEN}{mode:<{width}}{RESET} {summary:<55} {YELLOW}{flags}{RESET}")

    lines.append(f"\nRun '{BOLD}python multitool.py --mode-help <mode>{RESET}' for details on a specific mode.\n")
    return "\n".join(lines)


def print_mode_summary() -> None:
    """Print a summary table of all available modes, grouped by category."""
    print("\n" + get_mode_summary_text())


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


class ModeHelpAction(argparse.Action):
    """Custom argparse action that prints detailed help for one or all modes."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | None,
        option_string: str | None = None,
    ) -> None:
        if values in (None, "all"):
            # Show a summary table of all modes
            print_mode_summary()
            parser.exit()
        else:
            # Show detailed help for a single mode
            details = MODE_DETAILS.get(values)
            if not details:
                parser.error(f"Unknown mode: {values}")

            divider = f"{BLUE}{'─' * 80}{RESET}"
            block = [
                divider,
                f"{BOLD}MODE:{RESET} {GREEN}{values.upper()}{RESET}",
                divider,
                f"{BOLD}SUMMARY:{RESET}     {details['summary']}",
            ]

            if details.get("description"):
                # Simple indentation for description
                desc = details['description']
                block.append(f"{BOLD}DESCRIPTION:{RESET} {desc}")

            block.append(f"\n{BOLD}USAGE:{RESET}       python multitool.py {values} [FILES...] [FLAGS]")

            if details.get("flags"):
                block.append(f"{BOLD}FLAGS:{RESET}       {YELLOW}{details['flags']}{RESET}")

            if details.get("example"):
                block.append(f"\n{BOLD}EXAMPLE:{RESET}")
                block.append(f"  {BLUE}{details['example']}{RESET}")

            block.append(divider)

            parser.exit(message="\n" + "\n".join(block) + "\n\n")


def _build_parser() -> argparse.ArgumentParser:
    # Build a categorized mode summary for the epilog
    mode_summary = get_mode_summary_text()

    parser = argparse.ArgumentParser(
        description="A versatile tool for cleaning, extracting, and analyzing text files.",
        epilog=dedent(
            f"""
            {BLUE}Examples:{RESET}
              {GREEN}python multitool.py --mode-help{RESET}             # Show a summary of every mode
              {GREEN}python multitool.py --mode-help csv{RESET}         # Describe the CSV extraction mode
              {GREEN}python multitool.py arrow file.txt{RESET}          # Run a specific mode
              {GREEN}python multitool.py --mode csv --input file.txt{RESET}  # Old way to run the tool
            """
        ).strip() + "\n\n" + mode_summary,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--mode-help",
        nargs="?",
        choices=[*MODE_DETAILS.keys(), "all"],
        metavar="mode",
        action=ModeHelpAction,
        help="Display extended documentation for a specific mode or all modes.",
    )

    # Input/Output Group
    io_group = parser.add_argument_group(f"{BLUE}INPUT/OUTPUT OPTIONS{RESET}")
    io_group.add_argument(
        '-o', '--output',
        type=str,
        default='-',
        help="Where to save the results. Use '-' to print to the screen (default: the screen).",
    )
    io_group.add_argument(
        '-f', '--output-format', '--format',
        dest='output_format',
        choices=['line', 'json', 'csv', 'markdown', 'md-table', 'arrow', 'table', 'yaml'],
        metavar='FMT',
        default='line',
        help="Choose the format for the output (default: line). Choices: line, json, csv, markdown, md-table, arrow, table, yaml.",
    )
    io_group.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress bars and informational log output.',
    )

    # Processing Options Group
    proc_group = parser.add_argument_group(f"{BLUE}PROCESSING OPTIONS{RESET}")
    proc_group.add_argument(
        '-m', '--min-length',
        type=int,
        default=3,
        help="Skip items shorter than this (default: 3).",
    )
    proc_group.add_argument(
        '-M', '--max-length',
        type=int,
        default=1000,
        help="Skip items longer than this (default: 1000).",
    )
    proc_group.add_argument(
        '-P', '--process-output',
        action='store_true',
        help="Sort the output and remove duplicates.",
    )
    proc_group.add_argument(
        '--process',
        action='store_true',
        dest='process_output',
        help=argparse.SUPPRESS,
    )
    proc_group.add_argument(
        '-R', '--raw',
        action='store_true',
        help="Keep the original text. Do not change it to lowercase or remove punctuation.",
    )
    proc_group.add_argument(
        '-L', '--limit',
        type=int,
        help="Limit the number of items in the output.",
    )

    subparsers = parser.add_subparsers(dest='mode', required=True, help=argparse.SUPPRESS)

    arrow_parser = subparsers.add_parser(
        'arrow',
        help=MODE_DETAILS['arrow']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['arrow']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['arrow']['example']}{RESET}",
    )
    arrow_options = arrow_parser.add_argument_group(f"{BLUE}ARROW OPTIONS{RESET}")
    arrow_options.add_argument(
        '--right',
        action='store_true',
        help="Extract the right side (correction) instead of the left side (typo).",
    )
    _add_common_mode_arguments(arrow_parser)

    table_parser = subparsers.add_parser(
        'table',
        help=MODE_DETAILS['table']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['table']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['table']['example']}{RESET}",
    )
    table_options = table_parser.add_argument_group(f"{BLUE}TABLE OPTIONS{RESET}")
    table_options.add_argument(
        '--right',
        action='store_true',
        help="Extract the value (right side) instead of the key (left side).",
    )
    _add_common_mode_arguments(table_parser)

    backtick_parser = subparsers.add_parser(
        'backtick',
        help=MODE_DETAILS['backtick']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['backtick']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['backtick']['example']}{RESET}",
    )
    _add_common_mode_arguments(backtick_parser)

    csv_parser = subparsers.add_parser(
        'csv',
        help=MODE_DETAILS['csv']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['csv']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['csv']['example']}{RESET}",
    )
    csv_options = csv_parser.add_argument_group(f"{BLUE}CSV OPTIONS{RESET}")
    csv_options.add_argument(
        '--first-column',
        action='store_true',
        help='Extract the first column instead of subsequent columns.',
    )
    csv_options.add_argument(
        '-d', '--delimiter',
        type=str,
        default=',',
        help='The delimiter character for CSV files (default: ,).',
    )
    csv_options.add_argument(
        '-c', '--column',
        dest='columns',
        type=int,
        nargs='+',
        metavar='IDX',
        help='One or more 0-based column numbers to extract.',
    )
    _add_common_mode_arguments(csv_parser)

    markdown_parser = subparsers.add_parser(
        'markdown',
        help=MODE_DETAILS['markdown']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['markdown']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['markdown']['example']}{RESET}",
    )
    markdown_options = markdown_parser.add_argument_group(f"{BLUE}MARKDOWN OPTIONS{RESET}")
    markdown_options.add_argument(
        '--right',
        action='store_true',
        help="Extract the right side of a pair (split by ':' or '->') instead of the left side.",
    )
    _add_common_mode_arguments(markdown_parser)

    md_table_parser = subparsers.add_parser(
        'md-table',
        help=MODE_DETAILS['md-table']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['md-table']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['md-table']['example']}{RESET}",
    )
    md_table_options = md_table_parser.add_argument_group(f"{BLUE}MD TABLE OPTIONS{RESET}")
    md_table_options.add_argument(
        '--right',
        action='store_true',
        help="Extract the second column instead of the first.",
    )
    md_table_options.add_argument(
        '-c', '--column',
        dest='columns',
        type=int,
        nargs='+',
        metavar='IDX',
        help='One or more 0-based column numbers to extract.',
    )
    _add_common_mode_arguments(md_table_parser)

    json_parser = subparsers.add_parser(
        'json',
        help=MODE_DETAILS['json']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['json']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['json']['example']}{RESET}",
    )
    json_options = json_parser.add_argument_group(f"{BLUE}JSON OPTIONS{RESET}")
    json_options.add_argument(
        '-k', '--key',
        type=str,
        default='',
        help="The key path to extract (for example 'items.name'). If omitted, extracts from the root.",
    )
    _add_common_mode_arguments(json_parser)

    yaml_parser = subparsers.add_parser(
        'yaml',
        help=MODE_DETAILS['yaml']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['yaml']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['yaml']['example']}{RESET}",
    )
    yaml_options = yaml_parser.add_argument_group(f"{BLUE}YAML OPTIONS{RESET}")
    yaml_options.add_argument(
        '-k', '--key',
        type=str,
        default='',
        help="The key path to extract (for example 'config.items'). If omitted, extracts from the root.",
    )
    _add_common_mode_arguments(yaml_parser)

    combine_parser = subparsers.add_parser(
        'combine',
        help=MODE_DETAILS['combine']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['combine']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['combine']['example']}{RESET}",
    )
    _add_common_mode_arguments(combine_parser, include_process_output=False)

    unique_parser = subparsers.add_parser(
        'unique',
        help=MODE_DETAILS['unique']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['unique']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['unique']['example']}{RESET}",
    )
    _add_common_mode_arguments(unique_parser)

    line_parser = subparsers.add_parser(
        'line',
        help=MODE_DETAILS['line']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['line']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['line']['example']}{RESET}",
    )
    _add_common_mode_arguments(line_parser)

    count_parser = subparsers.add_parser(
        'count',
        help=MODE_DETAILS['count']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['count']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['count']['example']}{RESET}",
    )
    count_options = count_parser.add_argument_group(f"{BLUE}COUNT OPTIONS{RESET}")
    count_options.add_argument(
        '--min-count',
        type=int,
        default=1,
        help="Minimum occurrence count to include an item in the output (default: 1).",
    )
    count_options.add_argument(
        '--max-count',
        type=int,
        help="Maximum occurrence count to include an item in the output.",
    )
    count_options.add_argument(
        '-d', '--delimiter',
        type=str,
        help='The delimiter character to split words by (default: whitespace).',
    )
    count_options.add_argument(
        '-S', '--smart',
        action='store_true',
        help='Split by symbols and capital letters (for example, splitting "CamelCase" into "Camel" and "Case").',
    )
    count_options.add_argument(
        '-p', '--pairs',
        action='store_true',
        help='Count frequencies of word pairs (for example, typo -> correction) instead of single words.',
    )
    _add_common_mode_arguments(count_parser, include_process_output=False)

    filter_parser = subparsers.add_parser(
        'filterfragments',
        help=MODE_DETAILS['filterfragments']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['filterfragments']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['filterfragments']['example']}{RESET}",
    )
    filter_options = filter_parser.add_argument_group(f"{BLUE}FILTER FRAGMENTS OPTIONS{RESET}")
    filter_options.add_argument(
        '--file2',
        type=str,
        required=False,
        help='Path to the second file used for comparison.',
    )
    _add_common_mode_arguments(filter_parser)

    check_parser = subparsers.add_parser(
        'check',
        help=MODE_DETAILS['check']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['check']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['check']['example']}{RESET}",
    )
    _add_common_mode_arguments(check_parser)

    conflict_parser = subparsers.add_parser(
        'conflict',
        help=MODE_DETAILS['conflict']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['conflict']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['conflict']['example']}{RESET}",
    )
    _add_common_mode_arguments(conflict_parser)

    cycles_parser = subparsers.add_parser(
        'cycles',
        help=MODE_DETAILS['cycles']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['cycles']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['cycles']['example']}{RESET}",
    )
    _add_common_mode_arguments(cycles_parser)

    similarity_parser = subparsers.add_parser(
        'similarity',
        help=MODE_DETAILS['similarity']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['similarity']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['similarity']['example']}{RESET}",
    )
    similarity_options = similarity_parser.add_argument_group(f"{BLUE}SIMILARITY OPTIONS{RESET}")
    similarity_options.add_argument(
        '--min-dist',
        type=int,
        default=0,
        help="Minimum number of changes to include (default: 0).",
    )
    similarity_options.add_argument(
        '--max-dist',
        type=int,
        help="Maximum number of changes to include.",
    )
    similarity_options.add_argument(
        '--show-dist',
        action='store_true',
        help="Include the number of character changes in the output.",
    )
    _add_common_mode_arguments(similarity_parser)

    near_duplicates_parser = subparsers.add_parser(
        'near_duplicates',
        help=MODE_DETAILS['near_duplicates']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['near_duplicates']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['near_duplicates']['example']}{RESET}",
    )
    nd_options = near_duplicates_parser.add_argument_group(f"{BLUE}NEAR DUPLICATES OPTIONS{RESET}")
    nd_options.add_argument(
        '--min-dist',
        type=int,
        default=1,
        help="Minimum number of changes to include (default: 1).",
    )
    nd_options.add_argument(
        '--max-dist',
        type=int,
        default=1,
        help="Maximum number of changes to include (default: 1).",
    )
    nd_options.add_argument(
        '--show-dist',
        action='store_true',
        help="Include the number of character changes in the output.",
    )
    _add_common_mode_arguments(near_duplicates_parser)

    fuzzymatch_parser = subparsers.add_parser(
        'fuzzymatch',
        help=MODE_DETAILS['fuzzymatch']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['fuzzymatch']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['fuzzymatch']['example']}{RESET}",
    )
    fm_options = fuzzymatch_parser.add_argument_group(f"{BLUE}FUZZY MATCH OPTIONS{RESET}")
    fm_options.add_argument(
        '--file2',
        type=str,
        required=False,
        help='Path to the second file (dictionary) to match against.',
    )
    fm_options.add_argument(
        '--min-dist',
        type=int,
        default=1,
        help="Minimum number of changes to include (default: 1).",
    )
    fm_options.add_argument(
        '--max-dist',
        type=int,
        default=1,
        help="Maximum number of changes to include (default: 1).",
    )
    fm_options.add_argument(
        '--show-dist',
        action='store_true',
        help="Include the number of character changes in the output.",
    )
    _add_common_mode_arguments(fuzzymatch_parser)

    stats_parser = subparsers.add_parser(
        'stats',
        help=MODE_DETAILS['stats']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['stats']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['stats']['example']}{RESET}",
    )
    stats_options = stats_parser.add_argument_group(f"{BLUE}STATS OPTIONS{RESET}")
    stats_options.add_argument(
        '-p', '--pairs',
        action='store_true',
        help="Perform pair-level analysis (typos vs corrections) in addition to item-level stats.",
    )
    _add_common_mode_arguments(stats_parser, include_process_output=False)

    classify_parser = subparsers.add_parser(
        'classify',
        help=MODE_DETAILS['classify']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['classify']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['classify']['example']}{RESET}",
    )
    classify_options = classify_parser.add_argument_group(f"{BLUE}CLASSIFY OPTIONS{RESET}")
    classify_options.add_argument(
        '--show-dist',
        action='store_true',
        help="Include the number of character changes in the output labels.",
    )
    _add_common_mode_arguments(classify_parser)

    discovery_parser = subparsers.add_parser(
        'discovery',
        help=MODE_DETAILS['discovery']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['discovery']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['discovery']['example']}{RESET}",
    )
    discovery_options = discovery_parser.add_argument_group(f"{BLUE}DISCOVERY OPTIONS{RESET}")
    discovery_options.add_argument(
        '--rare-max',
        type=int,
        default=1,
        help="Maximum frequency for a word to be considered a potential typo (default: 1).",
    )
    discovery_options.add_argument(
        '--freq-min',
        type=int,
        default=5,
        help="Minimum frequency for a word to be considered a potential correction (default: 5).",
    )
    discovery_options.add_argument(
        '--min-dist',
        type=int,
        default=1,
        help="Minimum number of changes between typo and correction (default: 1).",
    )
    discovery_options.add_argument(
        '--max-dist',
        type=int,
        default=1,
        help="Maximum number of changes between typo and correction (default: 1).",
    )
    discovery_options.add_argument(
        '--show-dist',
        action='store_true',
        help="Include the number of character changes in the output.",
    )
    _add_common_mode_arguments(discovery_parser)

    casing_parser = subparsers.add_parser(
        'casing',
        help=MODE_DETAILS['casing']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['casing']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['casing']['example']}{RESET}",
    )
    casing_options = casing_parser.add_argument_group(f"{BLUE}CASING OPTIONS{RESET}")
    casing_options.add_argument(
        '-d', '--delimiter',
        type=str,
        help='The delimiter character to split words by (default: whitespace).',
    )
    casing_options.add_argument(
        '-S', '--smart',
        action='store_true',
        help='Split by symbols and capital letters (for example, splitting "CamelCase" into "Camel" and "Case").',
    )
    _add_common_mode_arguments(casing_parser)

    repeated_parser = subparsers.add_parser(
        'repeated',
        help=MODE_DETAILS['repeated']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['repeated']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['repeated']['example']}{RESET}",
    )
    repeated_options = repeated_parser.add_argument_group(f"{BLUE}REPEATED OPTIONS{RESET}")
    repeated_options.add_argument(
        '-d', '--delimiter',
        type=str,
        help='The delimiter character to split words by (default: whitespace).',
    )
    repeated_options.add_argument(
        '-S', '--smart',
        action='store_true',
        help='Split by symbols and capital letters (for example, splitting "CamelCase" into "Camel" and "Case").',
    )
    _add_common_mode_arguments(repeated_parser)

    set_parser = subparsers.add_parser(
        'set_operation',
        help=MODE_DETAILS['set_operation']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['set_operation']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['set_operation']['example']}{RESET}",
    )
    set_options = set_parser.add_argument_group(f"{BLUE}SET OPERATION OPTIONS{RESET}")
    set_options.add_argument(
        '--file2',
        type=str,
        required=False,
        help='Path to the second input file for set comparisons.',
    )
    set_options.add_argument(
        '--operation',
        type=str,
        choices=['intersection', 'union', 'difference'],
        required=True,
        help='Set operation to perform between the two files.',
    )
    _add_common_mode_arguments(set_parser)

    sample_parser = subparsers.add_parser(
        'sample',
        help=MODE_DETAILS['sample']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['sample']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['sample']['example']}{RESET}",
    )
    sample_options = sample_parser.add_argument_group(f"{BLUE}SAMPLE OPTIONS{RESET}")
    group = sample_options.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-n', '--n',
        dest='sample_count',
        type=int,
        help='Number of lines to sample.',
    )
    group.add_argument(
        '--percent',
        dest='sample_percent',
        type=float,
        help='Percentage of lines to sample (0-100).',
    )
    _add_common_mode_arguments(sample_parser)

    words_parser = subparsers.add_parser(
        'words',
        help=MODE_DETAILS['words']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['words']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['words']['example']}{RESET}",
    )
    words_options = words_parser.add_argument_group(f"{BLUE}WORDS OPTIONS{RESET}")
    words_options.add_argument(
        '-d', '--delimiter',
        type=str,
        help='The delimiter character to split words by (default: whitespace).',
    )
    words_options.add_argument(
        '-S', '--smart',
        action='store_true',
        help='Split by symbols and capital letters (for example, splitting "CamelCase" into "Camel" and "Case").',
    )
    _add_common_mode_arguments(words_parser)

    ngrams_parser = subparsers.add_parser(
        'ngrams',
        help=MODE_DETAILS['ngrams']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['ngrams']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['ngrams']['example']}{RESET}",
    )
    ngrams_options = ngrams_parser.add_argument_group(f"{BLUE}NGRAMS OPTIONS{RESET}")
    ngrams_options.add_argument(
        '-n', '--n',
        type=int,
        default=2,
        help='The number of words in each n-gram (default: 2).',
    )
    ngrams_options.add_argument(
        '-d', '--delimiter',
        type=str,
        help='The delimiter character to split words by (default: whitespace).',
    )
    ngrams_options.add_argument(
        '-S', '--smart',
        action='store_true',
        help='Split by symbols and capital letters (for example, splitting "CamelCase" into "Camel" and "Case").',
    )
    _add_common_mode_arguments(ngrams_parser)

    regex_parser = subparsers.add_parser(
        'regex',
        help=MODE_DETAILS['regex']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['regex']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['regex']['example']}{RESET}",
    )
    regex_options = regex_parser.add_argument_group(f"{BLUE}REGEX OPTIONS{RESET}")
    regex_options.add_argument(
        '-r', '--pattern',
        type=str,
        required=True,
        help="The regular expression pattern to match.",
    )
    _add_common_mode_arguments(regex_parser)

    map_parser = subparsers.add_parser(
        'map',
        help=MODE_DETAILS['map']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['map']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['map']['example']}{RESET}",
    )
    map_options = map_parser.add_argument_group(f"{BLUE}MAP OPTIONS{RESET}")
    map_options.add_argument(
        '-s', '--mapping',
        type=str,
        required=False,
        help='Path to the mapping file (CSV or Arrow format).',
    )
    map_options.add_argument(
        '--drop-missing',
        action='store_true',
        help='If set, items not found in the mapping are dropped. Default is to keep them.',
    )
    map_options.add_argument(
        '-p', '--pairs',
        action='store_true',
        help='Output the original word along with its transformation.',
    )
    map_options.add_argument(
        '--smart-case',
        action='store_true',
        help="Automatically match the casing of the original word (for example, 'TeH' -> 'The').",
    )
    _add_common_mode_arguments(map_parser)

    scrub_parser = subparsers.add_parser(
        'scrub',
        help=MODE_DETAILS['scrub']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['scrub']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['scrub']['example']}{RESET}",
    )
    scrub_options = scrub_parser.add_argument_group(f"{BLUE}SCRUB OPTIONS{RESET}")
    scrub_options.add_argument(
        '-s', '--mapping',
        type=str,
        required=False,
        help='Path to the mapping file.',
    )
    scrub_options.add_argument(
        '--in-place',
        nargs='?',
        const='',
        metavar='EXT',
        help="Modify files in place. If an extension is provided (for example, '.bak'), a backup is created.",
    )
    scrub_options.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be changed without modifying any files.",
    )
    scrub_options.add_argument(
        '--smart-case',
        action='store_true',
        help="Automatically match the casing of the original word (for example, 'Teh' -> 'The').",
    )
    _add_common_mode_arguments(scrub_parser, include_process_output=False, include_limit=False)

    diff_parser = subparsers.add_parser(
        'diff',
        help=MODE_DETAILS['diff']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['diff']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['diff']['example']}{RESET}",
    )
    diff_options = diff_parser.add_argument_group(f"{BLUE}DIFF OPTIONS{RESET}")
    diff_options.add_argument(
        '--file2',
        type=str,
        required=False,
        help='Path to the second file to compare against.',
    )
    diff_options.add_argument(
        '-p', '--pairs',
        action='store_true',
        help='Compare word pairs (typo -> correction) instead of single words.',
    )
    _add_common_mode_arguments(diff_parser)

    zip_parser = subparsers.add_parser(
        'zip',
        help=MODE_DETAILS['zip']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['zip']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['zip']['example']}{RESET}",
    )
    zip_options = zip_parser.add_argument_group(f"{BLUE}ZIP OPTIONS{RESET}")
    zip_options.add_argument(
        '--file2',
        type=str,
        required=False,
        help='Path to the second file to zip with the first.',
    )
    _add_common_mode_arguments(zip_parser)

    swap_parser = subparsers.add_parser(
        'swap',
        help=MODE_DETAILS['swap']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['swap']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['swap']['example']}{RESET}",
    )
    _add_common_mode_arguments(swap_parser)

    pairs_parser = subparsers.add_parser(
        'pairs',
        help=MODE_DETAILS['pairs']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['pairs']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['pairs']['example']}{RESET}",
    )
    _add_common_mode_arguments(pairs_parser)

    highlight_parser = subparsers.add_parser(
        'highlight',
        help=MODE_DETAILS['highlight']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['highlight']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['highlight']['example']}{RESET}",
    )
    highlight_options = highlight_parser.add_argument_group(f"{BLUE}HIGHLIGHT OPTIONS{RESET}")
    highlight_options.add_argument(
        '-s', '--mapping',
        type=str,
        required=False,
        help='Path to the mapping file or word list.',
    )
    highlight_options.add_argument(
        '-S', '--smart',
        action='store_true',
        help='Highlight subword matches (for example, highlighting "teh" inside "tehWord").',
    )
    _add_common_mode_arguments(highlight_parser)

    return parser


def _normalize_mode_args(
    argv: Sequence[str], parser: argparse.ArgumentParser
) -> List[str]:
    """Normalize legacy --mode usage into a positional subcommand."""
    if "--mode" not in argv:
        return list(argv)

    argv_list = list(argv)
    if argv_list.count("--mode") > 1:
        parser.error("Only one --mode flag may be provided.")

    mode_index = argv_list.index("--mode")
    if mode_index == len(argv_list) - 1:
        parser.error("--mode requires a value.")

    mode_value = argv_list[mode_index + 1]
    positional_mode = (
        argv_list[0] if argv_list and argv_list[0] in MODE_DETAILS else None
    )
    if positional_mode and positional_mode != mode_value:
        parser.error(
            f"--mode '{mode_value}' conflicts with positional mode '{positional_mode}'."
        )

    del argv_list[mode_index : mode_index + 2]
    if not positional_mode:
        argv_list.insert(0, mode_value)

    return argv_list


def main() -> None:
    if len(sys.argv) == 1:
        print_mode_summary()
        sys.exit(0)

    parser = _build_parser()
    argv = _normalize_mode_args(sys.argv[1:], parser)

    args = parser.parse_args(argv)

    log_level = logging.WARNING if args.quiet else logging.INFO
    # Use a custom handler and formatter to keep output clean
    handler = logging.StreamHandler()
    handler.setFormatter(MinimalFormatter())
    logging.basicConfig(level=log_level, handlers=[handler])

    if args.min_length < 1:
        logging.error("--min-length must be a number of 1 or more.")
        sys.exit(1)
    if args.max_length < args.min_length:
        logging.error("--max-length must be greater than or equal to --min-length.")
        sys.exit(1)

    # Resolve input arguments (positional vs flag)
    pos_inputs = getattr(args, 'input_files_pos', []) or []
    flag_inputs = getattr(args, 'input_files_flag', []) or []
    input_paths = pos_inputs + flag_inputs

    # Expand glob patterns for input paths
    expanded_paths = []
    for path in input_paths:
        if path == '-':
            expanded_paths.append(path)
        else:
            matches = glob.glob(path)
            if matches:
                expanded_paths.extend(sorted(matches))
            else:
                expanded_paths.append(path)
    input_paths = expanded_paths

    # Default to standard input ('-') if neither is provided
    if not input_paths:
        input_paths = ['-']

    # Store for handler
    args.input = input_paths

    # Fallback logic for modes that require a secondary file (for example, zip, map)
    # If the flag is missing but we have at least 2 positional arguments, use the last one as the secondary file.
    if args.mode in {'zip', 'filterfragments', 'set_operation', 'fuzzymatch', 'diff'}:
        if getattr(args, 'file2', None) is None and len(input_paths) >= 2:
            args.file2 = input_paths.pop()
            args.input = input_paths
    elif args.mode in {'map', 'scrub'}:
        if getattr(args, 'mapping', None) is None and len(input_paths) >= 2:
            args.mapping = input_paths.pop()
            args.input = input_paths

    file2 = getattr(args, 'file2', None)
    # Check for missing secondary files after fallback attempt
    if args.mode in {'zip', 'filterfragments', 'set_operation', 'fuzzymatch', 'diff'} and file2 is None:
        logging.error(f"{args.mode.capitalize()} mode requires a secondary file (provide FILE2 positionally or use --file2).")
        sys.exit(1)
    if args.mode in {'map', 'scrub'} and getattr(args, 'mapping', None) is None:
        logging.error(f"{args.mode.capitalize()} mode requires a mapping file (provide MAPPING positionally or use --mapping).")
        sys.exit(1)

    operation = getattr(args, 'operation', None)
    first_column = getattr(args, 'first_column', False)
    delimiter = getattr(args, 'delimiter', ',')
    right_side = getattr(args, 'right', False)
    sample_count = getattr(args, 'sample_count', None)
    sample_percent = getattr(args, 'sample_percent', None)
    limit = getattr(args, 'limit', None)
    output_format = getattr(args, 'output_format', 'line')


    clean_items = not getattr(args, 'raw', False)

    common_kwargs = {
        'input_files': args.input,
        'output_file': args.output,
        'min_length': args.min_length,
        'max_length': args.max_length,
        'process_output': getattr(args, 'process_output', False),
        'quiet': args.quiet,
        'clean_items': clean_items,
        'limit': limit,
    }

    handler_map = {
        'arrow': (
            arrow_mode,
            {
                **common_kwargs,
                'right_side': right_side,
                'output_format': output_format,
            },
        ),
        'ngrams': (
            ngrams_mode,
            {
                **common_kwargs,
                'n': getattr(args, 'n', 2),
                'delimiter': getattr(args, 'delimiter', None),
                'smart': getattr(args, 'smart', False),
                'output_format': output_format,
            },
        ),
        'classify': (
            classify_mode,
            {
                **common_kwargs,
                'show_dist': getattr(args, 'show_dist', False),
                'output_format': output_format,
            },
        ),
        'repeated': (
            repeated_mode,
            {
                **common_kwargs,
                'delimiter': getattr(args, 'delimiter', None),
                'smart': getattr(args, 'smart', False),
                'output_format': output_format,
            },
        ),
        'casing': (
            casing_mode,
            {
                **common_kwargs,
                'delimiter': getattr(args, 'delimiter', None),
                'smart': getattr(args, 'smart', False),
                'output_format': output_format,
            },
        ),
        'md-table': (
            md_table_mode,
            {
                **common_kwargs,
                'right_side': right_side,
                'output_format': output_format,
                'columns': getattr(args, 'columns', None),
            },
        ),
        'table': (
            table_mode,
            {
                **common_kwargs,
                'right_side': right_side,
                'output_format': output_format,
            },
        ),
        'backtick': (
            backtick_mode,
            {**common_kwargs, 'output_format': output_format},
        ),
        'csv': (
            csv_mode,
            {
                **common_kwargs,
                'first_column': first_column,
                'delimiter': delimiter,
                'output_format': output_format,
                'columns': getattr(args, 'columns', None),
            },
        ),
        'markdown': (
            markdown_mode,
            {
                **common_kwargs,
                'right_side': right_side,
                'output_format': output_format,
            },
        ),
        'yaml': (
            yaml_mode,
            {
                **common_kwargs,
                'key': getattr(args, 'key', ''),
                'output_format': output_format,
            },
        ),
        'json': (
            json_mode,
            {
                **common_kwargs,
                'key': getattr(args, 'key', ''),
                'output_format': output_format,
            },
        ),
        'line': (line_mode, {**common_kwargs, 'output_format': output_format}),
        'words': (
            words_mode,
            {
                **common_kwargs,
                'delimiter': getattr(args, 'delimiter', None),
                'smart': getattr(args, 'smart', False),
                'output_format': output_format,
            },
        ),
        'count': (
            count_mode,
            {
                **common_kwargs,
                'min_count': getattr(args, 'min_count', 1),
                'max_count': getattr(args, 'max_count', None),
                'output_format': output_format,
                'delimiter': getattr(args, 'delimiter', None),
                'smart': getattr(args, 'smart', False),
                'pairs': getattr(args, 'pairs', False),
            },
        ),
        'filterfragments': (
            filter_fragments_mode,
            {**common_kwargs, 'file2': file2, 'output_format': output_format},
        ),
        'check': (
            check_mode,
            {**common_kwargs, 'output_format': output_format},
        ),
        'set_operation': (
            set_operation_mode,
            {
                **common_kwargs,
                'file2': file2,
                'operation': operation,
                'output_format': output_format,
            },
        ),
        'combine': (
            combine_mode,
            {
                'input_files': input_paths,
                'output_file': args.output,
                'min_length': args.min_length,
                'max_length': args.max_length,
                'process_output': getattr(args, 'process_output', False),
                'quiet': args.quiet,
                'output_format': output_format,
                'clean_items': clean_items,
                'limit': limit,
            },
        ),
        'diff': (
            diff_mode,
            {
                **common_kwargs,
                'file2': file2,
                'pairs': getattr(args, 'pairs', False),
                'output_format': output_format,
            }
        ),
        'unique': (
            unique_mode,
            {
                'input_files': input_paths,
                'output_file': args.output,
                'min_length': args.min_length,
                'max_length': args.max_length,
                'process_output': getattr(args, 'process_output', False),
                'quiet': args.quiet,
                'output_format': output_format,
                'clean_items': clean_items,
                'limit': limit,
            },
        ),
        'sample': (
            sample_mode,
            {
                **common_kwargs,
                'sample_count': sample_count,
                'sample_percent': sample_percent,
                'output_format': output_format,
            },
        ),
        'regex': (
            regex_mode,
            {
                # regex_mode doesn't use clean_items from common_kwargs (it sets it to False)
                'input_files': args.input,
                'output_file': args.output,
                'min_length': args.min_length,
                'max_length': args.max_length,
                'process_output': getattr(args, 'process_output', False),
                'quiet': args.quiet,
                'pattern': getattr(args, 'pattern', ''),
                'output_format': output_format,
                'limit': limit,
            },
        ),
        'map': (
            map_mode,
            {
                **common_kwargs,
                'mapping_file': getattr(args, 'mapping', ''),
                'drop_missing': getattr(args, 'drop_missing', False),
                'output_format': output_format,
                'pairs': getattr(args, 'pairs', False),
                'smart_case': getattr(args, 'smart_case', False),
            }
        ),
        'scrub': (
            scrub_mode,
            {
                'input_files': args.input,
                'mapping_file': getattr(args, 'mapping', ''),
                'output_file': args.output,
                'min_length': args.min_length,
                'max_length': args.max_length,
                'process_output': False,
                'quiet': args.quiet,
                'clean_items': clean_items,
                'limit': limit,
                'in_place': getattr(args, 'in_place', None),
                'dry_run': getattr(args, 'dry_run', False),
                'smart_case': getattr(args, 'smart_case', False),
            }
        ),
        'zip': (
            zip_mode,
            {
                **common_kwargs,
                'file2': file2,
                'output_format': output_format,
            }
        ),
        'swap': (
            swap_mode,
            {
                **common_kwargs,
                'output_format': output_format,
            },
        ),
        'pairs': (
            pairs_mode,
            {
                **common_kwargs,
                'output_format': output_format,
            }
        ),
        'conflict': (
            conflict_mode,
            {**common_kwargs, 'output_format': output_format},
        ),
        'cycles': (
            cycles_mode,
            {**common_kwargs, 'output_format': output_format},
        ),
        'similarity': (
            similarity_mode,
            {
                **common_kwargs,
                'min_dist': getattr(args, 'min_dist', 0),
                'max_dist': getattr(args, 'max_dist', None),
                'show_dist': getattr(args, 'show_dist', False),
                'output_format': output_format,
            },
        ),
        'near_duplicates': (
            near_duplicates_mode,
            {
                **common_kwargs,
                'min_dist': getattr(args, 'min_dist', 1),
                'max_dist': getattr(args, 'max_dist', 1),
                'show_dist': getattr(args, 'show_dist', False),
                'output_format': output_format,
            },
        ),
        'fuzzymatch': (
            fuzzymatch_mode,
            {
                **common_kwargs,
                'file2': file2,
                'min_dist': getattr(args, 'min_dist', 1),
                'max_dist': getattr(args, 'max_dist', 1),
                'show_dist': getattr(args, 'show_dist', False),
                'output_format': output_format,
            },
        ),
        'stats': (
            stats_mode,
            {
                **common_kwargs,
                'include_pairs': getattr(args, 'pairs', False),
                'output_format': output_format,
            },
        ),
        'discovery': (
            discovery_mode,
            {
                **common_kwargs,
                'rare_max': getattr(args, 'rare_max', 1),
                'freq_min': getattr(args, 'freq_min', 5),
                'min_dist': getattr(args, 'min_dist', 1),
                'max_dist': getattr(args, 'max_dist', 1),
                'show_dist': getattr(args, 'show_dist', False),
                'output_format': output_format,
            },
        ),
        'highlight': (
            highlight_mode,
            {
                **common_kwargs,
                'mapping_file': getattr(args, 'mapping', ''),
                'smart': getattr(args, 'smart', False),
            }
        ),
    }

    handler, handler_args = handler_map[args.mode]
    try:
        handler(**handler_args)
    except FileNotFoundError as e:
        # If the exception has a filename attribute (common in OSError), use it.
        # Otherwise, fall back to a generic message.
        filename = getattr(e, 'filename', None)
        if filename:
            logging.error(f"File not found: '{filename}'")
        else:
            logging.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
