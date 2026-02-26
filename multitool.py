import argparse
import csv
import glob
from collections import Counter, defaultdict
import random
import contextlib
import sys
import re
from textwrap import dedent
from typing import Any, Callable, Iterable, List, Sequence, Tuple, TextIO, Union
from tqdm import tqdm
import logging
import ahocorasick
import json

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

# Disable colors if not running in a terminal
# We check both stdout and stderr as help goes to stdout and logging/stats to stderr
if not sys.stdout.isatty():
    BLUE = GREEN = RED = YELLOW = RESET = BOLD = ""
# Note: we use stdout's status for the global constants, but individual
# functions might still check stderr if they specifically log to it.


def filter_to_letters(text: str) -> str:
    """Return text containing only lowercase a-z characters."""
    return re.sub("[^a-z]", "", text.lower())


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
    lines = []
    used_encoding = 'utf-8'

    if path == '-':
        # For stdin, we rely on sys.stdin which is already open.
        try:
            lines = sys.stdin.readlines()
            used_encoding = sys.stdin.encoding or 'utf-8'
        except UnicodeDecodeError:
            logging.warning("Reading from stdin failed with encoding errors.")
            # Fallback logic for stdin is complex without buffering.
            # We assume valid text stream or return empty.
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
                cleaned_items.append(part)

    if apply_length_filter:
        cleaned_items = [
            item for item in cleaned_items if min_length <= len(item) <= max_length
        ]

    unique_items = list(dict.fromkeys(cleaned_items))
    return raw_items, cleaned_items, unique_items


def print_processing_stats(
    raw_item_count: int, filtered_items: Sequence[str], item_label: str = "item"
) -> None:
    """Print summary statistics for processed text items with visual hierarchy."""
    item_label_plural = f"{item_label}s"

    # Colors for stderr logging
    c_bold = BOLD if sys.stderr.isatty() else ""
    c_yellow = YELLOW if sys.stderr.isatty() else ""
    c_green = GREEN if sys.stderr.isatty() else ""
    c_reset = RESET if sys.stderr.isatty() else ""

    logging.info(f"\n{c_bold}ANALYSIS STATISTICS{c_reset}")
    logging.info(f"{c_bold}───────────────────────────────────────────────────────{c_reset}")
    logging.info(f"  {c_bold}{'Total ' + item_label_plural + ' encountered:':<35}{c_reset} {c_yellow}{raw_item_count}{c_reset}")
    logging.info(f"  {c_bold}{'Total ' + item_label_plural + ' after filtering:':<35}{c_reset} {c_green}{len(filtered_items)}{c_reset}")

    if raw_item_count > 0:
        retention = (len(filtered_items) / raw_item_count) * 100
        logging.info(f"  {BOLD}{'Retention rate:':<35}{RESET} {GREEN}{retention:.1f}%{RESET}")

    if filtered_items:
        shortest = min(filtered_items, key=len)
        longest = max(filtered_items, key=len)
        logging.info(
            f"  {BOLD}{'Shortest ' + item_label + ':':<35}{RESET} '{shortest}' (length: {len(shortest)})"
        )
        logging.info(
            f"  {BOLD}{'Longest ' + item_label + ':':<35}{RESET} '{longest}' (length: {len(longest)})"
        )
    else:
        logging.info(f"  {YELLOW}No {item_label_plural} passed the filtering criteria.{RESET}")
    logging.info("")


@contextlib.contextmanager
def smart_open_output(filename: str, encoding: str = 'utf-8', newline: str | None = None) -> Iterable[TextIO]:
    """
    Context manager that yields a file object for writing.
    If filename is '-', yields sys.stdout.
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
        output_file: Path to the output file or '-' for stdout.
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
                                if 'typo' in item and 'correct' in item:
                                    yield str(item['typo']), str(item['correct'])
                        else:
                            for k, v in data.items():
                                yield str(k), str(v)
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'typo' in item and 'correct' in item:
                                yield str(item['typo']), str(item['correct'])
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
                                if 'typo' in item and 'correct' in item:
                                    yield str(item['typo']), str(item['correct'])
                                else:
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

            if content.startswith('|') and content.count('|') >= 2:
                # Potential Markdown table row
                parts = [p.strip() for p in content.split('|')]
                # Filter out empty parts from edges if they exist
                if parts and not parts[0]: parts = parts[1:]
                if parts and not parts[-1]: parts = parts[:-1]

                if len(parts) >= 2:
                    # Skip divider lines like | --- | --- |
                    if all(re.match(r'^:?-+:?$', p) for p in parts):
                        continue
                    # Skip header line if it contains generic labels
                    if parts[0].lower() in ('typo', 'left', 'word 1', 'item') and \
                       parts[1].lower() in ('correction', 'right', 'word 2', 'count', 'corrections'):
                        continue
                    yield parts[0], parts[1]
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
        output_file: Path to the output file or '-' for stdout.
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
            left_header = "Left"
            right_header = "Right"
            if mode_label == "Conflict":
                left_header = "Typo"
                right_header = "Corrections"
            elif mode_label in ("Similarity", "Pairs", "Swap", "Zip"):
                left_header = "Typo"
                right_header = "Correction"
            elif mode_label == "NearDuplicates":
                left_header = "Word 1"
                right_header = "Word 2"

            out_file.write(f"| {left_header} | {right_header} |\n")
            out_file.write("| :--- | :--- |\n")
            for left, right in pairs_list:
                out_file.write(f"| {left} | {right} |\n")
        else:  # 'arrow' or 'line' or fallback
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

    def chained_extractor() -> Iterable[str]:
        for input_file in input_files:
            yield from extractor_func(input_file, quiet=quiet)

    raw_items = list(chained_extractor())
    filtered_items = clean_and_filter(raw_items, min_length, max_length, clean=clean_items)

    if process_output:
        # Note: If not cleaning, duplicates might differ by case/whitespace if user wants that.
        # But process_output implies "normalize, sort, dedup".
        # If clean_items is False, we just sort and dedup raw strings.
        filtered_items = sorted(set(filtered_items))

    write_output(filtered_items, output_file, output_format, quiet, limit=limit)

    print_processing_stats(len(raw_items), filtered_items)
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
        marker_found = False
        for index in range(1, len(parts), 2):
            item = parts[index].strip()
            if not item:
                continue

            candidates.append(item)
            preceding = parts[index - 1].lower()
            if any(marker in preceding for marker in context_markers):
                marker_found = True

            if marker_found:
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
    input_file: str, first_column: bool, delimiter: str = ',', quiet: bool = False
) -> Iterable[str]:
    """Yield fields from CSV rows based on column selection."""
    lines = _read_file_lines_robust(input_file)
    reader = csv.reader(lines, delimiter=delimiter)
    for row in tqdm(reader, desc=f'Processing {input_file} (CSV)', unit=' rows', disable=quiet):
        if first_column:
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


def _extract_md_table_items(input_file: str, right_side: bool = False, quiet: bool = False) -> Iterable[str]:
    """Yield text from a specific column in Markdown tables."""
    lines = _read_file_lines_robust(input_file)
    for line in tqdm(lines, desc=f'Processing {input_file} (md-table)', unit=' lines', disable=quiet):
        content = line.strip()
        if content.startswith('|') and content.count('|') >= 2:
            parts = [p.strip() for p in content.split('|')]
            if parts and not parts[0]: parts = parts[1:]
            if parts and not parts[-1]: parts = parts[:-1]

            if len(parts) >= 2:
                # Skip divider lines
                if all(re.match(r'^:?-+:?$', p) for p in parts):
                    continue
                # Skip headers
                if parts[0].lower() in ('typo', 'left', 'word 1', 'item') and \
                   parts[1].lower() in ('correction', 'right', 'word 2', 'count', 'corrections'):
                    continue

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
    extractor = lambda f, quiet=False: _extract_arrow_items(f, right_side=right_side, quiet=quiet)
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
    extractor = lambda f, quiet=False: _extract_table_items(f, right_side=right_side, quiet=quiet)
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
    extractor = lambda f, quiet=False: _extract_markdown_items(f, right_side=right_side, quiet=quiet)
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
) -> None:
    """Wrapper for processing items from Markdown tables."""
    extractor = lambda f, quiet=False: _extract_md_table_items(f, right_side=right_side, quiet=quiet)
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
    extractor = lambda f, quiet=False: _extract_json_items(f, key, quiet=quiet)
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
    extractor = lambda f, quiet=False: _extract_yaml_items(f, key, quiet=quiet)
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
) -> None:
    """
    Counts the frequency of each word in the input file(s) and writes the
    sorted results to the output file. Only words with length between
    min_length and max_length are counted.
    The stats are based on the raw count of words versus the filtered words.
    Note: process_output is ignored in count mode.
    """
    raw_count = 0
    filtered_words = []
    word_counts = Counter()

    for input_file in input_files:
        lines = _read_file_lines_robust(input_file)
        for line in tqdm(lines, desc=f'Counting words in {input_file}', unit=' lines', disable=quiet):
            words = line.split()
            raw_count += len(words)
            filtered = clean_and_filter(words, min_length, max_length, clean=clean_items)
            filtered_words.extend(filtered)
            word_counts.update(filtered)

    sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))

    # Apply frequency filtering
    final_results = []
    for word, count in sorted_words:
        if count < min_count:
            continue
        if max_count is not None and count > max_count:
            continue
        final_results.append((word, count))

    if limit is not None:
        final_results = final_results[:limit]

    # Determine newline behavior for CSV
    newline = '' if output_format == 'csv' else None

    with smart_open_output(output_file, newline=newline) as out_file:
        if output_format == 'json':
            json_data = [{"item": word, "count": count} for word, count in final_results]
            json.dump(json_data, out_file, indent=2)
            out_file.write('\n')
        elif output_format == 'csv':
            writer = csv.writer(out_file)
            for word, count in final_results:
                writer.writerow([word, count])
        elif output_format == 'markdown':
            for word, count in final_results:
                out_file.write(f"- {word}: {count}\n")
        elif output_format == 'md-table':
            out_file.write("| Item | Count |\n")
            out_file.write("| :--- | :--- |\n")
            for word, count in final_results:
                out_file.write(f"| {word} | {count} |\n")
        else:  # 'line' or fallback
            for word, count in final_results:
                out_file.write(f"{word}: {count}\n")

    print_processing_stats(raw_count, filtered_words, item_label="word")
    logging.info(
        f"[Count Mode] Word frequencies ({len(final_results)} items) have been written to '{output_file}' in {output_format} format."
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
            f.write("### ANALYSIS STATISTICS\n\n")
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
            report.append(f"\n{c_bold}ANALYSIS STATISTICS{c_reset}")
            report.append(f"{c_bold}───────────────────────────────────────────────────────{c_reset}")

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

            if "pairs" in stats:
                report.append(f"\n{c_bold}PAIRED DATA STATISTICS{c_reset}")
                report.append(f"{c_bold}───────────────────────────────────────────────────────{c_reset}")
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

    print_processing_stats(len(duplicates), filtered_items)
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

    logging.info(f"[Conflict Mode] Found {len(conflicts)} typos with conflicting corrections. Output written to '{output_file}'.")


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

    for i in range(num_items):
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

    print_processing_stats(raw_item_count, [pair[0] for pair in results] + [pair[1].rsplit(' (changes: ', 1)[0] for pair in results], item_label="near-duplicate")


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

    print_processing_stats(raw_item_count, [pair[0] for pair in results] + [pair[1].rsplit(' (changes: ', 1)[0] for pair in results], item_label="fuzzy-match")


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
) -> None:
    """Wrapper for extracting fields from CSV files."""
    extractor = lambda f, quiet=False: _extract_csv_items(f, first_column, delimiter, quiet=quiet)
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

    print_processing_stats(raw_item_count, combined_unique)
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

    print_processing_stats(raw_item_count, final_items)
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
        # Skip if both are empty after cleaning (if cleaning enabled)
        if not left and not right:
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

    raw_pairs = _extract_pairs(input_files, quiet=quiet)

    filtered_pairs = []
    for left, right in raw_pairs:
        # Clean if requested
        if clean_items:
            left = filter_to_letters(left)
            right = filter_to_letters(right)

        # Skip if both are empty after cleaning
        if not left and not right:
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

    raw_pairs = _extract_pairs(input_files, quiet=quiet)

    filtered_pairs = []
    for left, right in raw_pairs:
        # Swap
        new_left, new_right = right, left

        # Clean if requested
        if clean_items:
            new_left = filter_to_letters(new_left)
            new_right = filter_to_letters(new_right)

        # Skip if both are empty after cleaning
        if not new_left and not new_right:
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

    def chained_extractor() -> Iterable[str]:
        for input_file in input_files:
            yield from _extract_line_items(input_file, quiet=quiet)

    # Extract raw items first
    raw_items = list(chained_extractor())

    if not raw_items:
        logging.warning("Input is empty or no lines found.")
        # Create empty output using write_output to ensure consistent formatting (e.g. empty JSON list)
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

    print_processing_stats(len(raw_items), sampled_items)
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
    # because users often want exact matches (e.g. Emails, URLs, IDs).
    # Users can still use --process-output to sort/dedup, but we don't force lowercase/clean.
    extractor = lambda f, quiet=False: _extract_regex_items(f, pattern, quiet=quiet)
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
) -> None:
    """
    Transforms items based on a mapping file.
    """
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
            # If output is also cleaned later (e.g. by process_output)?
            # But here we are producing the item.
            # Let's keep value as is, but clean key.
            if cleaned_k:
                cleaned_mapping[cleaned_k] = v
        mapping = cleaned_mapping
    else:
        mapping = raw_mapping

    raw_item_count = 0
    transformed_items = []

    for input_file in input_files:
        # We reuse _load_and_clean_file logic to get items
        # But _load_and_clean_file returns unique items in the third return value.
        # We probably want raw_items (first return) to preserve count/order if process_output is False?
        # Or we want cleaned_items (second return)?
        # If clean_items=True, we want cleaned_items.
        # If clean_items=False, we want raw_items.

        raw, cleaned, _ = _load_and_clean_file(
            input_file,
            min_length,
            max_length,
            clean_items=clean_items,
        )

        source_items = cleaned
        raw_item_count += len(source_items)

        for item in source_items:
            if item in mapping:
                transformed = mapping[item]
                # Re-apply length filtering to the result of the mapping
                if min_length <= len(transformed) <= max_length:
                    transformed_items.append(transformed)
            elif not drop_missing:
                # item already passed length filter in _load_and_clean_file
                transformed_items.append(item)

    if process_output:
        # If processing output, we sort and dedup.
        # Also clean? The output might be "The" which is not "the".
        # If user wanted clean output, they got clean input.
        # The map result might introduce non-clean items.
        # Usually process_output implies sorting and deduping.
        # multitool usually assumes clean items for set operations.
        # Here we trust the map result.
        transformed_items = sorted(set(transformed_items))

    write_output(transformed_items, output_file, output_format, quiet, limit=limit)

    print_processing_stats(raw_item_count, transformed_items, item_label="item")
    logging.info(
        f"[Map Mode] Transformed items using '{mapping_file}'. Output written to '{output_file}'."
    )


def _add_common_mode_arguments(
    subparser: argparse.ArgumentParser, include_process_output: bool = True
) -> None:
    """Attach shared CLI arguments to a mode-specific subparser."""

    # Positional input arguments stay in the default group for prominence
    subparser.add_argument(
        'input_files_pos',
        nargs='*',
        metavar='FILE',
        help="Path(s) to the input file(s). Defaults to stdin ('-') if none provided.",
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
        help="Where to save the results. Use '-' for the screen (default: screen).",
    )
    io_group.add_argument(
        '-f', '--output-format', '--format',
        dest='output_format',
        choices=['line', 'json', 'csv', 'markdown', 'md-table', 'arrow', 'table', 'yaml'],
        default=argparse.SUPPRESS,
        help="Choose the format for the output (default: line).",
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
    proc_group.add_argument(
        '-L', '--limit',
        type=int,
        help="Limit the number of items in the output.",
    )
    if include_process_output:
        proc_group.add_argument(
            '-P', '--process-output',
            action='store_true',
            default=argparse.SUPPRESS,
            help="Sort the list and remove duplicates.",
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

    print_processing_stats(len(all_raw_list1), filtered_items)
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

    print_processing_stats(raw_item_count_a + len(raw_items_b), result_items)
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
        "description": "Gets data from CSV files. By default, it extracts every column except the first one. Use --first-column to get only the first column.",
        "example": "python multitool.py csv typos.csv -o corrections.txt",
        "flags": "[--first-column]",
    },
    "markdown": {
        "summary": "Extracts items from Markdown bulleted lists.",
        "description": "Finds text in lines starting with -, *, or +. It can also split items by ':' or '->' to extract one side of a pair (use --right for the second part).",
        "example": "python multitool.py markdown notes.md --output items.txt",
        "flags": "[--right]",
    },
    "md-table": {
        "summary": "Extracts text from Markdown tables.",
        "description": "Finds text in cells of a Markdown table. It saves the first column by default. Use --right to save the second column instead. It automatically skips header and divider rows.",
        "example": "python multitool.py md-table readme.md --right --output corrections.txt",
        "flags": "[--right]",
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
    "count": {
        "summary": "Counts how many times each word appears.",
        "description": "Counts word frequency and sorts the list from most frequent to least frequent.",
        "example": "python multitool.py count typos.log --min-count 5 --output-format json --output counts.json",
        "flags": "[--min-count N]",
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
        "description": "Replaces items in your list with new values from a mapping file. Supports CSV, Arrow, Table, JSON, and YAML mapping formats.",
        "example": "python multitool.py map input.txt mapping.csv -o fixed.txt",
        "flags": "[MAPPING]",
    },
    "zip": {
        "summary": "Pairs lines from two files together.",
        "description": "Joins two files line-by-line into a paired format like 'typo -> correction'. Useful for creating mapping files from two separate lists.",
        "example": "python multitool.py zip typos.txt corrections.txt --output-format table --output typos.toml",
        "flags": "[FILE2]",
    },
    "swap": {
        "summary": "Reverses the order of elements in paired data.",
        "description": "Flips the left and right elements of pairs (e.g., 'typo -> correction' becomes 'correction -> typo'). Supports Arrow, Table, CSV, and Markdown formats.",
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
        "description": "Filters pairs (typo -> correction) based on the number of character changes needed to turn one word into another. Use this to remove noise or find specific types of typos.",
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
        "description": "Provides a comprehensive summary of your dataset. It reports counts, unique items, length distributions, and (optionally) paired data stats like conflicts, overlaps, and the number of changes between words.",
        "example": "python multitool.py stats typos.csv --pairs --output-format json",
        "flags": "[--pairs]",
    },
}


def get_mode_summary_text() -> str:
    """Return a formatted summary table of all available modes as a string."""
    categories = {
        "Extraction": ["arrow", "table", "backtick", "csv", "markdown", "md-table", "json", "yaml", "line", "regex"],
        "Manipulation": ["combine", "unique", "filterfragments", "set_operation", "sample", "map", "zip", "swap", "pairs"],
        "Analysis": ["count", "check", "conflict", "similarity", "near_duplicates", "fuzzymatch", "stats"],
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

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            return record.getMessage()
        return f"{record.levelname}: {record.getMessage()}"


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
            """
            Examples:
              python multitool.py --mode-help             # Show a summary of every mode
              python multitool.py --mode-help csv         # Describe the CSV extraction mode
              python multitool.py arrow file.txt          # Run a specific mode
              python multitool.py --mode csv --input file.txt  # Old way to run the tool
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
        help="Where to save the results. Use '-' for the screen (default: screen).",
    )
    io_group.add_argument(
        '-f', '--output-format', '--format',
        dest='output_format',
        choices=['line', 'json', 'csv', 'markdown', 'md-table', 'arrow', 'table', 'yaml'],
        default='line',
        help="Choose the format for the output (default: line).",
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
        '-R', '--raw',
        action='store_true',
        help="Keep the original text. Do not change it to lowercase or remove punctuation.",
    )

    subparsers = parser.add_subparsers(dest='mode', required=True, metavar='mode', help=argparse.SUPPRESS)

    arrow_parser = subparsers.add_parser(
        'arrow',
        help=MODE_DETAILS['arrow']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['arrow']['description'],
        epilog=f"Example:\n  {MODE_DETAILS['arrow']['example']}",
    )
    arrow_options = arrow_parser.add_argument_group("Arrow Options")
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
        epilog=f"Example:\n  {MODE_DETAILS['table']['example']}",
    )
    table_options = table_parser.add_argument_group("Table Options")
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
        epilog=f"Example:\n  {MODE_DETAILS['backtick']['example']}",
    )
    _add_common_mode_arguments(backtick_parser)

    csv_parser = subparsers.add_parser(
        'csv',
        help=MODE_DETAILS['csv']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['csv']['description'],
        epilog=f"Example:\n  {MODE_DETAILS['csv']['example']}",
    )
    csv_options = csv_parser.add_argument_group("CSV Options")
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
    _add_common_mode_arguments(csv_parser)

    markdown_parser = subparsers.add_parser(
        'markdown',
        help=MODE_DETAILS['markdown']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['markdown']['description'],
        epilog=f"Example:\n  {MODE_DETAILS['markdown']['example']}",
    )
    markdown_options = markdown_parser.add_argument_group("Markdown Options")
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
        epilog=f"Example:\n  {MODE_DETAILS['md-table']['example']}",
    )
    md_table_options = md_table_parser.add_argument_group("Markdown Table Options")
    md_table_options.add_argument(
        '--right',
        action='store_true',
        help="Extract the second column instead of the first.",
    )
    _add_common_mode_arguments(md_table_parser)

    json_parser = subparsers.add_parser(
        'json',
        help=MODE_DETAILS['json']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['json']['description'],
        epilog=f"Example:\n  {MODE_DETAILS['json']['example']}",
    )
    json_options = json_parser.add_argument_group("JSON Options")
    json_options.add_argument(
        '-k', '--key',
        type=str,
        default='',
        help="The key path to extract (e.g. 'items.name'). If omitted, extracts from the root.",
    )
    _add_common_mode_arguments(json_parser)

    yaml_parser = subparsers.add_parser(
        'yaml',
        help=MODE_DETAILS['yaml']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['yaml']['description'],
        epilog=f"Example:\n  {MODE_DETAILS['yaml']['example']}",
    )
    yaml_options = yaml_parser.add_argument_group("YAML Options")
    yaml_options.add_argument(
        '-k', '--key',
        type=str,
        default='',
        help="The key path to extract (e.g. 'config.items'). If omitted, extracts from the root.",
    )
    _add_common_mode_arguments(yaml_parser)

    combine_parser = subparsers.add_parser(
        'combine',
        help=MODE_DETAILS['combine']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['combine']['description'],
        epilog=f"Example:\n  {MODE_DETAILS['combine']['example']}",
    )
    _add_common_mode_arguments(combine_parser, include_process_output=False)

    unique_parser = subparsers.add_parser(
        'unique',
        help=MODE_DETAILS['unique']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['unique']['description'],
        epilog=f"Example:\n  {MODE_DETAILS['unique']['example']}",
    )
    _add_common_mode_arguments(unique_parser)

    line_parser = subparsers.add_parser(
        'line',
        help=MODE_DETAILS['line']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['line']['description'],
        epilog=f"Example:\n  {MODE_DETAILS['line']['example']}",
    )
    _add_common_mode_arguments(line_parser)

    count_parser = subparsers.add_parser(
        'count',
        help=MODE_DETAILS['count']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['count']['description'],
        epilog=f"Example:\n  {MODE_DETAILS['count']['example']}",
    )
    count_options = count_parser.add_argument_group("Count Options")
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
    _add_common_mode_arguments(count_parser, include_process_output=False)

    filter_parser = subparsers.add_parser(
        'filterfragments',
        help=MODE_DETAILS['filterfragments']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['filterfragments']['description'],
        epilog=f"Example:\n  {MODE_DETAILS['filterfragments']['example']}",
    )
    filter_options = filter_parser.add_argument_group("Filter Options")
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
        epilog=f"Example:\n  {MODE_DETAILS['check']['example']}",
    )
    _add_common_mode_arguments(check_parser)

    conflict_parser = subparsers.add_parser(
        'conflict',
        help=MODE_DETAILS['conflict']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['conflict']['description'],
        epilog=f"Example:\n  {MODE_DETAILS['conflict']['example']}",
    )
    _add_common_mode_arguments(conflict_parser)

    similarity_parser = subparsers.add_parser(
        'similarity',
        help=MODE_DETAILS['similarity']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['similarity']['description'],
        epilog=f"Example:\n  {MODE_DETAILS['similarity']['example']}",
    )
    similarity_options = similarity_parser.add_argument_group("Similarity Options")
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
        epilog=f"Example:\n  {MODE_DETAILS['near_duplicates']['example']}",
    )
    nd_options = near_duplicates_parser.add_argument_group("Near Duplicate Options")
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
        epilog=f"Example:\n  {MODE_DETAILS['fuzzymatch']['example']}",
    )
    fm_options = fuzzymatch_parser.add_argument_group("Fuzzy Match Options")
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
        epilog=f"Example:\n  {MODE_DETAILS['stats']['example']}",
    )
    stats_options = stats_parser.add_argument_group("Stats Options")
    stats_options.add_argument(
        '-p', '--pairs',
        action='store_true',
        help="Perform pair-level analysis (typos vs corrections) in addition to item-level stats.",
    )
    _add_common_mode_arguments(stats_parser, include_process_output=False)

    set_parser = subparsers.add_parser(
        'set_operation',
        help=MODE_DETAILS['set_operation']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['set_operation']['description'],
        epilog=f"Example:\n  {MODE_DETAILS['set_operation']['example']}",
    )
    set_options = set_parser.add_argument_group("Set Operation Options")
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
        epilog=f"Example:\n  {MODE_DETAILS['sample']['example']}",
    )
    sample_options = sample_parser.add_argument_group("Sample Options")
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

    regex_parser = subparsers.add_parser(
        'regex',
        help=MODE_DETAILS['regex']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['regex']['description'],
        epilog=f"Example:\n  {MODE_DETAILS['regex']['example']}",
    )
    regex_options = regex_parser.add_argument_group("Regex Options")
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
        epilog=f"Example:\n  {MODE_DETAILS['map']['example']}",
    )
    map_options = map_parser.add_argument_group("Map Options")
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
    _add_common_mode_arguments(map_parser)

    zip_parser = subparsers.add_parser(
        'zip',
        help=MODE_DETAILS['zip']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['zip']['description'],
        epilog=f"Example:\n  {MODE_DETAILS['zip']['example']}",
    )
    zip_options = zip_parser.add_argument_group("Zip Options")
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
        epilog=f"Example:\n  {MODE_DETAILS['swap']['example']}",
    )
    _add_common_mode_arguments(swap_parser)

    pairs_parser = subparsers.add_parser(
        'pairs',
        help=MODE_DETAILS['pairs']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['pairs']['description'],
        epilog=f"Example:\n  {MODE_DETAILS['pairs']['example']}",
    )
    _add_common_mode_arguments(pairs_parser)

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
        logging.error("[Error] --min-length must be a positive integer.")
        sys.exit(1)
    if args.max_length < args.min_length:
        logging.error("[Error] --max-length must be greater than or equal to --min-length.")
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

    # Default to stdin ('-') if neither is provided
    if not input_paths:
        input_paths = ['-']

    # Store for handler
    args.input = input_paths

    # Fallback logic for modes that require a secondary file (e.g., zip, map)
    # If the flag is missing but we have at least 2 positional arguments, use the last one as the secondary file.
    if args.mode in {'zip', 'filterfragments', 'set_operation', 'fuzzymatch'}:
        if getattr(args, 'file2', None) is None and len(input_paths) >= 2:
            args.file2 = input_paths.pop()
            args.input = input_paths
    elif args.mode == 'map':
        if getattr(args, 'mapping', None) is None and len(input_paths) >= 2:
            args.mapping = input_paths.pop()
            args.input = input_paths

    file2 = getattr(args, 'file2', None)
    # Check for missing secondary files after fallback attempt
    if args.mode in {'zip', 'filterfragments', 'set_operation'} and file2 is None:
        logging.error(f"[Error] {args.mode.capitalize()} mode requires a secondary file (provide FILE2 positionally or use --file2).")
        sys.exit(1)
    if args.mode == 'map' and getattr(args, 'mapping', None) is None:
        logging.error("[Error] Map mode requires a mapping file (provide MAPPING positionally or use --mapping).")
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
        'md-table': (
            md_table_mode,
            {
                **common_kwargs,
                'right_side': right_side,
                'output_format': output_format,
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
        'count': (
            count_mode,
            {
                **common_kwargs,
                'min_count': getattr(args, 'min_count', 1),
                'max_count': getattr(args, 'max_count', None),
                'output_format': output_format,
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
    }

    handler, handler_args = handler_map[args.mode]
    try:
        handler(**handler_args)
    except FileNotFoundError as e:
        # If the exception has a filename attribute (common in OSError), use it.
        # Otherwise, fall back to a generic message.
        filename = getattr(e, 'filename', None)
        if filename:
            logging.error(f"[Error] File not found: '{filename}'")
        else:
            logging.error(f"[Error] File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"[Error] An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
