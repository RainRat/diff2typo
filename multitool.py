import os
import shutil
import argparse
import csv
import glob
import difflib
from collections import Counter, defaultdict, deque
import random
import contextlib
import sys
import re
import time
from textwrap import dedent
from typing import Any, Callable, Iterable, List, Mapping, Sequence, Tuple, TextIO, Optional
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

try:
    import tomllib
    _TOMLLIB_AVAILABLE = True
    _TOML_AVAILABLE = False
except ImportError:
    _TOMLLIB_AVAILABLE = False
    import importlib.util
    _TOML_AVAILABLE = importlib.util.find_spec("toml") is not None

# Cache for standard input to allow multiple passes
_STDIN_CACHE: List[str] | None = None

# ANSI Color Codes
BLUE = "\033[1;34m"
GREEN = "\033[1;32m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
MAGENTA = "\033[1;35m"
CYAN = "\033[1;36m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Disable colors if not running in a terminal or if NO_COLOR is set
# We check the main output and error output as help goes to the main output and logging/stats to error output
if (not sys.stdout.isatty() and 'FORCE_COLOR' not in os.environ) or 'NO_COLOR' in os.environ:
    BLUE = GREEN = RED = YELLOW = MAGENTA = CYAN = RESET = BOLD = ""
# Note: we use the main output's status for the global constants, but individual
# functions might still check the error output if they specifically log to it.


def _should_enable_color(stream: TextIO) -> bool:
    """Determine if ANSI color codes should be enabled for a given stream."""
    if 'NO_COLOR' in os.environ:
        return False
    if 'FORCE_COLOR' in os.environ:
        return True
    return stream.isatty()


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


def _detect_format_from_extension(path: str, allowed: Sequence[str], default: str) -> str:
    """
    Detect the output format based on the file extension.
    Returns the default if no match is found or no extension is present.
    """
    if not path or path == '-':
        return default

    ext = os.path.splitext(path)[1].lower().lstrip('.')
    if not ext:
        return default

    # Map common extensions to tool-supported formats
    mapping = {
        'txt': 'line',
        'json': 'json',
        'csv': 'csv',
        'md': 'markdown',
        'yaml': 'yaml',
        'yml': 'yaml',
        'toml': 'toml',
        'arrow': 'arrow',
        'table': 'table',
    }

    detected = mapping.get(ext)
    if detected in allowed:
        return detected

    return default


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
    Groups a typo based on its relationship to the correction.
    Returns a code: [K] Keyboard, [T] Transposition, [D] Deletion, [I] Insertion, [R] Replacement, [M] Multiple letters.
    """
    if not typo or not correction or typo == correction:
        return "[?]"

    t_len, c_len = len(typo), len(correction)
    len_diff = t_len - c_len

    # Handle same-length operations (Transposition, Keyboard, Replacement)
    if len_diff == 0:
        diffs = [i for i in range(t_len) if typo[i] != correction[i]]
        # 1. Transposition [T]
        if len(diffs) == 2 and diffs[1] == diffs[0] + 1:
            i, j = diffs
            if typo[i] == correction[j] and typo[j] == correction[i]:
                return "[T]"
        # 4. Replacement [R] or [K]
        if len(diffs) == 1:
            idx = diffs[0]
            t_char, c_char = typo[idx].lower(), correction[idx].lower()
            if t_char in adj_keys.get(c_char, set()):
                return "[K]"
            return "[R]"

        return "[M]"

    # 2. Deletion [D] - Typo is shorter (a character was removed)
    if len_diff == -1:
        for i in range(c_len):
            if correction[:i] + correction[i+1:] == typo:
                return "[D]"
        return "[M]"

    # 3. Insertion [I] - Typo is longer (a character was added)
    if len_diff == 1:
        for i in range(t_len):
            if typo[:i] + typo[i+1:] == correction:
                return "[I]"
        return "[M]"

    # 5. Multiple letters [M] - Fallback for any other length difference or non-match
    return "[M]"


def _smart_split(text: str) -> List[str]:
    """
    Splits text into subwords based on non-alphanumeric characters
    and casing boundaries (CamelCase).
    """
    # Use re.findall with a pattern that matches subwords directly,
    # implicitly skipping any non-alphanumeric delimiters.
    return re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+', text)


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

def _get_total_line_count(input_files: Sequence[str]) -> int:
    """Efficiently counts total lines across multiple files, respecting the stdin cache."""
    total = 0
    for f in input_files:
        if f == '-':
            total += len(_read_file_lines_robust('-'))
        else:
            try:
                # Use a fast byte-counting approach for regular files
                with open(f, 'rb') as fp:
                    # buffer-aware line counting is generally faster than sum(1 for _ in fp)
                    # but sum(1 for line in fp) is clear and optimized in Python 3
                    total += sum(1 for _ in fp)
            except Exception:
                pass
    return total


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
        if not os.path.exists(path):
            logging.error(f"Input file '{path}' not found.")
            sys.exit(1)

        if os.path.isdir(path):
            logging.warning(f"Input path '{path}' is a directory. Skipping.")
            return []

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


def _format_analysis_summary(
    raw_count: int,
    filtered_items: Sequence[Any],
    item_label: str = "item",
    start_time: float | None = None,
    use_color: bool = False,
    extra_metrics: Mapping[str, Any] | None = None,
    title: str = "ANALYSIS SUMMARY",
) -> List[str]:
    """
    Standardizes the "ANALYSIS SUMMARY" block with consistent colors and a visual retention bar.
    Returns a list of formatted lines.
    """
    item_label_plural = f"{item_label}s"
    c_bold = BOLD if use_color else ""
    c_blue = BLUE if use_color else ""
    c_green = GREEN if use_color else ""
    c_yellow = YELLOW if use_color else ""
    c_reset = RESET if use_color else ""

    padding = "  "
    label_width = 35
    report = []

    report.append(f"\n{padding}{c_bold}{c_blue}{title}{c_reset}")
    report.append(f"{padding}{c_bold}{c_blue}───────────────────────────────────────────────────────{c_reset}")

    report.append(
        f"  {c_bold}{c_blue}{'Total ' + item_label_plural + ' analyzed:':<{label_width}}{c_reset} {c_yellow}{raw_count}{c_reset}"
    )

    filtered_count = len(filtered_items)
    report.append(
        f"  {c_bold}{c_blue}{'Total ' + item_label_plural + ' after filtering:':<{label_width}}{c_reset} {c_green}{filtered_count}{c_reset}"
    )

    if raw_count > 0:
        retention = (filtered_count / raw_count) * 100
        # High-res visual bar for retention
        max_bar = 20
        total_blocks = (retention * max_bar) / 100
        full_blocks = int(total_blocks)
        fraction = total_blocks - full_blocks
        blocks = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
        frac_idx = int(fraction * 8)

        bar = "█" * full_blocks
        if full_blocks < max_bar:
            bar += blocks[frac_idx]
            bar += " " * (max_bar - full_blocks - 1)

        report.append(
            f"  {c_bold}{c_blue}{'Retention rate:':<{label_width}}{c_reset} {c_green}{retention:>5.1f}%{c_reset} {c_blue}{bar}{c_reset}"
        )

    # Unique Items
    try:
        # Check if items are hashable (like strings or tuples of strings)
        unique_count = len(set(filtered_items))
    except (TypeError, ValueError):
        unique_count = len(filtered_items)

    report.append(
        f"  {c_bold}{c_blue}{'Unique ' + item_label_plural + ':':<{label_width}}{c_reset} {c_green}{unique_count}{c_reset}"
    )

    # Shortest/Longest and stats
    if filtered_items:

        def format_item(it: Any) -> str:
            if isinstance(it, tuple):
                if len(it) == 2:
                    return f"{it[0]} -> {it[1]}"
                if len(it) == 3:
                    return f"{it[0]} -> {it[1]} {it[2]}"
            return str(it)

        try:
            lengths = [len(format_item(it)) for it in filtered_items]
            if lengths:
                min_len = min(lengths)
                max_len = max(lengths)
                avg_len = sum(lengths) / len(lengths)
                report.append(
                    f"  {c_bold}{c_blue}{'Min/Max/Avg length:':<{label_width}}{c_reset} {min_len} / {max_len} / {avg_len:.1f}"
                )

            shortest = min(filtered_items, key=lambda x: len(format_item(x)))
            longest = max(filtered_items, key=lambda x: len(format_item(x)))

            s_display = format_item(shortest)
            l_display = format_item(longest)

            report.append(
                f"  {c_bold}{c_blue}{'Shortest ' + item_label + ':':<{label_width}}{c_reset} '{s_display}' (length: {len(s_display)})"
            )
            report.append(
                f"  {c_bold}{c_blue}{'Longest ' + item_label + ':':<{label_width}}{c_reset} '{l_display}' (length: {len(l_display)})"
            )
        except (ValueError, TypeError):
            pass

    # Paired data distances
    if (
        filtered_items
        and isinstance(filtered_items[0], tuple)
        and len(filtered_items[0]) >= 2
    ):
        try:
            distances = [levenshtein_distance(str(p[0]), str(p[1])) for p in filtered_items]
            if distances:
                min_dist = min(distances)
                max_dist = max(distances)
                avg_dist = sum(distances) / len(distances)
                report.append(
                    f"  {c_bold}{c_blue}{'Min/Max/Avg changes:':<{label_width}}{c_reset} {min_dist} / {max_dist} / {avg_dist:.1f}"
                )
        except Exception:
            pass

    # Extra metrics
    if extra_metrics:
        for label, value in extra_metrics.items():
            report.append(f"  {c_bold}{c_blue}{label + ':':<{label_width}}{c_reset} {value}")

    if not filtered_items:
        report.append(
            f"  {c_yellow}No {item_label_plural} passed the filtering criteria.{c_reset}"
        )

    # Processing Time
    if start_time is not None:
        duration = time.perf_counter() - start_time
        report.append(
            f"  {c_bold}{c_blue}{'Processing time:':<{label_width}}{c_reset} {c_green}{duration:.3f}s{c_reset}"
        )

    report.append("")
    return report


def print_processing_stats(
    raw_item_count: int,
    filtered_items: Sequence[Any],
    item_label: str = "item",
    start_time: float | None = None,
) -> None:
    """Print summary statistics for processed text items with visual hierarchy."""
    use_color = _should_enable_color(sys.stderr)

    report = _format_analysis_summary(
        raw_item_count, filtered_items, item_label, start_time, use_color
    )
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
        elif output_format == 'toml':
            # Alias for table format for simple lists
            for item in items_list:
                outfile.write(f'{item} = ""\n')
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

        if ext.endswith('.toml'):
            if not _TOMLLIB_AVAILABLE and not _TOML_AVAILABLE:
                logging.error("TOML support requires Python 3.11+ or the 'toml' package.")
                continue
            content = "".join(_read_file_lines_robust(input_file))
            if content.strip():
                try:
                    if _TOMLLIB_AVAILABLE:
                        data = tomllib.loads(content)
                    else:
                        import toml
                        data = toml.loads(content)

                    if isinstance(data, dict):
                        if 'replacements' in data:
                            repls = data['replacements']
                            if isinstance(repls, list):
                                for item in repls:
                                    if isinstance(item, dict) and 'typo' in item:
                                        correct = item.get('correct', item.get('correction'))
                                        if correct is not None:
                                            yield str(item['typo']), str(correct)
                            elif isinstance(repls, dict):
                                for k, v in repls.items():
                                    if isinstance(v, list):
                                        for item in v:
                                            if isinstance(item, dict) and 'typo' in item:
                                                correct = item.get('correct', item.get('correction'))
                                                if correct is not None:
                                                    yield str(item['typo']), str(correct)
                                    elif not isinstance(v, dict):
                                        yield str(k), str(v)
                        else:
                            for k, v in data.items():
                                if not isinstance(v, (dict, list)):
                                    yield str(k), str(v)
                except Exception as e:
                    logging.error(f"Failed to parse TOML in '{input_file}': {e}")
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


def _write_diff_report(
    input_file: str,
    original_lines: List[str],
    modified_lines: List[str],
    out: TextIO,
) -> None:
    """Generate and write a colorized unified diff report."""
    # Strip newlines from lines for difflib compatibility
    orig_stripped = [line.rstrip('\n') for line in original_lines]
    mod_stripped = [line.rstrip('\n') for line in modified_lines]

    diff_gen = difflib.unified_diff(
        orig_stripped,
        mod_stripped,
        fromfile=f"a/{input_file}",
        tofile=f"b/{input_file}",
        lineterm=""
    )

    # Check if color is enabled for the output stream
    use_color = _should_enable_color(out)

    for line in diff_gen:
        if use_color:
            if line.startswith('+') and not line.startswith('+++'):
                out.write(f"{GREEN}{line}{RESET}\n")
            elif line.startswith('-') and not line.startswith('---'):
                out.write(f"{RED}{line}{RESET}\n")
            elif line.startswith('@@'):
                out.write(f"{BLUE}{line}{RESET}\n")
            else:
                out.write(f"{line}\n")
        else:
            out.write(f"{line}\n")


def _write_file_in_place(
    input_file: str,
    modified_lines: List[str],
    replacements: int,
    in_place_ext: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Handles in-place file modification with optional backup and dry-run reporting."""
    if replacements == 0:
        logging.info(f"No changes needed for '{input_file}'.")
        return

    if dry_run:
        logging.warning(f"[Dry Run] Would make {replacements} replacement(s) in '{input_file}'.")
        return

    # Backup if extension is provided
    if in_place_ext:
        backup_path = input_file + in_place_ext
        try:
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
        logging.info(f"Updated '{input_file}' in-place ({replacements} replacement(s)).")
    except Exception as e:
        logging.error(f"Failed to write to '{input_file}': {e}")
        sys.exit(1)


def _write_paired_output(
    pairs: Iterable[Tuple[str, ...]],
    output_file: str,
    output_format: str,
    mode_label: str,
    quiet: bool = False,
    limit: int | None = None,
    separator: str = " -> ",
) -> None:
    """
    Writes a collection of paired strings (optionally with attributes)
    to the output file in the specified format.

    Args:
        pairs: Collection of (left, right[, attr]) tuples.
        output_file: Path to the output file or '-' for the main output.
        output_format: Format (arrow, table, csv, markdown, md-table, json, yaml, aligned).
        mode_label: Label for the current mode (used for headers).
        quiet: If True, suppress informational output.
        limit: If provided, limit the output to the first N pairs.
        separator: The separator to use for 'aligned' format.
    """
    pairs_list = list(pairs)
    if limit is not None:
        pairs_list = pairs_list[:limit]

    # Detect if we have 3-tuples (attributes)
    has_attr = pairs_list and len(pairs_list[0]) == 3

    # Determine newline behavior for CSV
    newline = '' if output_format == 'csv' else None

    # Determine headers for paired data modes (used in md-table and arrow formats)
    left_header = "Left"
    right_header = "Right"
    attr_header = "Attr"
    if mode_label == "Conflict":
        left_header = "Typo"
        right_header = "Corrections"
    elif mode_label in ("Similarity", "Pairs", "Swap", "Zip", "Classify", "FuzzyMatch", "Discovery", "Map", "Resolve"):
        left_header = "Typo"
        right_header = "Correction"
    elif mode_label == "Rename":
        left_header = "Original"
        right_header = "New Name"
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
            if has_attr:
                json_data = {left: f"{right} {attr}".strip() for left, right, attr in pairs_list}
            else:
                json_data = {left: right for left, right in pairs_list}
            json.dump(json_data, out_file, indent=2)
            out_file.write('\n')
        elif output_format == 'yaml':
            try:
                import yaml
                if has_attr:
                    yaml_data = {left: f"{right} {attr}".strip() for left, right, attr in pairs_list}
                else:
                    yaml_data = dict(pairs_list)
                yaml.dump(yaml_data, out_file, default_flow_style=False, sort_keys=False)
            except ImportError:
                # Fallback to simple format if PyYAML not available
                for p in pairs_list:
                    left, right = p[0], p[1]
                    attr = p[2] if len(p) == 3 else ""
                    val = f"{right} {attr}".strip()
                    out_file.write(f"{left}: {val}\n")
        elif output_format == 'csv':
            writer = csv.writer(out_file)
            for p in pairs_list:
                writer.writerow(p)
        elif output_format in ('table', 'toml'):
            for p in pairs_list:
                left, right = p[0], p[1]
                attr = p[2] if len(p) == 3 else ""
                val = f"{right} {attr}".strip()
                out_file.write(f'{left} = "{val}"\n')
        elif output_format == 'markdown':
            for p in pairs_list:
                left, right = p[0], p[1]
                attr = p[2] if len(p) == 3 else ""
                val = f"{right} {attr}".strip()
                out_file.write(f"- {left}: {val}\n")
        elif output_format == 'md-table':
            if pairs_list:
                header = f"| {left_header} | {right_header} |"
                divider = "| :--- | :--- |"
                if has_attr:
                    header += f" {attr_header} |"
                    divider += " :--- |"
                out_file.write(header + "\n")
                out_file.write(divider + "\n")
                for p in pairs_list:
                    row = f"| {' | '.join(p)} |"
                    out_file.write(row + "\n")
        elif output_format == 'aligned':
            if pairs_list:
                # Calculate the maximum width of the left column for alignment
                max_left = max((len(str(p[0])) for p in pairs_list), default=0)
                for p in pairs_list:
                    left, right = p[0], p[1]
                    attr = p[2] if len(p) == 3 else ""
                    val = f"{right} {attr}".strip()
                    out_file.write(f"{left:<{max_left}}{separator}{val}\n")
        elif output_format == 'arrow':
            if pairs_list:
                # Dynamic column width calculation for aligned table
                max_left = max((len(str(p[0])) for p in pairs_list), default=len(left_header))
                max_left = max(max_left, len(left_header))
                max_right = max((len(str(p[1])) for p in pairs_list), default=len(right_header))
                max_right = max(max_right, len(right_header))

                # Colors for table
                show_color = _should_enable_color(out_file)
                c_bold = BOLD if show_color else ""
                c_blue = BLUE if show_color else ""
                c_green = GREEN if show_color else ""
                c_red = RED if show_color else ""
                c_cyan = CYAN if show_color else ""
                c_reset = RESET if show_color else ""

                # Header and divider
                padding = "  "
                sep = f"{c_bold}{c_blue}│{c_reset}"
                header = f"{padding}{c_bold}{c_blue}{left_header:<{max_left}}{c_reset} {sep} {c_bold}{c_blue}{right_header:<{max_right}}{c_reset}"
                visible_width = max_left + max_right + 3

                if has_attr:
                    max_attr = max((len(str(p[2])) for p in pairs_list), default=len(attr_header))
                    max_attr = max(max_attr, len(attr_header))
                    header += f" {sep} {c_bold}{c_blue}{attr_header:<{max_attr}}{c_reset}"
                    visible_width += 3 + max_attr

                divider = f"{padding}{c_bold}{c_blue}{'─' * visible_width}{c_reset}"

                out_file.write(f"\n{header}\n")
                out_file.write(f"{divider}\n")
                for p in pairs_list:
                    left, right = p[0], p[1]
                    row = f"{padding}{c_red}{left:<{max_left}}{c_reset} {sep} {c_green}{right:<{max_right}}{c_reset}"
                    if has_attr:
                        attr = p[2]
                        row += f" {sep} {c_cyan}{attr:<{max_attr}}{c_reset}"
                    out_file.write(row + "\n")
                out_file.write("\n")
        else:  # 'line' or fallback
            for p in pairs_list:
                left, right = p[0], p[1]
                attr = p[2] if len(p) == 3 else ""
                val = f"{right} {attr}".strip()
                out_file.write(f"{left} -> {val}\n")

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
    """Generic processing for modes that get raw string items from one or more files."""
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
                # Value is after ' = "' and ends with a quote. We get everything up to the last quote.
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


def _extract_quoted_items(input_file: str, quiet: bool = False) -> Iterable[str]:
    """Yield text found between double or single quotes."""
    # Matches "..." or '...' and handles backslash escaping
    pattern = re.compile(r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'')

    lines = _read_file_lines_robust(input_file)
    for line in tqdm(lines, desc=f'Processing {input_file} (quoted)', unit=' lines', disable=quiet):
        for match in pattern.finditer(line):
            content = match.group(1) if match.group(1) is not None else match.group(2)
            yield content


def _extract_between_items(
    input_file: str, start: str, end: str, multi_line: bool = False, quiet: bool = False
) -> Iterable[str]:
    """Yield text found between start and end markers."""
    flags = re.DOTALL if multi_line else 0
    # Escape markers for safe regex usage
    pattern = re.compile(re.escape(start) + r'(.*?)' + re.escape(end), flags)

    if multi_line:
        lines = _read_file_lines_robust(input_file)
        content = "".join(lines)
        for match in pattern.finditer(content):
            yield match.group(1)
    else:
        lines = _read_file_lines_robust(input_file)
        for line in tqdm(lines, desc=f'Processing {input_file} (between)', unit=' lines', disable=quiet):
            for match in pattern.finditer(line):
                yield match.group(1)


def _traverse_data(data: Any, path_parts: List[str]) -> Iterable[str]:
    """Recursively traverse a nested data structure (list/dict) to get values."""
    # If it's a list, apply the current path traversal to every item
    if isinstance(data, list):
        for item in data:
            yield from _traverse_data(item, path_parts)
        return

    # If we are at the end of the path, yield the string representation of the data
    if not path_parts:
        if isinstance(data, dict):
            # For a top level dictionary, yield the keys (common for typo mappings)
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


def _extract_toml_items(input_file: str, key_path: str, quiet: bool = False) -> Iterable[str]:
    """Yield values from TOML objects based on a dotted key path."""

    if not _TOMLLIB_AVAILABLE and not _TOML_AVAILABLE:
        logging.error("TOML support requires Python 3.11+ or the 'toml' package.")
        sys.exit(1)

    path_parts = key_path.split('.') if key_path else []

    lines = _read_file_lines_robust(input_file)
    content = "".join(lines)
    try:
        if not content.strip():
            return
        if _TOMLLIB_AVAILABLE:
            data = tomllib.loads(content)
        else:
            # Fallback to third-party toml package
            import toml
            data = toml.loads(content)
        yield from _traverse_data(data, path_parts)
    except Exception as e:
        logging.error(f"Failed to parse TOML in '{input_file}': {e}")
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


def _extract_char_items(input_file: str, quiet: bool = False) -> Iterable[str]:
    """Yield each character from the file (excluding newlines)."""
    lines = _read_file_lines_robust(input_file)
    for line in tqdm(lines, desc=f'Processing {input_file} (chars)', unit=' lines', disable=quiet):
        for char in line:
            if char not in ('\n', '\r'):
                yield char


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
            # This allows getting from pairs like "- typo: correction"
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
    """Wrapper for getting N-grams from file(s)."""
    def extractor(f, quiet=False):
        return _extract_ngram_items(
            f, n=n, delimiter=delimiter, quiet=quiet, smart=smart, clean_items=clean_items
        )
    # Pass clean_items=False to _process_items to preserve spaces in sequences of words.
    _process_items(
        extractor,
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        'Ngrams',
        f'Successfully got {n}-word sequences.',
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


def toml_mode(
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
    """Wrapper for getting fields from TOML files."""
    def extractor(f, quiet=False):
        return _extract_toml_items(f, key, quiet=quiet)
    _process_items(
        extractor,
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        'TOML',
        'Successfully got TOML values.',
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
        'Successfully got table fields.',
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
        'Successfully got Markdown list items.',
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
        'Successfully got Markdown table items.',
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
    """Wrapper for getting text between backticks."""
    _process_items(
        _extract_backtick_items,
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        'Backtick',
        'Successfully got strings.',
        output_format,
        quiet,
        clean_items=clean_items,
        limit=limit,
    )


def quoted_mode(
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
    """Wrapper for getting text between quotes."""
    _process_items(
        _extract_quoted_items,
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        'Quoted',
        'Successfully got quoted strings.',
        output_format,
        quiet,
        clean_items=clean_items,
        limit=limit,
    )


def between_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    start: str,
    end: str,
    multi_line: bool = False,
    output_format: str = 'line',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """Wrapper for getting text between markers."""
    def extractor(f, quiet=False):
        return _extract_between_items(f, start, end, multi_line=multi_line, quiet=quiet)
    _process_items(
        extractor,
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        'Between',
        'Successfully got strings.',
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
    """Wrapper for getting fields from JSON files."""
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
        'Successfully got JSON values.',
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
    """Wrapper for getting fields from YAML files."""
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
        'Successfully got YAML values.',
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
    lines: bool = False,
    chars: bool = False,
    mapping_file: str | None = None,
    ad_hoc: List[str] | None = None,
    by_file: bool = False,
) -> None:
    """
    Counts the frequency of each word, pair, line, or character in the input file(s)
    and writes the sorted results to the output file. Only items with length between
    min_length and max_length are counted.
    The stats are based on the raw count of items versus the filtered items.
    Note: process_output is ignored in count mode.
    """
    raw_count = 0
    filtered_items = []
    item_counts = Counter()

    start_time = time.perf_counter()

    # Mapping-based auditing
    mapping = None
    if mapping_file or ad_hoc:
        mapping = _resolve_full_mapping(mapping_file, ad_hoc, clean_items, quiet=quiet)
        pairs = True  # Automatically enable pairs for mapping audit

    if mapping:
        # Audit mode: Count matches of mapped typos in input files
        for input_file in input_files:
            if by_file:
                file_items = set()
            words_gen = _extract_words_items(input_file, delimiter=delimiter, quiet=quiet, smart=smart)
            for word in words_gen:
                raw_count += 1
                match_key = filter_to_letters(word) if clean_items else word
                if match_key in mapping:
                    correction = mapping[match_key]
                    if min_length <= len(match_key) <= max_length:
                        if by_file:
                            file_items.add((match_key, correction))
                        else:
                            filtered_items.append((match_key, correction))
                            item_counts.update([(match_key, correction)])
            if by_file:
                filtered_items.extend(list(file_items))
                item_counts.update(file_items)
    elif pairs:
        # Mode for counting typo -> correction pairs from existing mapping files
        for input_file in input_files:
            if by_file:
                file_items = set()
            for left, right in _extract_pairs([input_file], quiet=quiet):
                raw_count += 1
                if clean_items:
                    left = filter_to_letters(left)
                    right = filter_to_letters(right)
                if not left or not right:
                    continue
                if min_length <= len(left) <= max_length and min_length <= len(right) <= max_length:
                    if by_file:
                        file_items.add((left, right))
                    else:
                        filtered_items.append((left, right))
                        item_counts.update([(left, right)])
            if by_file:
                filtered_items.extend(list(file_items))
                item_counts.update(file_items)
    else:
        # Default mode for counting individual words, lines, or characters
        for input_file in input_files:
            if by_file:
                file_items = set()

            if lines:
                items_gen = _extract_line_items(input_file, quiet=quiet)
            elif chars:
                items_gen = _extract_char_items(input_file, quiet=quiet)
            else:
                items_gen = _extract_words_items(input_file, delimiter=delimiter, quiet=quiet, smart=smart)

            for item in items_gen:
                raw_count += 1
                # Filter and clean the item
                filtered = clean_and_filter([item], min_length, max_length, clean=clean_items)
                if filtered:
                    if by_file:
                        file_items.update(filtered)
                    else:
                        filtered_items.extend(filtered)
                        item_counts.update(filtered)
            if by_file:
                filtered_items.extend(list(file_items))
                item_counts.update(file_items)

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
            count_label = "files" if by_file else "count"
            if pairs:
                writer.writerow(["typo", "correction", count_label])
                for item, count in final_results:
                    writer.writerow([item[0], item[1], count])
            else:
                if by_file:
                    writer.writerow(["item", count_label])
                for item, count in final_results:
                    writer.writerow([item, count])
        elif output_format == 'markdown':
            for item, count in final_results:
                label = f"{item[0]} -> {item[1]}" if pairs else item
                out_file.write(f"- {label}: {count}\n")
        elif output_format == 'md-table':
            count_label = "Files" if by_file else "Count"
            if pairs:
                out_file.write(f"| Typo | Correction | {count_label} |\n")
                out_file.write("| :--- | :--- | :--- |\n")
                for item, count in final_results:
                    out_file.write(f"| {item[0]} | {item[1]} | {count} |\n")
            else:
                out_file.write(f"| Item | {count_label} |\n")
                out_file.write("| :--- | :--- |\n")
                for item, count in final_results:
                    out_file.write(f"| {item} | {count} |\n")
        elif output_format == 'arrow':
            # Rich visual report for arrow format
            total_count = sum(item_counts.values())
            # For file-based counting, percentages are relative to total files
            total_for_pct = len(input_files) if by_file else total_count

            # Find max width for common columns
            max_count_len = max((len(str(count)) for item, count in final_results), default=5)
            count_header_label = "Files" if by_file else "Count"
            max_count_len = max(max_count_len, len(count_header_label))
            max_pct = 6  # "100.0%"
            max_bar = 20

            # Colors for output
            # We determine color availability independently for stdout and stderr to preserve correct behavior when piping.
            use_color_out = _should_enable_color(out_file)
            use_color_err = _should_enable_color(sys.stderr)

            c_out_bold = BOLD if use_color_out else ""
            c_out_blue = BLUE if use_color_out else ""
            c_out_green = GREEN if use_color_out else ""
            c_out_red = RED if use_color_out else ""
            c_out_yellow = YELLOW if use_color_out else ""
            c_out_reset = RESET if use_color_out else ""

            # Header and divider elements
            padding = "  "
            # Bold blue for table visual elements
            sep = f"{c_out_bold}{c_out_blue}│{c_out_reset}"

            if pairs:
                item_header_left = "Typo"
                item_header_right = "Correction"
                max_left = max((len(str(item[0])) for item, _ in final_results), default=len(item_header_left))
                max_left = max(max_left, len(item_header_left))
                max_right = max((len(str(item[1])) for item, _ in final_results), default=len(item_header_right))
                max_right = max(max_right, len(item_header_right))

                header = (
                    f"{padding}{c_out_bold}{c_out_blue}{item_header_left:<{max_left}}{c_out_reset} {sep} "
                    f"{c_out_bold}{c_out_blue}{item_header_right:<{max_right}}{c_out_reset} {sep} "
                    f"{c_out_bold}{c_out_blue}{count_header_label:>{max_count_len}}{c_out_reset} {sep} "
                    f"{c_out_bold}{c_out_blue}{'%':>{max_pct}}{c_out_reset} {sep} "
                    f"{c_out_bold}{c_out_blue}{'Visual':<{max_bar}}{c_out_reset}"
                )
                # 3 chars for each " │ " (total 4 * 3 = 12)
                visible_header_len = max_left + max_right + max_count_len + max_pct + max_bar + 12
            else:
                if lines:
                    item_header = "Line"
                elif chars:
                    item_header = "Character"
                else:
                    item_header = "Word"

                max_item = max((len(str(item)) for item, _ in final_results), default=len(item_header))
                max_item = max(max_item, len(item_header))

                header = (
                    f"{padding}{c_out_bold}{c_out_blue}{item_header:<{max_item}}{c_out_reset} {sep} "
                    f"{c_out_bold}{c_out_blue}{count_header_label:>{max_count_len}}{c_out_reset} {sep} "
                    f"{c_out_bold}{c_out_blue}{'%':>{max_pct}}{c_out_reset} {sep} "
                    f"{c_out_bold}{c_out_blue}{'Visual':<{max_bar}}{c_out_reset}"
                )
                visible_header_len = max_item + max_count_len + max_pct + max_bar + 9

            divider = f"{padding}{c_out_bold}{c_out_blue}{'─' * visible_header_len}{c_out_reset}"
            header_block = f"\n{header}\n{divider}\n"

            # If not quiet OR writing to a file, prepare and write the summary and header block.
            if not quiet or output_file != '-':
                if pairs:
                    item_label = "pair"
                elif lines:
                    item_label = "line"
                elif chars:
                    item_label = "character"
                else:
                    item_label = "word"

                extra_metrics = {}
                if by_file:
                    extra_metrics["Total files processed"] = len(input_files)

                summary_lines = _format_analysis_summary(
                    raw_count, filtered_items, item_label, start_time, use_color_err,
                    extra_metrics=extra_metrics
                )
                summary_text = "\n".join(summary_lines) + "\n"

                # When the output is the console ('-'), we write the analysis summary and table
                # header to stderr. This keeps stdout clean for piped data.
                if output_file == '-':
                    sys.stderr.write(summary_text)
                    sys.stderr.write(header_block)
                    sys.stderr.flush()
                else:
                    # When writing to a file, we include everything in the file.
                    out_file.write(summary_text)
                    out_file.write(header_block)

            for item, count in final_results:
                percent = (count / total_for_pct * 100) if total_for_pct > 0 else 0

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

                if pairs:
                    row = (
                        f"{padding}{c_out_red}{item[0]:<{max_left}}{c_out_reset} {sep} "
                        f"{c_out_green}{item[1]:<{max_right}}{c_out_reset} {sep} "
                        f"{c_out_yellow}{count:>{max_count_len}}{c_out_reset} {sep} "
                        f"{c_out_green}{percent:>5.1f}%{c_out_reset} {sep} "
                        f"{c_out_blue}{bar}{c_out_reset}"
                    )
                else:
                    row = (
                        f"{padding}{c_out_green}{str(item):<{max_item}}{c_out_reset} {sep} "
                        f"{c_out_yellow}{count:>{max_count_len}}{c_out_reset} {sep} "
                        f"{c_out_green}{percent:>5.1f}%{c_out_reset} {sep} "
                        f"{c_out_blue}{bar}{c_out_reset}"
                    )
                out_file.write(f"{row}\n")
            out_file.write("\n")
        else:  # 'line' or fallback
            for item, count in final_results:
                label = f"{item[0]} -> {item[1]}" if pairs else item
                out_file.write(f"{label}: {count}\n")

    if output_format != 'arrow':
        if pairs:
            item_label = "pair"
        elif lines:
            item_label = "line"
        elif chars:
            item_label = "character"
        else:
            item_label = "word"

        print_processing_stats(
            raw_count,
            filtered_items,
            item_label=item_label,
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
    Groups typo corrections based on their error type.
    """
    start_time = time.perf_counter()
    raw_pairs = _extract_pairs(input_files, quiet=quiet)
    adj_keys = get_adjacent_keys()

    results = []
    stats_items = []
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
                label = f"{label} [D:{dist}]"
            results.append((left, right, label))
            stats_items.append((left, right))

    if process_output:
        results = sorted(set(results))
        stats_items = results

    _write_paired_output(
        results,
        output_file,
        output_format,
        "Classify",
        quiet,
        limit=limit
    )

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
    filtered_items = []

    for input_file in input_files:
        raw, cleaned, _ = _load_and_clean_file(
            input_file,
            min_length,
            max_length,
            split_whitespace=True,
            clean_items=clean_items,
        )
        raw_item_count += len(raw)
        filtered_items.extend(cleaned)

    unique_items = list(dict.fromkeys(filtered_items))
    unique_count = len(unique_items)

    stats = {
        "items": {
            "total_analyzed": raw_item_count,
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
            f.write(f"| Total items analyzed | {stats['items']['total_analyzed']} |\n")
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
            use_color = _should_enable_color(f)

            # In stats_mode, filtered_items is the primary list of items collected
            report = _format_analysis_summary(
                stats['items']['total_analyzed'],
                filtered_items,
                "item",
                start_time,
                use_color
            )
            f.write("\n".join(report))

            if "pairs" in stats:
                pair_metrics = {
                    "Unique typos / corrections": f"{stats['pairs']['unique_typos']} / {stats['pairs']['unique_corrections']}",
                    "Conflicts (1 typo -> N corr)": stats['pairs']['conflicts'],
                    "Overlaps (typo == correction)": stats['pairs']['overlaps']
                }
                # filtered_pairs is filtered by length and cleaning
                pair_report = _format_analysis_summary(
                    stats['pairs']['total_extracted'],
                    filtered_pairs,
                    "pair",
                    None,
                    use_color,
                    pair_metrics,
                    title="PAIRED DATA STATISTICS"
                )
                f.write("\n".join(pair_report))

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
    Finds typos that are associated with more than one unique correction.
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
    Finds repeated loops in typo-correction pairs.
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
            # Re-initialize path tracking for each component search
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
    logging.info(f"[Cycles Mode] Found {len(cycles)} repeated loops. Output written to '{output_file}'.")


def similarity_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    min_dist: int = 0,
    max_dist: int | None = None,
    show_dist: bool = False,
    keyboard: bool = False,
    transposition: bool = False,
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
    adj_keys = get_adjacent_keys()

    # Adjust max_dist if filters are used but default (None) or restrictive dist was kept
    if transposition and max_dist == 1:
        max_dist = 2
    elif keyboard and max_dist == 0:
        max_dist = 1

    filtered_results = []
    stats_items = []
    raw_count = 0
    for left, right in raw_pairs:
        raw_count += 1
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

        label = classify_typo(left, right, adj_keys)
        if keyboard or transposition:
            matches_filter = False
            if keyboard and label == "[K]":
                matches_filter = True
            if transposition and label == "[T]":
                matches_filter = True
            if not matches_filter:
                continue

        attr = label
        if show_dist:
            attr = f"{label} [D:{dist}]"
        
        filtered_results.append((left, right, attr))
        stats_items.append((left, right))

    if process_output:
        filtered_results = sorted(set(filtered_results))
        stats_items = filtered_results

    _write_paired_output(
        filtered_results,
        output_file,
        output_format,
        "Similarity",
        quiet,
        limit=limit
    )

    print_processing_stats(
        raw_count, stats_items, item_label="similar-pair", start_time=start_time
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
    keyboard: bool = False,
    transposition: bool = False,
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
    adj_keys = get_adjacent_keys()

    # Adjust max_dist if transposition filter is used but default dist was kept
    if transposition and max_dist == 1:
        max_dist = 2

    results = []
    stats_items = []
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
                label = classify_typo(word_i, word_j, adj_keys)
                if keyboard or transposition:
                    matches_filter = False
                    if keyboard and label == "[K]":
                        matches_filter = True
                    if transposition and label == "[T]":
                        matches_filter = True
                    if not matches_filter:
                        continue

                attr = label
                if show_dist:
                    attr = f"{label} [D:{dist}]"
                results.append((word_i, word_j, attr))
                stats_items.append((word_i, word_j))

    if process_output:
        results = sorted(set(results))
        stats_items = results

    _write_paired_output(
        results,
        output_file,
        output_format,
        "NearDuplicates",
        quiet,
        limit=limit
    )

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
    keyboard: bool = False,
    transposition: bool = False,
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
    adj_keys = get_adjacent_keys()

    # Adjust max_dist if transposition filter is used but default dist was kept
    if transposition and max_dist == 1:
        max_dist = 2

    results = []
    stats_items = []

    for word_i in tqdm(list1_unique, desc="Finding similar words", disable=quiet):
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
                label = classify_typo(word_i, word_j, adj_keys)
                if keyboard or transposition:
                    matches_filter = False
                    if keyboard and label == "[K]":
                        matches_filter = True
                    if transposition and label == "[T]":
                        matches_filter = True
                    if not matches_filter:
                        continue

                attr = label
                if show_dist:
                    attr = f"{label} [D:{dist}]"
                results.append((word_i, word_j, attr))
                stats_items.append((word_i, word_j))

    if process_output:
        results = sorted(set(results))
        stats_items = results

    _write_paired_output(
        results,
        output_file,
        output_format,
        "FuzzyMatch",
        quiet,
        limit=limit
    )

    print_processing_stats(
        raw_item_count, stats_items, item_label="similar-word-match", start_time=start_time
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
    Finds words that appear with inconsistent capitalization.
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
    Finds added, removed, and changed items between two files or lists.
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
            use_color = _should_enable_color(out)
            c_red = RED if use_color else ""
            c_green = GREEN if use_color else ""
            c_yellow = YELLOW if use_color else ""
            c_reset = RESET if use_color else ""

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
    Finds consecutive identical words.
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
    keyboard: bool = False,
    transposition: bool = False,
    output_format: str = 'arrow',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
    delimiter: str | None = None,
    smart: bool = False,
) -> None:
    """
    Finds potential typos by comparing rare words to frequent words.
    """
    start_time = time.perf_counter()
    word_counts = Counter()
    raw_item_count = 0

    for input_file in input_files:
        # Use the shared getting logic to support custom delimiters and smart splitting
        words_gen = _extract_words_items(input_file, delimiter=delimiter, quiet=quiet, smart=smart)
        for word in words_gen:
            raw_item_count += 1
            # Filter and clean the word
            filtered = clean_and_filter([word], min_length, max_length, clean=clean_items)
            if filtered:
                word_counts.update(filtered)

    # Find rare and frequent words
    rare_words = sorted([word for word, count in word_counts.items() if count <= rare_max])
    frequent_words = sorted([word for word, count in word_counts.items() if count >= freq_min], key=len)
    adj_keys = get_adjacent_keys()

    # Adjust max_dist if transposition filter is used but default dist was kept
    if transposition and max_dist == 1:
        max_dist = 2

    results = []
    stats_items = []
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
                label = classify_typo(rare, freq, adj_keys)
                if keyboard or transposition:
                    matches_filter = False
                    if keyboard and label == "[K]":
                        matches_filter = True
                    if transposition and label == "[T]":
                        matches_filter = True
                    if not matches_filter:
                        continue

                if show_dist:
                    attr = f"{label} [D:{dist}]"
                    results.append((rare, freq, attr))
                else:
                    results.append((rare, freq, label))
                stats_items.append((rare, freq))

    if process_output:
        results = sorted(set(results))
        stats_items = results

    _write_paired_output(
        results,
        output_file,
        output_format,
        "Discovery",
        quiet,
        limit=limit
    )

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
    """Wrapper for getting fields from CSV files."""
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
        'Successfully got CSV fields.',
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
    """Wrapper for getting individual words from file(s)."""
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
        'Successfully got words.',
        output_format,
        quiet,
        clean_items=clean_items,
        limit=limit,
    )


def _format_search_line(
    filename: str,
    line_idx: int,
    line_content: str,
    is_match: bool,
    show_filename: bool,
    line_numbers: bool,
    use_color: bool,
) -> str:
    """Formats a single line for search/scan output with optional filename and line number."""
    sep_char = ":" if is_match else "-"

    if not show_filename and not line_numbers:
        return line_content

    if use_color:
        style = BOLD if is_match else ""
        c_sep = style + CYAN
        parts = []
        if show_filename:
            parts.append(f"{style}{MAGENTA}{filename}{RESET}")
        if line_numbers:
            parts.append(f"{style}{GREEN}{line_idx + 1}{RESET}")

        if not parts:
            return line_content

        prefix = f"{c_sep}{sep_char}{RESET}".join(parts) + f"{c_sep}{sep_char}{RESET}"
        return f"{prefix} {line_content}"
    else:
        prefix_parts = []
        if show_filename:
            prefix_parts.append(filename)
        if line_numbers:
            prefix_parts.append(str(line_idx + 1))
        raw_prefix = sep_char.join(prefix_parts) + sep_char
        return f"{raw_prefix} {line_content}"


def _render_context_to_lines(
    match_indices: Mapping[int, str],
    file_contents: Sequence[str],
    before_context: int,
    after_context: int,
    filename: str,
    show_filename: bool,
    line_numbers: bool,
    use_color: bool,
) -> List[str]:
    """Renders match lines and their surrounding context lines into a list of strings."""
    accumulated_lines = []
    sorted_indices = sorted(match_indices.keys())
    last_rendered_idx = -1

    for idx in sorted_indices:
        # Determine block start and end
        start = max(0, idx - before_context)
        end = min(len(file_contents), idx + after_context + 1)

        # If there's a gap between blocks, add separator
        if last_rendered_idx != -1 and start > last_rendered_idx:
            separator = f"{BOLD}{BLUE}--{RESET}" if use_color else "--"
            accumulated_lines.append(separator)

        # Render lines in the window that haven't been rendered yet
        current_start = max(start, last_rendered_idx)
        for j in range(current_start, end):
            is_match = j in match_indices
            content = match_indices[j] if is_match else file_contents[j]
            accumulated_lines.append(
                _format_search_line(
                    filename, j, content, is_match, show_filename, line_numbers, use_color
                )
            )

        last_rendered_idx = end
    return accumulated_lines


def search_mode(
    input_files: Sequence[str],
    query: str,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    max_dist: int = 0,
    keyboard: bool = False,
    transposition: bool = False,
    smart: bool = False,
    line_numbers: bool = False,
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
    with_filename: bool | None = None,
    before_context: int = 0,
    after_context: int = 0,
) -> None:
    """
    Searches for words or patterns in text files, supporting similar word matching and smart subword detection.
    """
    start_time = time.perf_counter()
    total_matches = 0
    matched_files_count = 0
    query_clean = filter_to_letters(query) if clean_items else query.lower()
    adj_keys = get_adjacent_keys() if (keyboard or transposition) else {}

    # If transposition is requested but max_dist is at its default (0 for search), bump it to 2
    if transposition and max_dist == 0:
        max_dist = 2
    # If keyboard slip is requested but max_dist is 0, bump it to 1
    elif keyboard and max_dist == 0:
        max_dist = 1

    # Safety check for match-all queries
    if clean_items and not query_clean:
        logging.warning(
            f"Search query \"{query}\" contains no alphanumeric characters and \"clean_items\" is enabled. "
            "Skipping search to avoid matching every line."
        )
        return

    # Split query for smart matching if requested
    query_parts = _smart_split(query) if smart else [query]
    query_parts_clean = (
        [filter_to_letters(p) for p in query_parts]
        if clean_items
        else [p.lower() for p in query_parts]
    )

    accumulated_lines = []
    # Check if color is enabled for the output
    use_color = _should_enable_color(sys.stdout) if output_file == '-' else ('FORCE_COLOR' in os.environ and 'NO_COLOR' not in os.environ)

    # Determine if filenames should be shown
    show_filename = with_filename
    if show_filename is None:
        show_filename = len(input_files) > 1

    # Pre-compile patterns outside the loop for better performance
    lit_pattern = re.compile(re.escape(query), re.IGNORECASE)
    word_pattern = re.compile(r"([a-zA-Z0-9]+)")
    query_len = len(query_clean) if clean_items else len(query)
    apply_literal_match = min_length <= query_len <= max_length

    # Pre-calculate total lines for a single cohesive progress bar
    pbar = None
    if not quiet:
        total_lines = _get_total_line_count(input_files)
        pbar = tqdm(total=total_lines, desc="Searching", unit=" lines", disable=quiet)

    for input_file in input_files:
        file_lines = _read_file_lines_robust(input_file)
        # Store original lines without trailing newlines for consistent rendering
        file_contents = [line.rstrip("\n") for line in file_lines]
        match_indices = {}  # index -> highlighted_line

        if pbar:
            pbar.set_postfix(file=os.path.basename(input_file), refresh=True)

        for i, line_content in enumerate(file_contents):
            if pbar:
                pbar.update(1)
            spans = []

            # 1. Exact match on whole line first (case-insensitive)
            if apply_literal_match:
                for m in lit_pattern.finditer(line_content):
                    spans.append(m.span())

            # 2. Word-by-word logic for filtering/similar word/smart matching.
            for m_word in word_pattern.finditer(line_content):
                word = m_word.group(0)
                word_start, word_end = m_word.span()

                word_clean = filter_to_letters(word) if clean_items else word.lower()
                if not word_clean:
                    continue

                if not (min_length <= len(word_clean) <= max_length):
                    continue

                match_found_in_word = False

                if query_clean in word_clean:
                    for m_lit in lit_pattern.finditer(word):
                        spans.append((word_start + m_lit.start(), word_start + m_lit.end()))
                        match_found_in_word = True

                    if not match_found_in_word:
                        spans.append((word_start, word_end))
                        match_found_in_word = True

                elif (
                    max_dist > 0
                    and levenshtein_distance(word_clean, query_clean) <= max_dist
                ):
                    if keyboard or transposition:
                        label = classify_typo(word_clean, query_clean, adj_keys)
                        matches_filter = False
                        if keyboard and label == "[K]":
                            matches_filter = True
                        if transposition and label == "[T]":
                            matches_filter = True
                        if not matches_filter:
                            continue

                    spans.append((word_start, word_end))
                    match_found_in_word = True

                elif smart:
                    sub_parts = _smart_split(word)
                    for sp in sub_parts:
                        sp_clean = filter_to_letters(sp) if clean_items else sp.lower()
                        if not sp_clean:
                            continue

                        for qp_clean in query_parts_clean:
                            if levenshtein_distance(sp_clean, qp_clean) <= max_dist:
                                if keyboard or transposition:
                                    label = classify_typo(sp_clean, qp_clean, adj_keys)
                                    matches_filter = False
                                    if keyboard and label == "[K]":
                                        matches_filter = True
                                    if transposition and label == "[T]":
                                        matches_filter = True
                                    if not matches_filter:
                                        continue

                                spans.append((word_start, word_end))
                                match_found_in_word = True
                                break
                        if match_found_in_word:
                            break

            if spans:
                if use_color:
                    spans.sort()
                    merged = []
                    curr_start, curr_end = spans[0]
                    for next_start, next_end in spans[1:]:
                        if next_start <= curr_end:
                            curr_end = max(curr_end, next_end)
                        else:
                            merged.append((curr_start, curr_end))
                            curr_start, curr_end = next_start, next_end
                    merged.append((curr_start, curr_end))

                    last_idx = 0
                    highlighted_line = ""
                    for start, end in merged:
                        highlighted_line += line_content[last_idx:start]
                        highlighted_line += f"{BOLD}{YELLOW}{line_content[start:end]}{RESET}"
                        last_idx = end
                    highlighted_line += line_content[last_idx:]
                    match_indices[i] = highlighted_line
                else:
                    match_indices[i] = line_content

        if not match_indices:
            continue

        total_matches += len(match_indices)
        matched_files_count += 1

        # Collect context blocks
        accumulated_lines.extend(
            _render_context_to_lines(
                match_indices,
                file_contents,
                before_context,
                after_context,
                input_file,
                show_filename,
                line_numbers,
                use_color,
            )
        )

    if process_output:
        accumulated_lines = sorted(set(accumulated_lines))

    if pbar:
        pbar.close()

    if limit is not None:
        accumulated_lines = accumulated_lines[:limit]

    with smart_open_output(output_file) as out:
        for line in accumulated_lines:
            out.write(line + "\n")

    duration = time.perf_counter() - start_time
    # Use color for feedback if stderr is a terminal
    c_blue = (BOLD + BLUE) if _should_enable_color(sys.stderr) else ""
    c_green = GREEN if _should_enable_color(sys.stderr) else ""
    c_reset = RESET if _should_enable_color(sys.stderr) else ""

    logging.info(
        f"{c_blue}[Search Mode]{c_reset} Completed search in {len(input_files)} file(s) for \"{query}\". "
        f"Found {c_green}{total_matches}{c_reset} match(es) in {c_green}{matched_files_count}{c_reset} of {len(input_files)} files. "
        f"Output written to \"{output_file}\". Processing time: {duration:.3f}s"
    )


def _collect_unique_items(
    input_files: Sequence[str],
    min_length: int,
    max_length: int,
    clean_items: bool,
) -> Tuple[int, List[str]]:
    """Helper to collect and deduplicate items from multiple files while preserving order."""
    raw_item_count = 0
    combined_items = []
    for file_path in input_files:
        raw, _, unique = _load_and_clean_file(
            file_path,
            min_length,
            max_length,
            clean_items=clean_items,
        )
        raw_item_count += len(raw)
        combined_items.extend(unique)

    # Deduplicate while preserving order of first appearance
    return raw_item_count, list(dict.fromkeys(combined_items))


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
    raw_count, unique_items = _collect_unique_items(
        input_files, min_length, max_length, clean_items
    )

    # Combine mode always sorts results alphabetically
    final_items = sorted(unique_items)

    write_output(final_items, output_file, output_format, quiet, limit=limit)
    print_processing_stats(raw_count, final_items, start_time=start_time)
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
    raw_count, final_items = _collect_unique_items(
        input_files, min_length, max_length, clean_items
    )

    if process_output:
        # If the user explicitly requested -P, we sort alphabetically.
        # But by default unique mode is order-preserving.
        final_items.sort()

    write_output(final_items, output_file, output_format, quiet, limit=limit)
    print_processing_stats(raw_count, final_items, start_time=start_time)
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


def _process_pairs(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    mode_label: str,
    item_label: str,
    output_format: str,
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
    transform: Callable[[str, str], Tuple[str, str]] | None = None,
    separator: str = " -> ",
) -> None:
    """Generic processing for modes that handle paired data (typo -> correction)."""
    start_time = time.perf_counter()

    raw_pairs = _extract_pairs(input_files, quiet=quiet)

    filtered_pairs = []
    raw_count = 0
    for left, right in raw_pairs:
        raw_count += 1

        if transform:
            left, right = transform(left, right)

        if clean_items:
            left = filter_to_letters(left)
            right = filter_to_letters(right)

        if not left or not right:
            continue

        if min_length <= len(left) <= max_length and min_length <= len(right) <= max_length:
            filtered_pairs.append((left, right))

    if process_output:
        filtered_pairs = sorted(set(filtered_pairs))

    _write_paired_output(
        filtered_pairs,
        output_file,
        output_format,
        mode_label,
        quiet,
        limit=limit,
        separator=separator
    )

    print_processing_stats(
        raw_count, filtered_pairs, item_label=item_label, start_time=start_time
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
    _process_pairs(
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        "Pairs",
        "pair",
        output_format,
        quiet,
        clean_items,
        limit
    )


def align_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    separator: str = " -> ",
    output_format: str = 'aligned',
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
) -> None:
    """Extracts pairs from any supported format and outputs them in aligned columns."""
    _process_pairs(
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        "Align",
        "pair",
        output_format,
        quiet,
        clean_items,
        limit,
        separator=separator
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
    _process_pairs(
        input_files,
        output_file,
        min_length,
        max_length,
        process_output,
        "Swap",
        "swapped-pair",
        output_format,
        quiet,
        clean_items,
        limit,
        transform=lambda left, right: (right, left)
    )


def resolve_mode(
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
    """
    Finds and shortens chains of typo corrections. For example, if a list
    contains A -> B and B -> C, this mode will update it to A -> C and B -> C.
    """
    start_time = time.perf_counter()
    raw_pairs = list(_extract_pairs(input_files, quiet=quiet))
    mapping = {}

    for left, right in raw_pairs:
        if clean_items:
            left = filter_to_letters(left)
            right = filter_to_letters(right)

        if not left or not right:
            continue

        if min_length <= len(left) <= max_length and min_length <= len(right) <= max_length:
            # We take the first mapping analyzed for each typo if there are conflicts.
            # Using dict.setdefault ensures the first entry wins.
            mapping.setdefault(left, right)

    resolved_pairs = []

    for typo in sorted(mapping.keys()):
        visited = {typo}
        curr = mapping[typo]

        # Follow the chain until it reaches a word not in the mapping (a terminal value)
        # or we hit a cycle.
        while curr in mapping and mapping[curr] not in visited:
            visited.add(curr)
            curr = mapping[curr]

        resolved_pairs.append((typo, curr))

    if process_output:
        resolved_pairs = sorted(set(resolved_pairs))

    _write_paired_output(
        resolved_pairs,
        output_file,
        output_format,
        "Resolve",
        quiet,
        limit=limit
    )

    print_processing_stats(
        len(raw_pairs), resolved_pairs, item_label="resolved-pair", start_time=start_time
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
    """Wrapper for getting text matching a regex pattern."""
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
        'Successfully got regex matches.',
        output_format,
        quiet,
        clean_items=False,
        limit=limit,
    )


def _resolve_full_mapping(
    mapping_file: str | None,
    ad_hoc_pairs: List[str] | None,
    clean_items: bool,
    quiet: bool = False,
) -> dict[str, str]:
    """
    Merge a mapping file with extra pairs from the command line.
    """
    full_mapping = {}

    # 1. Load from file if provided
    if mapping_file:
        # Load mapping or list
        if mapping_file.lower().endswith(('.json', '.csv', '.yaml', '.yml', '.toml')):
            full_mapping.update(dict(_extract_pairs([mapping_file], quiet=quiet)))
        else:
            # Treat as a simple list of words if not a common mapping format
            lines = _read_file_lines_robust(mapping_file)
            for line in lines:
                content = line.strip()
                if not content or content.startswith('#'):
                    continue

                if " -> " in content:
                    parts = content.split(" -> ", 1)
                    full_mapping[parts[0].strip()] = parts[1].strip()
                elif ": " in content:
                    parts = content.split(": ", 1)
                    full_mapping[parts[0].strip()] = parts[1].strip()
                else:
                    full_mapping[content] = ""

    # 2. Add extra pairs (for example, "teh:the" or "old:correction")
    if ad_hoc_pairs:
        for pair in ad_hoc_pairs:
            if ":" in pair:
                k, v = pair.split(":", 1)
                full_mapping[k.strip()] = v.strip()
            else:
                # If no colon, treat as a word to be highlighted or matched (empty value)
                full_mapping[pair.strip()] = ""

    # 3. Clean mapping keys if requested
    if clean_items:
        cleaned_mapping = {}
        for k, v in full_mapping.items():
            cleaned_k = filter_to_letters(k)
            if cleaned_k:
                cleaned_mapping[cleaned_k] = v
        return cleaned_mapping

    return full_mapping


def map_mode(
    input_files: Sequence[str],
    mapping_file: str | None,
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
    ad_hoc: List[str] | None = None,
) -> None:
    """
    Transforms items based on a mapping file or extra pairs.
    """
    start_time = time.perf_counter()
    # Load and merge mappings
    mapping = _resolve_full_mapping(mapping_file, ad_hoc, clean_items, quiet=quiet)

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
            part = line_content
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


def _scrub_line(
    line: str,
    mapping: Mapping[str, str],
    pattern: re.Pattern,
    clean_items: bool = True,
    smart_case: bool = False,
    standardize: bool = False,
) -> Tuple[str, int]:
    """
    Replaces words in a line based on a mapping.
    Returns a tuple of (modified_line, replacement_count).
    """
    def get_replacement(word: str) -> str | None:
        """Helper to get a replacement for a word from the mapping."""
        if standardize:
            key = filter_to_letters(word) if clean_items else word.lower()
        else:
            key = filter_to_letters(word) if clean_items else word

        if key in mapping:
            res = mapping[key]
            if smart_case:
                res = _apply_smart_case(word, res)
            return res
        return None

    parts = pattern.split(line)
    new_parts = []
    replacements = 0
    for part in parts:
        if not part:
            continue

        if pattern.match(part):
            # It's a word candidate.
            replacement = get_replacement(part)
            if replacement is not None:
                new_parts.append(replacement)
                if replacement != part:
                    replacements += 1
            else:
                # Try subword replacement if the whole word didn't match.
                sub_parts = _smart_split(part)
                new_sub_parts = []
                for sp in sub_parts:
                    sub_repl = get_replacement(sp)
                    if sub_repl is not None:
                        new_sub_parts.append(sub_repl)
                        if sub_repl != sp:
                            replacements += 1
                    else:
                        new_sub_parts.append(sp)
                new_parts.append("".join(new_sub_parts))
        else:
            # It's a delimiter (punctuation, whitespace)
            new_parts.append(part)

    return "".join(new_parts), replacements


def standardize_mode(
    input_files: Sequence[str],
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
    in_place: str | None = None,
    dry_run: bool = False,
    fuzzy: int = 0,
    threshold: float = 10.0,
    diff: bool = False,
) -> None:
    """
    Standardizes inconsistent casing and optionally spelling of words within files
    by replacing less frequent variations with the most frequent one found
    across all input files.
    """
    start_time = time.perf_counter()
    pattern = re.compile(r'([a-zA-Z0-9]+)')

    # Pass 1: Count variations
    variation_counts = defaultdict(Counter)

    for input_file in input_files:
        file_lines = _read_file_lines_robust(input_file)
        for line in tqdm(file_lines, desc=f"Analyzing {input_file}", unit=" lines", disable=quiet):
            parts = pattern.findall(line)
            for part in parts:
                # Full word analysis
                norm = filter_to_letters(part) if clean_items else part.lower()
                if norm:
                    if min_length <= len(norm) <= max_length:
                        variation_counts[norm][part] += 1

                # Sub-word analysis for CamelCase/snake_case consistency
                sub_parts = _smart_split(part)
                if len(sub_parts) > 1:
                    for sp in sub_parts:
                        sp_norm = filter_to_letters(sp) if clean_items else sp.lower()
                        if sp_norm:
                            if min_length <= len(sp_norm) <= max_length:
                                variation_counts[sp_norm][sp] += 1

    # Pass 2: Find "winners" and build mapping
    mapping = {}

    # 2a: Determine the best casing variation for each normalized word
    norm_winners = {}
    norm_totals = Counter()
    for norm, counts in variation_counts.items():
        norm_winners[norm] = counts.most_common(1)[0][0]
        norm_totals[norm] = sum(counts.values())

    # 2b: If similar word matching is enabled, group similar normalized words
    if fuzzy > 0:
        # Sort normalized words by total frequency descending to find "anchors"
        sorted_norms = sorted(norm_totals.keys(), key=lambda n: norm_totals[n], reverse=True)
        fuzzy_groups = {}  # rare_norm -> frequent_norm

        for i, frequent in enumerate(sorted_norms):
            f_count = norm_totals[frequent]

            for j in range(i + 1, len(sorted_norms)):
                rare = sorted_norms[j]
                if rare in fuzzy_groups:
                    continue

                r_count = norm_totals[rare]

                # Only consider if the frequent word is significantly more common
                if f_count >= r_count * threshold:
                    if levenshtein_distance(frequent, rare) <= fuzzy:
                        fuzzy_groups[rare] = frequent
                        logging.info(f"[Fuzzy] Identified likely typo: '{rare}' ({r_count}) -> '{frequent}' ({f_count})")

        # Build the final mapping using both casing and fuzzy logic
        for norm in norm_totals:
            target_norm = fuzzy_groups.get(norm, norm)
            winner = norm_winners[target_norm]

            # If norm was a rare word that was fuzzy-mapped, or it has casing variations
            if norm in fuzzy_groups or len(variation_counts[norm]) > 1:
                # Double check that we aren't mapping a word to itself identically
                # (Can happen if casing winner matches common variation of rare word by chance)
                if norm not in variation_counts[norm] or variation_counts[norm][norm] < norm_totals[norm] or norm != winner:
                    mapping[norm] = winner
    else:
        # Original logic for casing-only standardization
        for norm, counts in variation_counts.items():
            if len(counts) > 1:
                mapping[norm] = norm_winners[norm]

    if not mapping:
        logging.info("No inconsistencies found. Everything is standardized.")
        # If not in-place, we still proceed to Pass 3 to ensure output is written if requested.
        # But if in-place, we can exit early.
        if in_place is not None:
            return

    # Pass 3: Transformation
    total_replacements = 0
    accumulated_lines = []

    with (smart_open_output(output_file) if diff else contextlib.nullcontext()) as diff_out:
        for input_file in input_files:
            if input_file == '-' and in_place is not None:
                logging.warning("In-place modification requested for standard input; ignoring.")

            file_lines = _read_file_lines_robust(input_file)
            # Store original lines without newlines for diffing
            original_lines = [line.rstrip('\n') for line in file_lines]
            modified_lines = []
            file_replacements = 0

            for line in tqdm(file_lines, desc=f"Standardizing {input_file}", unit=" lines", disable=quiet):
                modified_line, replacements = _scrub_line(
                    line, mapping, pattern, clean_items, smart_case=False, standardize=True
                )
                modified_lines.append(modified_line)
                file_replacements += replacements

            total_replacements += file_replacements

            if diff and file_replacements > 0:
                _write_diff_report(input_file, original_lines, modified_lines, diff_out)

            if in_place is not None and input_file != '-':
                _write_file_in_place(
                    input_file,
                    modified_lines,
                    file_replacements,
                    in_place_ext=in_place,
                    dry_run=dry_run
                )
            else:
                accumulated_lines.extend(modified_lines)

    if in_place is None:
        if limit is not None:
            accumulated_lines = accumulated_lines[:limit]

        duration = time.perf_counter() - start_time
        if dry_run:
            logging.warning(f"[Dry Run] Total replacements that would be made: {total_replacements}. Processing time: {duration:.3f}s")
        elif not diff:
            with smart_open_output(output_file) as out:
                for line in accumulated_lines:
                    out.write(line)
                    if not line.endswith('\n'):
                        out.write('\n')

            logging.info(
                f"[Standardize Mode] Completed standardizing {len(input_files)} file(s). "
                f"Made {total_replacements} replacements. Output written to '{output_file}'. "
                f"Processing time: {duration:.3f}s"
            )
        else:
            logging.info(
                f"[Standardize Mode] Completed standardizing {len(input_files)} file(s). "
                f"Made {total_replacements} replacements. Diff report written to '{output_file}'. "
                f"Processing time: {duration:.3f}s"
            )
    elif dry_run:
        logging.warning(f"[Dry Run] Total replacements that would be made across all files: {total_replacements}")


def scrub_mode(
    input_files: Sequence[str],
    mapping_file: str | None,
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
    ad_hoc: List[str] | None = None,
    diff: bool = False,
) -> None:
    """
    Performs replacements of typos in text files based on a mapping file or extra pairs.
    Supports in-place modification and dry-run preview.
    """
    start_time = time.perf_counter()
    # Load and merge mappings
    mapping = _resolve_full_mapping(mapping_file, ad_hoc, clean_items, quiet=quiet)

    total_replacements = 0
    # Pattern for splitting lines into words and non-words (delimiters)
    # This ensures we preserve whitespace and punctuation exactly.
    pattern = re.compile(r'([a-zA-Z0-9]+)')

    # If in_place, we process each file individually.
    # Otherwise, we accumulate and write to output_file.
    accumulated_lines = []

    with (smart_open_output(output_file) if diff else contextlib.nullcontext()) as diff_out:
        for input_file in input_files:
            if input_file == '-' and in_place is not None:
                logging.warning("In-place modification requested for standard input; ignoring.")

            file_lines = _read_file_lines_robust(input_file)
            # Store original lines without newlines for diffing
            original_lines = [line.rstrip('\n') for line in file_lines]
            modified_lines = []
            file_replacements = 0

            for line in tqdm(file_lines, desc=f"Fixing typos in {input_file}", unit=" lines", disable=quiet):
                modified_line, replacements = _scrub_line(
                    line, mapping, pattern, clean_items, smart_case
                )
                modified_lines.append(modified_line)
                file_replacements += replacements

            total_replacements += file_replacements

            if diff and file_replacements > 0:
                _write_diff_report(input_file, original_lines, modified_lines, diff_out)

            if in_place is not None and input_file != '-':
                _write_file_in_place(
                    input_file,
                    modified_lines,
                    file_replacements,
                    in_place_ext=in_place,
                    dry_run=dry_run
                )
            else:
                accumulated_lines.extend(modified_lines)

    if in_place is None:
        if limit is not None:
            accumulated_lines = accumulated_lines[:limit]

        duration = time.perf_counter() - start_time
        if dry_run:
            logging.warning(f"[Dry Run] Total replacements that would be made: {total_replacements}. Processing time: {duration:.3f}s")
        elif not diff:
            with smart_open_output(output_file) as out:
                for line in accumulated_lines:
                    out.write(line)
                    if not line.endswith('\n'):
                        out.write('\n')

            logging.info(
                f"[Scrub Mode] Completed fixing typos in {len(input_files)} file(s) using '{mapping_file}'. "
                f"Made {total_replacements} replacements. Output written to '{output_file}'. "
                f"Processing time: {duration:.3f}s"
            )
        else:
            logging.info(
                f"[Scrub Mode] Completed fixing typos in {len(input_files)} file(s) using '{mapping_file}'. "
                f"Made {total_replacements} replacements. Diff report written to '{output_file}'. "
                f"Processing time: {duration:.3f}s"
            )
    elif dry_run:
        logging.warning(f"[Dry Run] Total replacements that would be made across all files: {total_replacements}")


def rename_mode(
    input_files: Sequence[str],
    mapping_file: str | None,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
    in_place: bool = False,
    dry_run: bool = False,
    smart_case: bool = False,
    ad_hoc: List[str] | None = None,
) -> None:
    """
    Renames files and directories using a mapping file or extra pairs.
    """
    start_time = time.perf_counter()
    # Load and merge mappings
    mapping = _resolve_full_mapping(mapping_file, ad_hoc, clean_items, quiet=quiet)

    total_renames = 0
    pattern = re.compile(r'([a-zA-Z0-9]+)')

    # To handle nested renames safely, we must rename from bottom to top.
    # Deduplicate and normalize paths first.
    normalized_paths = [os.path.normpath(p) for p in input_files if p != '-']
    unique_paths = list(dict.fromkeys(normalized_paths))
    # Sort by depth descending.
    sorted_paths = sorted(unique_paths, key=lambda p: (len(p.split(os.sep)), len(p)), reverse=True)

    rename_results = []

    for path in tqdm(sorted_paths, desc="Processing renames", disable=quiet):
        if not os.path.exists(path):
            continue

        dirname, basename = os.path.split(path)

        # Apply _scrub_line logic to the basename only
        new_basename, replacements = _scrub_line(
            basename, mapping, pattern, clean_items, smart_case
        )

        if replacements > 0 and new_basename != basename:
            new_path = os.path.join(dirname, new_basename)
            rename_results.append((path, new_path))
            total_renames += 1

            if in_place and not dry_run:
                try:
                    os.rename(path, new_path)
                    logging.info(f"Renamed '{path}' to '{new_path}'")
                except Exception as e:
                    logging.error(f"Failed to rename '{path}' to '{new_path}': {e}")
                    sys.exit(1)
            elif dry_run:
                logging.warning(f"[Dry Run] Would rename '{path}' to '{new_path}'")

    if in_place:
        if dry_run:
            logging.warning(f"[Dry Run] Total renames that would be made: {total_renames}")
        else:
            logging.info(f"[Rename Mode] Successfully made {total_renames} rename(s).")
    else:
        _write_paired_output(
            rename_results,
            output_file,
            'arrow',
            'Rename',
            quiet,
            limit=limit
        )

    # For stats, use the paths as the items
    stats_items = [r[1] for r in rename_results]
    print_processing_stats(
        len(unique_paths), stats_items, item_label="rename", start_time=start_time
    )


def highlight_mode(
    input_files: Sequence[str],
    mapping_file: str | None,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
    smart: bool = False,
    ad_hoc: List[str] | None = None,
) -> None:
    """
    Highlights words from a mapping file or extra pairs within the input text files.
    """
    start_time = time.perf_counter()
    # Load and merge mappings
    mapping = _resolve_full_mapping(mapping_file, ad_hoc, clean_items, quiet=quiet)

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
                        new_parts.append(f"{BOLD}{YELLOW}{part}{RESET}")
                        file_highlights += 1
                    elif smart:
                        # Try subword matching
                        sub_parts = _smart_split(part)
                        new_sub_parts = []
                        for sp in sub_parts:
                            sm_key = filter_to_letters(sp) if clean_items else sp
                            if sm_key in mapping:
                                new_sub_parts.append(f"{BOLD}{YELLOW}{sp}{RESET}")
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


def scan_mode(
    input_files: Sequence[str],
    mapping_file: str | None,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
    smart: bool = False,
    line_numbers: bool = False,
    with_filename: bool | None = None,
    ad_hoc: List[str] | None = None,
    before_context: int = 0,
    after_context: int = 0,
) -> None:
    """
    Scans files for matches of words from a mapping file or extra pairs, providing context.
    """
    start_time = time.perf_counter()
    # Load and merge mappings
    mapping = _resolve_full_mapping(mapping_file, ad_hoc, clean_items, quiet=quiet)

    total_matches = 0
    matched_files_count = 0
    pattern = re.compile(r'([a-zA-Z0-9]+)')

    accumulated_lines = []
    # Check if color is enabled for the output
    use_color = _should_enable_color(sys.stdout) if output_file == '-' else ('FORCE_COLOR' in os.environ and 'NO_COLOR' not in os.environ)

    # Determine if filenames should be shown
    show_filename = with_filename
    if show_filename is None:
        show_filename = len(input_files) > 1

    # Pre-calculate total lines for a single cohesive progress bar
    pbar = None
    if not quiet:
        total_lines = _get_total_line_count(input_files)
        pbar = tqdm(total=total_lines, desc="Scanning", unit=" lines", disable=quiet)

    for input_file in input_files:
        file_lines = _read_file_lines_robust(input_file)
        file_contents = [line.rstrip('\n') for line in file_lines]
        match_indices = {}  # index -> highlighted_line

        if pbar:
            pbar.set_postfix(file=os.path.basename(input_file), refresh=True)

        for i, line_content in enumerate(file_contents):
            if pbar:
                pbar.update(1)
            parts = pattern.split(line_content)
            match_found = False

            # First pass: check if any word in the line is in our mapping
            for part in parts:
                if not part or not pattern.match(part):
                    continue

                match_key = filter_to_letters(part) if clean_items else part
                if match_key in mapping:
                    match_found = True
                    break

                if smart:
                    sub_parts = _smart_split(part)
                    for sp in sub_parts:
                        sm_key = filter_to_letters(sp) if clean_items else sp
                        if sm_key in mapping:
                            match_found = True
                            break
                    if match_found:
                        break

            if match_found:
                # Highlight if color is enabled
                if use_color:
                    new_parts = []
                    for part in parts:
                        if not part:
                            continue
                        if pattern.match(part):
                            mk = filter_to_letters(part) if clean_items else part
                            if mk in mapping:
                                new_parts.append(f"{BOLD}{YELLOW}{part}{RESET}")
                            elif smart:
                                sub_parts = _smart_split(part)
                                sub_new_parts = []
                                for sp in sub_parts:
                                    smk = filter_to_letters(sp) if clean_items else sp
                                    if smk in mapping:
                                        sub_new_parts.append(f"{BOLD}{YELLOW}{sp}{RESET}")
                                    else:
                                        sub_new_parts.append(sp)
                                new_parts.append("".join(sub_new_parts))
                            else:
                                new_parts.append(part)
                        else:
                            new_parts.append(part)
                    match_indices[i] = "".join(new_parts)
                else:
                    match_indices[i] = line_content

        if not match_indices:
            continue

        total_matches += len(match_indices)
        matched_files_count += 1

        # Collect context blocks
        accumulated_lines.extend(
            _render_context_to_lines(
                match_indices,
                file_contents,
                before_context,
                after_context,
                input_file,
                show_filename,
                line_numbers,
                use_color,
            )
        )

    if process_output:
        accumulated_lines = sorted(set(accumulated_lines))

    if pbar:
        pbar.close()

    if limit is not None:
        accumulated_lines = accumulated_lines[:limit]

    with smart_open_output(output_file) as out:
        for line in accumulated_lines:
            out.write(line + "\n")

    duration = time.perf_counter() - start_time
    # Use color for feedback if stderr is a terminal
    c_blue = (BOLD + BLUE) if _should_enable_color(sys.stderr) else ""
    c_green = GREEN if _should_enable_color(sys.stderr) else ""
    c_reset = RESET if _should_enable_color(sys.stderr) else ""

    logging.info(
        f"{c_blue}[Scan Mode]{c_reset} Completed scanning {len(input_files)} file(s) for items in '{mapping_file}'. "
        f"Found {c_green}{total_matches}{c_reset} match(es) in {c_green}{matched_files_count}{c_reset} of {len(input_files)} files. "
        f"Output written to '{output_file}'. Processing time: {duration:.3f}s"
    )

def verify_mode(
    input_files: Sequence[str],
    mapping_file: str | None,
    output_file: str,
    min_length: int,
    max_length: int,
    process_output: bool,
    quiet: bool = False,
    clean_items: bool = True,
    limit: int | None = None,
    smart: bool = False,
    prune: bool = False,
    ad_hoc: List[str] | None = None,
) -> None:
    """
    Finds which entries in a mapping file are present in the provided input files.
    """
    start_time = time.perf_counter()
    # Load and merge mappings
    mapping = _resolve_full_mapping(mapping_file, ad_hoc, clean_items, quiet=quiet)

    if not mapping:
        logging.error("No mapping provided to verify. Use --mapping or --add.")
        sys.exit(1)

    found_keys = set()
    pattern = re.compile(r'([a-zA-Z0-9]+)')
    total_keys = len(mapping)

    for input_file in input_files:
        if len(found_keys) == total_keys:
            break

        file_lines = _read_file_lines_robust(input_file)
        for line in tqdm(file_lines, desc=f"Verifying {input_file}", unit=" lines", disable=quiet):
            parts = pattern.findall(line)
            for part in parts:
                match_key = filter_to_letters(part) if clean_items else part
                if match_key in mapping:
                    found_keys.add(match_key)
                    if len(found_keys) == total_keys:
                        break

                if smart:
                    sub_parts = _smart_split(part)
                    for sp in sub_parts:
                        sm_key = filter_to_letters(sp) if clean_items else sp
                        if sm_key in mapping:
                            found_keys.add(sm_key)
                            if len(found_keys) == total_keys:
                                break
                    if len(found_keys) == total_keys:
                        break
            if len(found_keys) == total_keys:
                break

    duration = time.perf_counter() - start_time
    found_count = len(found_keys)
    missing_count = total_keys - found_count

    if prune:
        # Output a filtered mapping containing only the found keys
        results = []
        for key in sorted(found_keys) if process_output else mapping.keys():
            if key in found_keys:
                results.append((key, mapping[key]))

        if limit is not None:
            results = results[:limit]

        _write_paired_output(
            results,
            output_file,
            'arrow', # Default to arrow for mapping format
            "Verify (Pruned)",
            quiet,
            limit=limit
        )
        logging.info(
            f"[Verify Mode] Pruned mapping saved to '{output_file}'. "
            f"Kept {found_count} of {total_keys} entries. "
            f"Processing time: {duration:.3f}s"
        )
    else:
        # Human-readable report
        report = [
            f"{BOLD}VERIFICATION REPORT{RESET}",
            f"{BOLD}───────────────────{RESET}",
            f"Total entries in mapping: {total_keys}",
            f"Entries found in files:   {GREEN}{found_count}{RESET}",
            f"Entries missing:          {RED}{missing_count}{RESET}",
            "",
            f"{BOLD}STATUS:{RESET} " + (f"{GREEN}All entries verified.{RESET}" if missing_count == 0 else f"{YELLOW}{found_count}/{total_keys} entries verified.{RESET}"),
            ""
        ]

        if missing_count > 0:
            report.append(f"{BOLD}MISSING ENTRIES:{RESET}")
            missing_keys = [k for k in mapping.keys() if k not in found_keys]
            if process_output:
                missing_keys.sort()

            for i, key in enumerate(missing_keys):
                if limit is not None and i >= limit:
                    report.append(f"... and {len(missing_keys) - limit} more.")
                    break
                report.append(f"  - {key}")

        with smart_open_output(output_file) as out:
            for line in report:
                out.write(line + '\n')

        logging.info(
            f"[Verify Mode] Verification complete. Found {found_count}/{total_keys} entries. "
            f"Report written to '{output_file}'. "
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
        choices=['line', 'json', 'csv', 'markdown', 'md-table', 'arrow', 'table', 'yaml', 'toml'],
        metavar='FORMAT',
        default=argparse.SUPPRESS,
        help="Choose the format for the output. If not provided, it is automatically detected from the output file extension. Choices: line, json, csv, markdown, md-table, arrow, table, yaml, toml.",
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
        help="Skip items shorter than this (default: 1 for most modes, 3 for word extraction modes like 'words' and 'count').",
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
    # Items were cleaned during loading; only length filtering is needed now.
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
    """Perform set operations (intersection, union, difference, symmetric_difference) between input files (merged) and a second file."""
    start_time = time.perf_counter()
    allowed_operations = {'intersection', 'union', 'difference', 'symmetric_difference'}
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
    elif operation == 'difference':
        result_items = [item for item in unique_a if item not in set_b]
    else:  # symmetric_difference
        # Items in either set but not both
        only_a = [item for item in unique_a if item not in set_b]
        set_a = set(unique_a)
        only_b = [item for item in unique_b if item not in set_a]
        result_items = only_a + only_b

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
        "summary": "Extracts text from arrow lines",
        "description": "Finds text in lines that use arrows (like 'typo -> correction'). It saves the left side by default. Use --right to save the right side instead.",
        "example": "python multitool.py arrow typos.log --right --output corrections.txt",
        "flags": "[--right]",
    },
    "table": {
        "summary": "Extracts text from key=value",
        "description": "Gets keys or values from entries like 'key = \"value\"'. It saves the key by default. Use --right to save the quoted value instead.",
        "example": "python multitool.py table typos.toml --right -o corrections.txt",
        "flags": "[--right]",
    },
    "combine": {
        "summary": "Merges multiple files into one",
        "description": "Combines several files into one list. It removes duplicates and sorts the results alphabetically.",
        "example": "python multitool.py combine typos1.txt typos2.txt --output all_typos.txt",
        "flags": "",
    },
    "unique": {
        "summary": "Removes duplicates, keeps order",
        "description": "Removes duplicate items from your list. Unlike 'combine', it preserves the order in which items first appeared in your files.",
        "example": "python multitool.py unique raw_typos.txt --output clean_typos.txt",
        "flags": "",
    },
    "backtick": {
        "summary": "Extracts text inside backticks",
        "description": "Finds text inside backticks (like `code`). It prioritizes items near words like 'error' or 'warning' to find the most relevant data.",
        "example": "python multitool.py backtick build.log --output suspects.txt",
        "flags": "",
    },
    "quoted": {
        "summary": "Extracts text inside quotes",
        "description": "Finds text inside double (\") or single (') quotes. It handles backslash escaping (like \\\" or \\') to correctly extract strings from code or data files.",
        "example": "python multitool.py quoted source.py --output strings.txt",
        "flags": "",
    },
    "between": {
        "summary": "Extracts text between markers",
        "description": "Finds text between a starting marker and an ending marker. It supports simple text markers and can work across multiple lines if the --multi-line flag is used.",
        "example": "python multitool.py between input.txt --start '{{' --end '}}' --output items.txt",
        "flags": "--start START --end END [--multi-line]",
    },
    "csv": {
        "summary": "Extracts columns from CSV",
        "description": "Gets data from CSV files. By default, it gets every column except the first one. Use --first-column to get only the first column, --column to pick specific numbers, or --delimiter to use a custom separator.",
        "example": "python multitool.py csv typos.csv --column 2 --delimiter ';' -o corrections.txt",
        "flags": "[--first-column] [-c N] [-d DELIM]",
    },
    "markdown": {
        "summary": "Extracts Markdown list items",
        "description": "Finds text in lines starting with -, *, or +. It can also split items by ':' or '->' to get one side of a pair (use --right for the second part).",
        "example": "python multitool.py markdown notes.md --output items.txt",
        "flags": "[--right]",
    },
    "md-table": {
        "summary": "Extracts Markdown table items",
        "description": "Finds text in cells of a Markdown table. It saves the first column by default. Use --right to save the second column instead, or --column to pick specific numbers. It automatically skips header and divider rows.",
        "example": "python multitool.py md-table readme.md --column 2 --output corrections.txt",
        "flags": "[--right] [-c N]",
    },
    "json": {
        "summary": "Extracts JSON values by key",
        "description": "Finds values for a specific key in a JSON file. Use dots for nested keys (like 'user.name'). If no key is provided, it gets items from the top level. It automatically handles lists of objects.",
        "example": "python multitool.py json list.json -o items.txt",
        "flags": "[-k KEY]",
    },
    "yaml": {
        "summary": "Extracts YAML values by key",
        "description": "Finds values for a specific key in a YAML file. Use dots for nested keys (like 'config.items'). If no key is provided, it gets items from the top level. It automatically handles lists.",
        "example": "python multitool.py yaml list.yaml -o items.txt",
        "flags": "[-k KEY]",
    },
    "toml": {
        "summary": "Extracts TOML values by key",
        "description": "Finds values for a specific key in a TOML file. Use dots for nested keys (like 'tool.poetry.dependencies'). If no key is provided, it gets items from the top level. It automatically handles nested tables.",
        "example": "python multitool.py toml pyproject.toml -k tool.poetry.dependencies --output-format toml",
        "flags": "[-k KEY]",
    },
    "line": {
        "summary": "Reads a file line by line",
        "description": "Reads every line from a file, cleans the text, and writes it to the output. Useful for simple cleaning and filtering.",
        "example": "python multitool.py line raw_words.txt --output filtered.txt",
        "flags": "",
    },
    "words": {
        "summary": "Extracts words from a file",
        "description": "Splits a file into individual words using whitespace or a custom delimiter. It's the standard way to get a list of every word used in a document. Use --smart to split by capital letters and symbols.",
        "example": "python multitool.py words report.txt --smart --output wordlist.txt",
        "flags": "[-d DELIM] [-S]",
    },
    "ngrams": {
        "summary": "Extracts sequences of words",
        "description": "Gets sequences of words from a file. This is useful for finding common phrases or context around typos. It supports sequences across line boundaries.",
        "example": "python multitool.py ngrams report.txt -n 2 --smart --output phrases.txt",
        "flags": "[-n N] [-d DELIM] [-S]",
    },
    "count": {
        "summary": "Counts how often items appear",
        "description": "Counts how often each word, pair, line, or character appears and sorts the list by frequency. Use -f arrow for a rich visual report with bar charts. Use --pairs to count word pairs, --lines to count raw lines, or --chars to count individual characters. Use --by-file to count how many files contain each item. You can also provide a mapping (via --mapping or --add) to count matches of specific typos across your files.",
        "example": "python multitool.py count . --lines --min-count 5 -f arrow",
        "flags": "[-s MAPPING] [-a K:V] [-d DELIM] [-S] [-p] [-l] [-c] [-B]",
    },
    "filterfragments": {
        "summary": "Removes words found inside others",
        "description": "Removes words from your list if they appear anywhere (even as a fragment) inside words in a second file.",
        "example": "python multitool.py filterfragments list.txt reference.txt --output unique.txt",
        "flags": "[FILE2]",
    },
    "check": {
        "summary": "Finds words that are typos & fixes",
        "description": "Checks for words that appear in both the typo and correction columns of a file. Use this to find errors in your typo lists.",
        "example": "python multitool.py check typos.csv --output duplicates.txt",
        "flags": "",
    },
    "set_operation": {
        "summary": "Compares files using set logic",
        "description": "Compares two files to find shared lines (intersection), all lines (union), lines unique to the first file (difference), or lines unique to either file (symmetric_difference).",
        "example": "python multitool.py set_operation fileA.txt fileB.txt --operation difference --output unique.txt",
        "flags": "[FILE2] --operation OP",
    },
    "sample": {
        "summary": "Picks a random set of lines",
        "description": "Selects a random subset of lines. You can choose a specific number of lines (-n) or a percentage (--percent).",
        "example": "python multitool.py sample big_log.txt -n 100 -o sample.txt",
        "flags": "[-n N|--percent P]",
    },
    "regex": {
        "summary": "Finds text matching a pattern",
        "description": "Finds and gets all text that matches a Python regular expression pattern.",
        "example": "python multitool.py regex inputs.txt --pattern 'user_\\w+' --output users.txt",
        "flags": "[-r PATTERN]",
    },
    "map": {
        "summary": "Replaces items using a mapping",
        "description": "Replaces items in your list with values from a mapping file or extra pairs provided via --add. Supports CSV, Arrow, Table, JSON, and YAML mapping formats. Use --smart-case to preserve capitalization and --pairs to see both original and changed words. Length filters are re-applied to items after they are changed.",
        "example": "python multitool.py map input.txt --add teh:the --smart-case --pairs",
        "flags": "[MAPPING] [-s MAPPING] [FILES...] [-a K:V] [--smart-case] [-p]",
    },
    "zip": {
        "summary": "Pairs lines from two files",
        "description": "Joins two files line-by-line into a paired format like 'typo -> correction'. Useful for creating mapping files from two separate lists. Length filters are applied to both items in each pair.",
        "example": "python multitool.py zip typos.txt corrections.txt --output-format table --output typos.toml",
        "flags": "[FILE2]",
    },
    "swap": {
        "summary": "Reverses the order of pairs",
        "description": "Flips the left and right elements of pairs (for example, 'typo -> correction' becomes 'correction -> typo'). Supports Arrow, Table, CSV, and Markdown formats.",
        "example": "python multitool.py swap typos.csv --output-format arrow --output flipped.txt",
        "flags": "",
    },
    "pairs": {
        "summary": "Converts paired data formats",
        "description": "Reads pairs (like 'typo -> correction') from any supported format and writes them to the specified output format. Useful for cleaning, filtering, and format conversion.",
        "example": "python multitool.py pairs typos.json --output-format csv --output typos.csv",
        "flags": "",
    },
    "conflict": {
        "summary": "Finds typos with multiple fixes",
        "description": "Finds typos in your paired data that are associated with more than one unique correction. Use this to find inconsistencies in your typo lists.",
        "example": "python multitool.py conflict typos.csv --output-format arrow --output conflicts.txt",
        "flags": "",
    },
    "similarity": {
        "summary": "Filters pairs by changes",
        "description": "Filters pairs (typo -> correction) based on the number of character changes needed to turn one word into another. Use this to remove extra data or find specific types of typos.",
        "example": "python multitool.py similarity typos.txt --keyboard --show-dist",
        "flags": "[--max-dist N] [-k] [-t] [--show-dist]",
    },
    "near_duplicates": {
        "summary": "Finds similar words in one list",
        "description": "Finds pairs of words in your list that are very similar (only a few characters apart). Use this to find potential typos or unintended duplicates in a project.",
        "example": "python multitool.py near_duplicates words.txt --keyboard --show-dist",
        "flags": "[--max-dist N] [-k] [-t] [--show-dist]",
    },
    "fuzzymatch": {
        "summary": "Finds similar words in two lists",
        "description": "Finds words in your list that are similar to words in a second list (large dictionary). Use this to find likely corrections for typos. It defaults to a threshold of 1 character change.",
        "example": "python multitool.py fuzzymatch typos.txt words.csv --keyboard --show-dist",
        "flags": "[FILE2] [--max-dist N] [-k] [-t] [--show-dist]",
    },
    "stats": {
        "summary": "Calculates stats for a list",
        "description": "Provides a detailed overview of your dataset. It reports counts, unique items, statistics, and (optionally) paired data stats like conflicts, overlaps, and the number of changes between words.",
        "example": "python multitool.py stats typos.csv --pairs --output-format json",
        "flags": "[-p]",
    },
    "classify": {
        "summary": "Groups typos by error type",
        "description": "Labels typo pairs with error codes like [K] Keyboard, [T] Transposition, [D] Deletion, [I] Insertion, [R] Replacement, and [M] Multiple letters. Use --show-dist to include the number of character changes.",
        "example": "python multitool.py classify typos.txt --show-dist --output labeled.txt",
        "flags": "[--show-dist]",
    },
    "discovery": {
        "summary": "Finds typos in rare words",
        "description": "Automatically finds potential typos in a text by seeing rare words that are very similar to frequent words. It assumes that frequent words are likely correct and rare variations are likely typos. This is a powerful way to find errors without needing a dictionary.",
        "example": "python multitool.py discovery code.py --keyboard --smart",
        "flags": "[--rare-max N] [--freq-min N] [-S] [-k] [-t]",
    },
    "casing": {
        "summary": "Finds inconsistent casing",
        "description": "Finds words that appear in your files with multiple different casing styles (for example, 'hello', 'Hello', 'HELLO'). This is useful for seeing inconsistent naming or typos that differ only by case.",
        "example": "python multitool.py casing report.txt --smart --output-format arrow",
        "flags": "[-d DELIM] [-S]",
    },
    "cycles": {
        "summary": "Finds loops in typo pairs",
        "description": "Detects cycles in your typo mappings (for example, 'A' maps to 'B' and 'B' maps back to 'A'). Repeated loops can cause issues when automatically fixing typos and represent logic errors in your data.",
        "example": "python multitool.py cycles typos.csv --output-format arrow",
        "flags": "",
    },
    "repeated": {
        "summary": "Finds doubled words",
        "description": "Finds doubled words (for example, 'the the') in your text. It outputs the duplicated pair and the suggested fix. Use --smart to handle CamelCase or punctuation.",
        "example": "python multitool.py repeated report.txt --smart --output-format arrow",
        "flags": "[-d DELIM] [-S]",
    },
    "standardize": {
        "summary": "Fixes casing/spelling project-wide",
        "description": "Analyzes your files to find words used with different capitalization (for example, 'database' vs 'Database') or similar spelling (for example, 'teh' vs 'the'). It then automatically replaces all less frequent versions with the most popular one across the entire project. Use --fuzzy to enable similar word matching based on your project's dominant patterns.",
        "example": "python multitool.py standardize . --diff --min-length 4 --fuzzy 1",
        "flags": "[FILES...] [--in-place] [--dry-run] [--fuzzy N] [--threshold RATIO] [--diff]",
    },
    "search": {
        "summary": "Searches for words or patterns",
        "description": "A typo-aware search tool. It searches for a query in your files and can find similar words (typos) or subword matches. It supports highlighting, line numbers, and context lines.",
        "example": "python multitool.py search 'teh' report.txt --keyboard --line-numbers",
        "flags": "[QUERY] [-Q QUERY] [FILES...] [-S] [-k] [-t] [-n] [-B N] [-A N] [-C N]",
    },
    "scan": {
        "summary": "Scans project for known typos",
        "description": "Like a batch version of the 'search' mode. It searches for every word in a mapping file or provided via --add and reports all matches with filename, line number, and highlighting. It also supports context lines.",
        "example": "python multitool.py scan . --add teh:the --smart -A 1",
        "flags": "[MAPPING] [-s MAPPING] [FILES...] [-a K:V] [-S] [-B N] [-A N] [-C N]",
    },
    "verify": {
        "summary": "Checks if typos exist in project",
        "description": "Checks a mapping file or extra pairs against your files to see which ones are actually present. Use --prune to output a mapping containing only the found typos. Use --smart to also search for subword matches in larger compound words.",
        "example": "python multitool.py verify . --mapping typos.csv --prune",
        "flags": "[MAPPING] [-s MAPPING] [FILES...] [-a K:V] [-S] [--prune]",
    },
    "scrub": {
        "summary": "Fixes typos in text files",
        "description": "Performs in-place replacements of typos in your text files using a mapping file or extra pairs provided via --add. It tries to preserve the surrounding context (punctuation, whitespace) while fixing errors. It automatically handles compound words like 'CamelCase' and 'snake_case' variables. Supports CSV, Arrow, Table, JSON, and YAML mapping formats.",
        "example": "python multitool.py scrub input.txt --add teh:the --diff",
        "flags": "[MAPPING] [-s MAPPING] [FILES...] [-a K:V] [--in-place] [--smart-case] [--diff]",
    },
    "align": {
        "summary": "Aligns typo-correction pairs",
        "description": "Extracts typo-correction pairs from any supported format (CSV, arrow, Markdown lists/tables) and outputs them in perfectly aligned columns by automatically calculating the maximum width of the left column. It supports a custom separator string via the --sep flag.",
        "example": "python multitool.py align typos.csv --sep ' -> ' --output-format aligned",
        "flags": "[--sep SEP]",
    },
    "rename": {
        "summary": "Batch renames files and folders",
        "description": "Renames files or directories based on a typo mapping or extra pairs provided via --add. It preserves the directory structure and can automatically handle CamelCase or snake_case names using --smart-case. It handles nested renames by processing files before their parent directories.",
        "example": "python multitool.py rename src/ --add teh:the --in-place",
        "flags": "[MAPPING] [-s MAPPING] [FILES...] [-a K:V] [--in-place] [--dry-run] [--smart-case]",
    },
    "diff": {
        "summary": "Finds differences between files",
        "description": "Finds differences between two files or lists. It can track simple word additions/removals or (with --pairs) find changed corrections for existing typos. Color-coded output highlights what is added (+), what is removed (-), and what changed (~).",
        "example": "python multitool.py diff old_typos.csv new_typos.csv --pairs --output-format json",
        "flags": "[FILE2] [-p]",
    },
    "highlight": {
        "summary": "Color-codes words from a list",
        "description": "Searches for words from a mapping file or extra pairs provided via --add and highlights them with color in the output. Useful as a non-destructive preview before using 'scrub'. Supports the same smart word detection as the typo-fixing tool.",
        "example": "python multitool.py highlight input.txt --add teh:the",
        "flags": "[MAPPING] [-s MAPPING] [FILES...] [-a K:V] [-S]",
    },
    "resolve": {
        "summary": "Shortens typo correction chains",
        "description": "Finds and shortens chains of corrections (for example, 'A' -> 'B' and 'B' -> 'C' becomes 'A' -> 'C'). This ensures that your mapping files always point directly to the final correct word, which improves the efficiency of fixing typos and analysis.",
        "example": "python multitool.py resolve mappings.csv --output resolved.csv",
        "flags": "",
    },
}


def get_mode_summary_text() -> str:
    """Return a formatted summary table of all available modes as a string."""
    categories = {
        "GET DATA": ["arrow", "table", "backtick", "quoted", "between", "csv", "markdown", "md-table", "json", "yaml", "toml", "line", "words", "ngrams", "regex"],
        "CHANGE DATA": ["combine", "unique", "diff", "highlight", "resolve", "align", "rename", "filterfragments", "set_operation", "sample", "map", "zip", "swap", "pairs", "scrub", "standardize"],
        "CHECK & ANALYZE": ["count", "check", "conflict", "cycles", "similarity", "near_duplicates", "fuzzymatch", "stats", "classify", "discovery", "casing", "repeated", "search", "scan", "verify"],
    }

    use_color = _should_enable_color(sys.stdout)
    c_bold = BOLD if use_color else ""
    c_blue = BLUE if use_color else ""
    c_green = GREEN if use_color else ""
    c_yellow = YELLOW if use_color else ""
    c_reset = RESET if use_color else ""

    lines = []
    lines.append(f"{c_bold}Available Modes:{c_reset}")

    width_mode = 15
    width_summary = 33 # Adjusted for alignment with vertical bars

    # Header and divider elements
    padding = "  "
    sep = f"{c_bold}{c_blue}│{c_reset}"
    cross = f"{c_bold}{c_blue}┼{c_reset}"
    bottom = f"{c_bold}{c_blue}┴{c_reset}"

    header = (
        f"{padding}{c_bold}{c_blue}{'Mode':<{width_mode}}{c_reset} {sep} "
        f"{c_bold}{c_blue}{'Summary':<{width_summary}}{c_reset} {sep} "
        f"{c_bold}{c_blue}Primary Options{c_reset}"
    )
    divider = f"{padding}{c_bold}{c_blue}{'─' * width_mode}─{cross}─{'─' * width_summary}─{cross}─{'─' * 15}{c_reset}"

    lines.append("\n" + header)
    lines.append(divider)

    for category, modes in categories.items():
        cat_header = f"{padding}{c_bold}{c_blue}{category}{c_reset}"
        lines.append(cat_header)
        lines.append(divider)

        for mode in modes:
            if mode in MODE_DETAILS:
                details = MODE_DETAILS[mode]
                summary = details['summary'].rstrip('.')
                flags = details.get('flags', '')
                if len(summary) > width_summary:
                    summary = summary[:width_summary-3] + "..."

                row = (
                    f"{padding}{c_bold}{c_green}{mode:<{width_mode}}{c_reset} {sep} "
                    f"{summary:<{width_summary}} {sep} "
                    f"{c_yellow}{flags}{c_reset}"
                )
                lines.append(row)

    # Closing line for the table
    lines.append(f"{padding}{c_bold}{c_blue}{'─' * width_mode}─{bottom}─{'─' * width_summary}─{bottom}─{'─' * 15}{c_reset}")

    # Quick Tips section
    lines.append(f"\n{padding}{c_bold}{c_blue}QUICK TIPS & GLOBAL FLAGS{c_reset}")
    lines.append(f"{padding}{c_bold}{c_blue}─────────────────────────{c_reset}")
    tips = [
        ("-f arrow", "Rich visual reports with bar charts and percentages"),
        ("-P", "Automatically sort results and remove duplicates"),
        ("--in-place", "Modify files directly (for scrub, standardize, rename)"),
        ("--raw (-R)", "Keep original text (no lowercase, no cleaning)"),
        ("--limit (-L)", "Restrict the number of items in the output")
    ]
    for flag, desc in tips:
        lines.append(f"{padding}{c_bold}{c_green}{flag:<13}{c_reset} {desc}")

    lines.append(f"\nRun '{c_bold}python multitool.py help <mode>{c_reset}' for details on a specific mode.\n")
    return "\n".join(lines)


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
        # Determine if color should be used for this log record
        use_color = _should_enable_color(sys.stderr)

        if use_color and levelname:
            color = self.LEVEL_COLORS.get(record.levelno)
            if color:
                levelname = f"{color}{levelname}{RESET}"

        return f"{levelname}: {record.getMessage()}"


def show_mode_help(mode_name: str | None, parser: argparse.ArgumentParser) -> None:
    """Prints detailed help for one or all modes and exits."""
    if mode_name in (None, "all"):
        # Show a summary table of all modes
        print("\n" + get_mode_summary_text())
        parser.exit()
    else:
        # Show detailed help for a single mode
        details = MODE_DETAILS.get(mode_name)
        if not details:
            parser.error(f"Unknown mode: {mode_name}")

        divider = f"{BLUE}{'─' * 80}{RESET}"
        label_color = f"{BLUE}{BOLD}"
        block = [
            divider,
            f"{label_color}{'MODE:':<13}{RESET}{GREEN}{mode_name.upper()}{RESET}",
            divider,
            f"{label_color}{'SUMMARY:':<13}{RESET}{details['summary']}",
        ]

        if details.get("description"):
            desc = details['description']
            block.append(f"{label_color}{'DESCRIPTION:':<13}{RESET}{desc}")

        flags_str = details.get("flags", "[FILES...]")
        all_tokens = flags_str.split()

        # Identify positional arguments (uppercase words not preceded by a flag)
        pos_args = []
        for i, token in enumerate(all_tokens):
            # Positional if it doesn't start with - and is not an argument to a preceding flag
            if token.startswith('-') or (token.startswith('[') and token[1] == '-'):
                continue
            if i > 0:
                prev = all_tokens[i-1]
                if prev.startswith('-') or (prev.startswith('[') and prev[1] == '-'):
                    continue

            clean_token = token.strip('[]().')
            if clean_token.isupper() and clean_token not in ("FILES...", "FILES"):
                pos_args.append(token)

        pos_args_str = (" " + " ".join(pos_args)) if pos_args else ""
        usage_line = f"python {parser.prog} {mode_name}{pos_args_str} [FILES...] [OPTIONS]"

        block.append(f"\n{label_color}{'USAGE:':<13}{RESET}{BOLD}{usage_line}{RESET}")

        if details.get("flags"):
            # Filter out positional arguments and generic FILES labels from the options display
            filtered_tokens = []
            for token in all_tokens:
                if token in pos_args or token in ("[FILES...]", "FILES..."):
                    continue
                filtered_tokens.append(token)

            if filtered_tokens:
                options_str = " ".join(filtered_tokens)
                block.append(f"{label_color}{'OPTIONS:':<13}{RESET}{YELLOW}{options_str}{RESET}")

        if details.get("example"):
            block.append(f"\n{label_color}{'EXAMPLE:':<13}{RESET}")
            block.append(f"  {GREEN}{details['example']}{RESET}")

        block.append(divider)

        parser.exit(message="\n" + "\n".join(block) + "\n\n")


class ModeHelpAction(argparse.Action):
    """Custom argparse action that prints detailed help for one or all modes."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | None,
        option_string: str | None = None,
    ) -> None:
        show_mode_help(values, parser)


def _build_parser() -> argparse.ArgumentParser:
    # Build a grouped mode summary for the epilog
    mode_summary = get_mode_summary_text()

    parser = argparse.ArgumentParser(
        description="A multipurpose tool for cleaning, getting, and analyzing text files.",
        epilog=dedent(
            f"""
            {BLUE}Examples:{RESET}
              {GREEN}python multitool.py --mode-help{RESET}             # Show a summary of every mode
              {GREEN}python multitool.py --mode-help csv{RESET}         # Describe the CSV getting mode
              {GREEN}python multitool.py arrow file.txt{RESET}          # Run a specific mode
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
        choices=['line', 'json', 'csv', 'markdown', 'md-table', 'arrow', 'table', 'yaml', 'toml'],
        metavar='FORMAT',
        default=None,
        help="Choose the format for the output. If not provided, it is automatically detected from the output file extension. Choices: line, json, csv, markdown, md-table, arrow, table, yaml, toml.",
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
        default=None,
        help="Skip items shorter than this (default: 1 for most modes, 3 for word extraction modes like 'words' and 'count').",
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

    help_parser = subparsers.add_parser(
        'help',
        help="Show help for a specific mode or a summary of all modes.",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Displays extended documentation for the requested mode. If no mode is provided, shows a summary table of all available modes.",
        epilog=f"{BLUE}Examples:{RESET}\n  {GREEN}python multitool.py help{RESET}          # Show summary of all modes\n  {GREEN}python multitool.py help count{RESET}    # Show detailed help for 'count' mode",
    )
    help_parser.add_argument(
        'mode_to_help',
        nargs='?',
        choices=[*MODE_DETAILS.keys(), 'all'],
        metavar='MODE',
        help="The mode to show help for (for example, 'count', 'scrub', 'standardize').",
    )

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
        help="Get the right side (correction) instead of the left side (typo).",
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
        help="Get the value (right side) instead of the key (left side).",
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

    quoted_parser = subparsers.add_parser(
        'quoted',
        help=MODE_DETAILS['quoted']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['quoted']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['quoted']['example']}{RESET}",
    )
    _add_common_mode_arguments(quoted_parser)

    between_parser = subparsers.add_parser(
        'between',
        help=MODE_DETAILS['between']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['between']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['between']['example']}{RESET}",
    )
    between_options = between_parser.add_argument_group(f"{BLUE}BETWEEN OPTIONS{RESET}")
    between_options.add_argument(
        '--start',
        type=str,
        required=True,
        help="The starting marker to find.",
    )
    between_options.add_argument(
        '--end',
        type=str,
        required=True,
        help="The ending marker to find.",
    )
    between_options.add_argument(
        '--multi-line',
        action='store_true',
        help="Allow markers to span across multiple lines.",
    )
    _add_common_mode_arguments(between_parser)

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
        help='Get the first column instead of subsequent columns.',
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
        metavar='NUMBER',
        help='One or more 0-based column numbers to get.',
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
        help="Get the right side of a pair (split by ':' or '->') instead of the left side.",
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
        help="Get the second column instead of the first.",
    )
    md_table_options.add_argument(
        '-c', '--column',
        dest='columns',
        type=int,
        nargs='+',
        metavar='NUMBER',
        help='One or more 0-based column numbers to get.',
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
        help="The key path to get (for example 'items.name'). If not provided, gets from the top level.",
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
        help="The key path to get (for example 'config.items'). If not provided, gets from the top level.",
    )
    _add_common_mode_arguments(yaml_parser)

    toml_parser = subparsers.add_parser(
        'toml',
        help=MODE_DETAILS['toml']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['toml']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['toml']['example']}{RESET}",
    )
    toml_options = toml_parser.add_argument_group(f"{BLUE}TOML OPTIONS{RESET}")
    toml_options.add_argument(
        '-k', '--key',
        type=str,
        default='',
        help="The key path to get (for example 'tool.poetry.dependencies'). If not provided, gets from the top level.",
    )
    _add_common_mode_arguments(toml_parser)

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
        help="Minimum match count to include an item in the output (default: 1).",
    )
    count_options.add_argument(
        '--max-count',
        type=int,
        help="Maximum match count to include an item in the output.",
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
        '-B', '--by-file',
        action='store_true',
        help='Count how many files contain each item instead of total matches.',
    )
    unit_group = count_options.add_mutually_exclusive_group()
    unit_group.add_argument(
        '-p', '--pairs',
        action='store_true',
        help='Count frequencies of word pairs (for example, typo -> correction) instead of single words.',
    )
    unit_group.add_argument(
        '-l', '--lines',
        action='store_true',
        help='Count frequencies of raw lines instead of individual words.',
    )
    unit_group.add_argument(
        '-c', '--chars',
        action='store_true',
        help='Count frequencies of individual characters instead of individual words.',
    )
    count_options.add_argument(
        '-s', '--mapping',
        type=str,
        required=False,
        help='Path to the mapping file for auditing.',
    )
    count_options.add_argument(
        '-a', '--add',
        dest='ad_hoc',
        type=str,
        nargs='+',
        metavar='KEY:VALUE',
        help='Extra mapping pairs (for example "teh:the") for auditing.',
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
        '-k', '--keyboard',
        action='store_true',
        help="Only include typos likely caused by hitting a nearby key.",
    )
    similarity_options.add_argument(
        '-t', '--transposition',
        action='store_true',
        help="Only include typos likely caused by swapping two adjacent letters.",
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
        '-k', '--keyboard',
        action='store_true',
        help="Only include typos likely caused by hitting a nearby key.",
    )
    nd_options.add_argument(
        '-t', '--transposition',
        action='store_true',
        help="Only include typos likely caused by swapping two adjacent letters.",
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
    fm_options = fuzzymatch_parser.add_argument_group(f"{BLUE}SIMILAR WORD MATCH OPTIONS{RESET}")
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
        '-k', '--keyboard',
        action='store_true',
        help="Only include typos likely caused by hitting a nearby key.",
    )
    fm_options.add_argument(
        '-t', '--transposition',
        action='store_true',
        help="Only include typos likely caused by swapping two adjacent letters.",
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
        '-k', '--keyboard',
        action='store_true',
        help="Only include typos likely caused by hitting a nearby key.",
    )
    discovery_options.add_argument(
        '-t', '--transposition',
        action='store_true',
        help="Only include typos likely caused by swapping two adjacent letters.",
    )
    discovery_options.add_argument(
        '--show-dist',
        action='store_true',
        help="Include the number of character changes in the output.",
    )
    discovery_options.add_argument(
        '-d', '--delimiter',
        type=str,
        help='The delimiter character to split words by (default: whitespace).',
    )
    discovery_options.add_argument(
        '-S', '--smart',
        action='store_true',
        help='Split by symbols and capital letters (for example, splitting "CamelCase" into "Camel" and "Case").',
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

    search_parser = subparsers.add_parser(
        'search',
        help=MODE_DETAILS['search']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['search']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['search']['example']}{RESET}",
    )
    search_options = search_parser.add_argument_group(f"{BLUE}SEARCH OPTIONS{RESET}")
    search_options.add_argument(
        '-Q', '--query',
        type=str,
        required=False,
        help="The word or pattern to search for.",
    )
    search_options.add_argument(
        '--max-dist',
        type=int,
        default=0,
        help="Maximum character changes for similar word matching (default: 0).",
    )
    search_options.add_argument(
        '-S', '--smart',
        action='store_true',
        help='Search for subwords using smart splitting (for example, finding "teh" inside "tehWord").',
    )
    search_options.add_argument(
        '-k', '--keyboard',
        action='store_true',
        help="Only include matches likely caused by hitting a nearby key.",
    )
    search_options.add_argument(
        '-t', '--transposition',
        action='store_true',
        help="Only include matches likely caused by swapping two adjacent letters.",
    )
    search_options.add_argument(
        '-n', '--line-numbers',
        action='store_true',
        help="Show line numbers in the search results.",
    )
    search_options.add_argument(
        '--with-filename',
        action='store_true',
        dest='with_filename',
        default=None,
        help="Force the output of the filename for each match.",
    )
    search_options.add_argument(
        '--no-filename',
        action='store_false',
        dest='with_filename',
        help="Suppress the prefixing of filenames on output.",
    )
    search_options.add_argument(
        '-B', '--before-context',
        type=int,
        default=0,
        help="Show this many lines of context before each match.",
    )
    search_options.add_argument(
        '-A', '--after-context',
        type=int,
        default=0,
        help="Show this many lines of context after each match.",
    )
    search_options.add_argument(
        '-C', '--context',
        type=int,
        help="Show this many lines of context before and after each match.",
    )
    _add_common_mode_arguments(search_parser)

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
        choices=['intersection', 'union', 'difference', 'symmetric_difference'],
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
        help='The number of words in each sequence (default: 2).',
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
        '-a', '--add',
        dest='ad_hoc',
        type=str,
        nargs='+',
        metavar='KEY:VALUE',
        help='Extra mapping pairs (for example "teh:the") or words to match.',
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
        '-a', '--add',
        dest='ad_hoc',
        type=str,
        nargs='+',
        metavar='KEY:VALUE',
        help='Extra mapping pairs (for example "teh:the") or words to match.',
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
    scrub_options.add_argument(
        '--diff',
        action='store_true',
        help="Show a unified diff of the changes that would be made.",
    )
    _add_common_mode_arguments(scrub_parser, include_process_output=False, include_limit=False)

    rename_parser = subparsers.add_parser(
        'rename',
        help=MODE_DETAILS['rename']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['rename']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['rename']['example']}{RESET}",
    )
    rename_options = rename_parser.add_argument_group(f"{BLUE}RENAME OPTIONS{RESET}")
    rename_options.add_argument(
        '-s', '--mapping',
        type=str,
        required=False,
        help='Path to the mapping file.',
    )
    rename_options.add_argument(
        '-a', '--add',
        dest='ad_hoc',
        type=str,
        nargs='+',
        metavar='KEY:VALUE',
        help='Extra mapping pairs (for example "teh:the") or words to match.',
    )
    rename_options.add_argument(
        '--in-place',
        action='store_true',
        help="Perform the actual renaming of files and directories.",
    )
    rename_options.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be renamed without making any changes.",
    )
    rename_options.add_argument(
        '--smart-case',
        action='store_true',
        help="Automatically match the casing of the original word.",
    )
    _add_common_mode_arguments(rename_parser, include_process_output=False, include_limit=True)

    standardize_parser = subparsers.add_parser(
        'standardize',
        help=MODE_DETAILS['standardize']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['standardize']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['standardize']['example']}{RESET}",
    )
    standardize_options = standardize_parser.add_argument_group(f"{BLUE}STANDARDIZE OPTIONS{RESET}")
    standardize_options.add_argument(
        '--in-place',
        nargs='?',
        const='',
        metavar='EXT',
        help="Modify files in place. If an extension is provided (for example, '.bak'), a backup is created.",
    )
    standardize_options.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be changed without modifying any files.",
    )
    standardize_options.add_argument(
        '--fuzzy',
        type=int,
        default=0,
        help="Maximum distance for similar word matching (for example, --fuzzy 1 to fix 'teh' -> 'the'). Set to 0 to only fix casing (default: 0).",
    )
    standardize_options.add_argument(
        '--threshold',
        type=float,
        default=10.0,
        help="The minimum frequency ratio to consider a rare word a typo (default: 10.0).",
    )
    standardize_options.add_argument(
        '--diff',
        action='store_true',
        help="Show a unified diff of the changes that would be made.",
    )
    _add_common_mode_arguments(standardize_parser, include_process_output=False, include_limit=True)

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
        '-a', '--add',
        dest='ad_hoc',
        type=str,
        nargs='+',
        metavar='KEY:VALUE',
        help='Extra mapping pairs (for example "teh:the") or words to match.',
    )
    highlight_options.add_argument(
        '-S', '--smart',
        action='store_true',
        help='Highlight subword matches (for example, highlighting "teh" inside "tehWord").',
    )
    _add_common_mode_arguments(highlight_parser)

    scan_parser = subparsers.add_parser(
        'scan',
        help=MODE_DETAILS['scan']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['scan']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['scan']['example']}{RESET}",
    )
    scan_options = scan_parser.add_argument_group(f"{BLUE}SCAN OPTIONS{RESET}")
    scan_options.add_argument(
        '-s', '--mapping',
        type=str,
        required=False,
        help='Path to the mapping file or word list.',
    )
    scan_options.add_argument(
        '-a', '--add',
        dest='ad_hoc',
        type=str,
        nargs='+',
        metavar='KEY:VALUE',
        help='Extra mapping pairs (for example "teh:the") or words to match.',
    )
    scan_options.add_argument(
        '-S', '--smart',
        action='store_true',
        help='Scan for subword matches (for example, finding "teh" inside "tehWord").',
    )
    scan_options.add_argument(
        '-n', '--line-numbers',
        action='store_true',
        help="Show line numbers in the results.",
    )
    scan_options.add_argument(
        '--with-filename',
        action='store_true',
        dest='with_filename',
        default=None,
        help="Force the output of the filename for each match.",
    )
    scan_options.add_argument(
        '--no-filename',
        action='store_false',
        dest='with_filename',
        help="Suppress the prefixing of filenames on output.",
    )
    scan_options.add_argument(
        '-B', '--before-context',
        type=int,
        default=0,
        help="Show this many lines of context before each match.",
    )
    scan_options.add_argument(
        '-A', '--after-context',
        type=int,
        default=0,
        help="Show this many lines of context after each match.",
    )
    scan_options.add_argument(
        '-C', '--context',
        type=int,
        help="Show this many lines of context before and after each match.",
    )
    _add_common_mode_arguments(scan_parser)

    verify_parser = subparsers.add_parser(
        'verify',
        help=MODE_DETAILS['verify']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['verify']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['verify']['example']}{RESET}",
    )
    verify_options = verify_parser.add_argument_group(f"{BLUE}VERIFY OPTIONS{RESET}")
    verify_options.add_argument(
        '-s', '--mapping',
        type=str,
        required=False,
        help='Path to the mapping file or word list.',
    )
    verify_options.add_argument(
        '-a', '--add',
        dest='ad_hoc',
        type=str,
        nargs='+',
        metavar='KEY:VALUE',
        help='Extra mapping pairs (for example "teh:the") or words to verify.',
    )
    verify_options.add_argument(
        '-S', '--smart',
        action='store_true',
        help='Search for subword matches in larger compound words.',
    )
    verify_options.add_argument(
        '--prune',
        action='store_true',
        help='Output a mapping containing only the found typos.',
    )
    _add_common_mode_arguments(verify_parser)

    resolve_parser = subparsers.add_parser(
        'resolve',
        help=MODE_DETAILS['resolve']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['resolve']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['resolve']['example']}{RESET}",
    )
    _add_common_mode_arguments(resolve_parser)

    align_parser = subparsers.add_parser(
        'align',
        help=MODE_DETAILS['align']['summary'],
        formatter_class=argparse.RawTextHelpFormatter,
        description=MODE_DETAILS['align']['description'],
        epilog=f"{BLUE}Example:{RESET}\n  {GREEN}{MODE_DETAILS['align']['example']}{RESET}",
    )
    align_options = align_parser.add_argument_group(f"{BLUE}ALIGN OPTIONS{RESET}")
    align_options.add_argument(
        '--sep',
        type=str,
        default=" -> ",
        help="The separator string to use between aligned columns (default: ' -> ').",
    )
    _add_common_mode_arguments(align_parser)


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
        print("\n" + get_mode_summary_text())
        sys.exit(0)

    parser = _build_parser()
    argv = _normalize_mode_args(sys.argv[1:], parser)

    args = parser.parse_args(argv)

    # Implement sensible context-sensitive defaults for --min-length
    if args.min_length is None:
        if args.mode in ('words', 'ngrams', 'stats'):
            args.min_length = 3
        elif args.mode == 'count':
            # Count mode uses 3 for word extraction, but 1 for auditing, lines, or character counting
            if any([getattr(args, 'pairs', False), getattr(args, 'chars', False), getattr(args, 'lines', False),
                    getattr(args, 'mapping', None), getattr(args, 'ad_hoc', None)]):
                args.min_length = 1
            else:
                args.min_length = 3
        else:
            # Finding, auditing, and processing modes default to 1 to avoid missing data
            args.min_length = 1

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

    # Expand glob patterns and directories for input paths
    expanded_paths = []
    for path in input_paths:
        if path == '-':
            expanded_paths.append(path)
            continue

        matches = glob.glob(path)
        if not matches:
            matches = [path]

        for match in sorted(matches):
            if os.path.isdir(match):
                # Recursively expand directory. For 'rename' mode, use bottom-up traversal
                # to ensure contents are processed before their parent directories.
                # Skip common noise folders for performance and to reduce clutter.
                exclude = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.pytest_cache', '.ruff_cache', '.vscode', '.idea', 'dist', 'build'}
                for root, dirs, files in os.walk(match, topdown=(args.mode != 'rename')):
                    # If we encounter an excluded folder in the path, skip it and its contents
                    if any(part in exclude for part in root.split(os.sep)):
                        dirs[:] = []  # Don't recurse further if topdown=True
                        continue
                    
                    if args.mode != 'rename':
                        # Prune directories in-place for efficiency when topdown=True
                        dirs[:] = [d for d in dirs if d not in exclude]

                    if args.mode == 'rename':
                        for d in sorted(dirs):
                            if d not in exclude:
                                expanded_paths.append(os.path.join(root, d))
                    for f in sorted(files):
                        expanded_paths.append(os.path.join(root, f))

                if args.mode == 'rename':
                    expanded_paths.append(match)
            else:
                expanded_paths.append(match)

    # Deduplicate while preserving order
    input_paths = list(dict.fromkeys(expanded_paths))

    # Default to standard input ('-') if neither is provided
    if not input_paths:
        input_paths = ['-']

    # Store for handler
    args.input = input_paths

    # Resolve context flags (merge -C into -A and -B if -A/-B are not set)
    before_context = getattr(args, 'before_context', 0)
    after_context = getattr(args, 'after_context', 0)
    context = getattr(args, 'context', None)
    if context is not None:
        if before_context == 0:
            before_context = context
        if after_context == 0:
            after_context = context

    # Fallback logic for modes that require a secondary file (for example, zip, map, scrub, highlight)
    # If the flag is missing, we check positional arguments.
    if args.mode in {'zip', 'filterfragments', 'set_operation', 'fuzzymatch', 'diff'}:
        if getattr(args, 'file2', None) is None:
            if len(input_paths) >= 2:
                # For comparison/joining modes, use the last positional argument as the secondary file.
                args.file2 = input_paths.pop()
                args.input = input_paths
            elif len(input_paths) == 1 and input_paths[0] != '-':
                # Use the only positional argument as the secondary file and read input from stdin.
                args.file2 = input_paths[0]
                args.input = ['-']
    elif args.mode in {'map', 'scrub', 'rename', 'highlight', 'scan', 'verify'}:
        if getattr(args, 'mapping', None) is None and not getattr(args, 'ad_hoc', None):
            if len(input_paths) >= 2:
                # For pattern/mapping modes, use the first positional argument as the mapping.
                args.mapping = input_paths.pop(0)
                args.input = input_paths
            elif len(input_paths) == 1 and input_paths[0] != '-':
                # Use the only positional argument as the mapping and read input from stdin.
                args.mapping = input_paths[0]
                args.input = ['-']
    elif args.mode == 'search':
        if getattr(args, 'query', None) is None:
            if len(input_paths) >= 2:
                # Use the first positional argument as the query.
                args.query = input_paths.pop(0)
                args.input = input_paths
            elif len(input_paths) == 1 and input_paths[0] != '-':
                # Use the only positional argument as the query and read input from stdin.
                args.query = input_paths[0]
                args.input = ['-']

    file2 = getattr(args, 'file2', None)
    # Check for missing secondary files after fallback attempt
    if args.mode in {'zip', 'filterfragments', 'set_operation', 'fuzzymatch', 'diff'} and file2 is None:
        logging.error(f"{args.mode.capitalize()} mode requires a secondary file (provide FILE2 positionally or use --file2).")
        sys.exit(1)
    if args.mode in {'map', 'scrub', 'rename', 'highlight', 'scan', 'verify'} and \
       getattr(args, 'mapping', None) is None and not getattr(args, 'ad_hoc', None):
        logging.error(f"{args.mode.capitalize()} mode requires a mapping file or extra pairs (use --mapping or --add).")
        sys.exit(1)
    if args.mode == 'search' and getattr(args, 'query', None) is None:
        logging.error("Search mode requires a search query (provide QUERY positionally or use --query).")
        sys.exit(1)

    operation = getattr(args, 'operation', None)
    first_column = getattr(args, 'first_column', False)
    delimiter = getattr(args, 'delimiter', None)
    if delimiter == "":
        delimiter = None

    right_side = getattr(args, 'right', False)
    sample_count = getattr(args, 'sample_count', None)
    sample_percent = getattr(args, 'sample_percent', None)
    limit = getattr(args, 'limit', None)
    output_format = getattr(args, 'output_format', None)
    if output_format is None:
        allowed_formats = ['line', 'json', 'csv', 'markdown', 'md-table', 'arrow', 'table', 'yaml', 'toml']
        output_format = _detect_format_from_extension(args.output, allowed_formats, 'line')

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
        'align': (
            align_mode,
            {
                **common_kwargs,
                'separator': getattr(args, 'sep', " -> "),
                'output_format': output_format if output_format != 'line' else 'aligned',
            },
        ),
        'arrow': (
            arrow_mode,
            {
                **common_kwargs,
                'right_side': right_side,
                'output_format': output_format,
            },
        ),
        'resolve': (
            resolve_mode,
            {
                **common_kwargs,
                'output_format': output_format,
            },
        ),
        'ngrams': (
            ngrams_mode,
            {
                **common_kwargs,
                'n': getattr(args, 'n', 2),
                'delimiter': delimiter,
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
                'delimiter': delimiter,
                'smart': getattr(args, 'smart', False),
                'output_format': output_format,
            },
        ),
        'casing': (
            casing_mode,
            {
                **common_kwargs,
                'delimiter': delimiter,
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
        'quoted': (
            quoted_mode,
            {**common_kwargs, 'output_format': output_format},
        ),
        'between': (
            between_mode,
            {
                **common_kwargs,
                'start': getattr(args, 'start', ''),
                'end': getattr(args, 'end', ''),
                'multi_line': getattr(args, 'multi_line', False),
                'output_format': output_format,
            },
        ),
        'csv': (
            csv_mode,
            {
                **common_kwargs,
                'first_column': first_column,
                'delimiter': delimiter or ',',
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
        'toml': (
            toml_mode,
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
                'delimiter': delimiter,
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
                'delimiter': delimiter,
                'smart': getattr(args, 'smart', False),
                'pairs': getattr(args, 'pairs', False),
                'lines': getattr(args, 'lines', False),
                'chars': getattr(args, 'chars', False),
                'mapping_file': getattr(args, 'mapping', None),
                'ad_hoc': getattr(args, 'ad_hoc', None),
                'by_file': getattr(args, 'by_file', False),
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
        'search': (
            search_mode,
            {
                **common_kwargs,
                'query': getattr(args, 'query', ''),
                'max_dist': getattr(args, 'max_dist', 0),
                'keyboard': getattr(args, 'keyboard', False),
                'transposition': getattr(args, 'transposition', False),
                'smart': getattr(args, 'smart', False),
                'line_numbers': getattr(args, 'line_numbers', False),
                'with_filename': getattr(args, 'with_filename', None),
                'before_context': before_context,
                'after_context': after_context,
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
                'mapping_file': getattr(args, 'mapping', None),
                'ad_hoc': getattr(args, 'ad_hoc', None),
                'drop_missing': getattr(args, 'drop_missing', False),
                'output_format': output_format,
                'pairs': getattr(args, 'pairs', False),
                'smart_case': getattr(args, 'smart_case', False),
            }
        ),
        'rename': (
            rename_mode,
            {
                'input_files': args.input,
                'mapping_file': getattr(args, 'mapping', None),
                'ad_hoc': getattr(args, 'ad_hoc', None),
                'output_file': args.output,
                'min_length': args.min_length,
                'max_length': args.max_length,
                'process_output': False,
                'quiet': args.quiet,
                'clean_items': clean_items,
                'limit': limit,
                'in_place': getattr(args, 'in_place', False),
                'dry_run': getattr(args, 'dry_run', False),
                'smart_case': getattr(args, 'smart_case', False),
            }
        ),
        'standardize': (
            standardize_mode,
            {
                'input_files': args.input,
                'output_file': args.output,
                'min_length': args.min_length,
                'max_length': args.max_length,
                'process_output': False,
                'quiet': args.quiet,
                'clean_items': clean_items,
                'limit': limit,
                'in_place': getattr(args, 'in_place', None),
                'dry_run': getattr(args, 'dry_run', False),
                'fuzzy': getattr(args, 'fuzzy', 0),
                'threshold': getattr(args, 'threshold', 10.0),
                'diff': getattr(args, 'diff', False),
            }
        ),
        'scrub': (
            scrub_mode,
            {
                'input_files': args.input,
                'mapping_file': getattr(args, 'mapping', None),
                'ad_hoc': getattr(args, 'ad_hoc', None),
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
                'diff': getattr(args, 'diff', False),
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
                'keyboard': getattr(args, 'keyboard', False),
                'transposition': getattr(args, 'transposition', False),
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
                'keyboard': getattr(args, 'keyboard', False),
                'transposition': getattr(args, 'transposition', False),
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
                'keyboard': getattr(args, 'keyboard', False),
                'transposition': getattr(args, 'transposition', False),
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
                'keyboard': getattr(args, 'keyboard', False),
                'transposition': getattr(args, 'transposition', False),
                'output_format': output_format,
                'delimiter': delimiter,
                'smart': getattr(args, 'smart', False),
            },
        ),
        'highlight': (
            highlight_mode,
            {
                **common_kwargs,
                'mapping_file': getattr(args, 'mapping', None),
                'ad_hoc': getattr(args, 'ad_hoc', None),
                'smart': getattr(args, 'smart', False),
            }
        ),
        'scan': (
            scan_mode,
            {
                **common_kwargs,
                'mapping_file': getattr(args, 'mapping', None),
                'ad_hoc': getattr(args, 'ad_hoc', None),
                'smart': getattr(args, 'smart', False),
                'line_numbers': getattr(args, 'line_numbers', False),
                'with_filename': getattr(args, 'with_filename', None),
                'before_context': before_context,
                'after_context': after_context,
            }
        ),
        'verify': (
            verify_mode,
            {
                **common_kwargs,
                'mapping_file': getattr(args, 'mapping', None),
                'ad_hoc': getattr(args, 'ad_hoc', None),
                'smart': getattr(args, 'smart', False),
                'prune': getattr(args, 'prune', False),
            }
        ),
    }

    if args.mode == 'help':
        show_mode_help(args.mode_to_help, parser)
        return

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
