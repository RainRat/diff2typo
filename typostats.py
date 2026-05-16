from collections import defaultdict
import json
import sys
import logging
import csv
import io
import os
import re
import time
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

VERSION = "1.1.0"

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

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _YAML_AVAILABLE = False

# Cache for standard input to allow multiple passes
_STDIN_CACHE: List[str] | None = None

# ANSI Color Codes (Internal constants)
_BLUE = "\033[1;34m"
_GREEN = "\033[1;32m"
_RED = "\033[1;31m"
_YELLOW = "\033[1;33m"
_MAGENTA = "\033[1;35m"
_CYAN = "\033[1;36m"
_RESET = "\033[0m"
_BOLD = "\033[1m"

# Global color constants for general use (legacy support)
BLUE = _BLUE
GREEN = _GREEN
RED = _RED
YELLOW = _YELLOW
MAGENTA = _MAGENTA
CYAN = _CYAN
RESET = _RESET
BOLD = _BOLD

# Global color constants are initialized based on stdout.
# Specific functions (like generate_report) may use _should_enable_color
# for more granular stream-based color detection.
if (not sys.stdout.isatty() and 'FORCE_COLOR' not in os.environ) or 'NO_COLOR' in os.environ:
    BLUE = GREEN = RED = YELLOW = MAGENTA = CYAN = RESET = BOLD = ""


class MinimalFormatter(logging.Formatter):
    """A logging formatter that removes prefixes for INFO level messages."""

    LEVEL_COLORS = {
        logging.WARNING: _YELLOW,
        logging.ERROR: _RED,
        logging.CRITICAL: _RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            return record.getMessage()

        levelname = record.levelname
        # Colorize the level name if stderr is a terminal and color is available
        if _should_enable_color(sys.stderr) and levelname:
            color = self.LEVEL_COLORS.get(record.levelno)
            if color:
                levelname = f"{color}{levelname}{_RESET}"

        return f"{levelname}: {record.getMessage()}"


def _should_enable_color(stream: Any) -> bool:
    """Check if color should be enabled for a given stream."""
    if os.environ.get('NO_COLOR'):
        return False
    if os.environ.get('FORCE_COLOR'):
        return True
    return hasattr(stream, 'isatty') and stream.isatty()


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
        'txt': 'arrow',
        'json': 'json',
        'csv': 'csv',
        'yaml': 'yaml',
        'yml': 'yaml',
        'arrow': 'arrow',
    }

    detected = mapping.get(ext)
    if detected in allowed:
        return detected

    return default


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


def _format_analysis_summary(
    raw_count: int,
    filtered_items: Sequence[Any],
    item_label: str = "item",
    start_time: float | None = None,
    use_color: bool = False,
    extra_metrics: Mapping[str, Any] | None = None,
    title: str = "ANALYSIS SUMMARY",
    total_input_items: int | None = None,
) -> List[str]:
    """
    Standardizes the "ANALYSIS SUMMARY" block with consistent colors and a visual retention bar.
    Returns a list of formatted lines.
    """
    item_label_plural = f"{item_label}s"
    c_bold = _BOLD if use_color else ""
    c_blue = _BLUE if use_color else ""
    c_green = _GREEN if use_color else ""
    c_yellow = _YELLOW if use_color else ""
    c_red = _RED if use_color else ""
    c_cyan = _CYAN if use_color else ""
    c_magenta = _MAGENTA if use_color else ""
    c_reset = _RESET if use_color else ""

    padding = "  "
    label_width = 35
    report = []

    report.append(f"\n{padding}{c_bold}{c_blue}{title}{c_reset}")
    report.append(f"{padding}{c_bold}{c_blue}───────────────────────────────────────────────────────{c_reset}")

    if total_input_items is not None:
        report.append(
            f"  {c_bold}{c_blue}{'Total word pairs analyzed:':<{label_width}}{c_reset} {c_yellow}{total_input_items}{c_reset}"
        )

    # In typostats, raw_count is the total number of patterns analyzed, but filtered_items are patterns after filtering.
    raw_label = f"Total {item_label_plural} analyzed:"
    filtered_label = f"Total {item_label_plural} after filtering:"

    report.append(
        f"  {c_bold}{c_blue}{raw_label:<{label_width}}{c_reset} {c_yellow}{raw_count}{c_reset}"
    )

    filtered_count = len(filtered_items)
    report.append(
        f"  {c_bold}{c_blue}{filtered_label:<{label_width}}{c_reset} {c_green}{filtered_count}{c_reset}"
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

    unique_label = "Unique patterns:" if item_label in ("replacement", "pattern") else f"Unique {item_label_plural}:"
    report.append(
        f"  {c_bold}{c_blue}{unique_label:<{label_width}}{c_reset} {c_green}{unique_count}{c_reset}"
    )

    # Shortest/Longest and stats
    if filtered_items:

        def format_item(it: Any) -> str:
            if isinstance(it, tuple) and len(it) == 2:
                return f"{it[1]} -> {it[0]}"
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
        and len(filtered_items[0]) == 2
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


def is_transposition(typo: str, correction: str) -> list[tuple[str, str]]:
    """
    Checks if a mistake was caused by swapping two letters next to each other.

    For example, it shows 'teh' instead of 'the' as a swapped letter mistake.

    Returns:
      A list with the swapped characters if found, otherwise an empty list.
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
    Creates a map of keys that are next to each other on a QWERTY keyboard.

    This is used to find typos caused by a finger slipping to a nearby key.

    Args:
        include_diagonals: Whether to count keys that are diagonal to each other.

    Returns:
        A dictionary where each key points to a set of its neighbors.
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
                if not _YAML_AVAILABLE:
                    logging.error("PyYAML not installed.")
                    continue
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
            except Exception as e:
                logging.error(f"Failed to parse YAML in '{input_file}': {e}")
            continue

        # Text formats
        lines = _read_file_lines_robust(input_file)
        if not quiet:
            iterator = tqdm(lines, desc=f'Processing {input_file}', unit=' lines', leave=False)
        else:
            iterator = lines

        for line in iterator:
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


def is_one_letter_replacement(
    typo: str,
    correction: str,
    allow_1to2: bool = False,
    allow_2to1: bool = False,
    include_deletions: bool = False,
) -> list[tuple[str, str]]:
    """
    Checks if a mistake was caused by changing, adding, or removing letters.

    It can show when one letter was swapped for another, or more complex
    cases like replacing 'm' with 'rn'.

    Returns:
      A list of the found changes. Each change is a pair showing what was
      expected and what was actually typed.
    """

    # Same length scenario: one-to-one replacement
    if len(typo) == len(correction):
        differences = [
            (c_char, t_char)
            for t_char, c_char in zip(typo, correction)
            if t_char != c_char
        ]

        return differences if len(differences) == 1 else []

    # One-to-two replacement scenario allowed only if difference in length is 1
    if (allow_1to2 or include_deletions) and len(typo) == len(correction) + 1:
        # Find the first position i where correction[i] is replaced by typo[i:i+2].
        for i in range(len(correction)):
            # To be a replacement of correction[i] with typo[i:i+2],
            # the prefix correction[:i] must match typo[:i], and
            # the suffix correction[i+1:] must match typo[i+2:].
            if typo[:i] == correction[:i] and typo[i+2:] == correction[i+1:]:
                repl_correction = correction[i]
                repl_typo = typo[i:i+2]

                is_insertion = repl_correction in repl_typo
                if is_insertion:
                    if not include_deletions:
                        continue
                else:  # It's a 1-to-2 replacement
                    if not allow_1to2:
                        continue

                return [(repl_correction, repl_typo)]

    # Two-to-one replacement scenario (for example 'ph' -> 'f')
    if (allow_2to1 or include_deletions) and len(typo) == len(correction) - 1:
        for i in range(len(typo)):
            # To be a replacement of correction[i:i+2] with typo[i],
            # the prefix correction[:i] must match typo[:i], and
            # the suffix correction[i+2:] must match typo[i+1:].
            if correction[:i] == typo[:i] and correction[i+2:] == typo[i+1:]:
                repl_correction = correction[i:i+2]
                repl_typo = typo[i]

                is_deletion = repl_typo in repl_correction
                if is_deletion:
                    if not include_deletions:
                        continue
                else:  # It's a 2-to-1 replacement
                    if not allow_2to1:
                        continue

                return [(repl_correction, repl_typo)]

    return []


def process_typos(
    pairs: Iterable[tuple[str, str]],
    allow_1to2: bool = False,
    allow_2to1: bool = False,
    include_deletions: bool = False,
    allow_transposition: bool = False,
) -> tuple[dict[tuple[str, str], int], int]:
    """
    Finds common mistake patterns in a list of typo corrections.

    This function analyzes typo-correction pairs and shows how letters were replaced.
    It can find simple one-letter mistakes, swapped letters, or cases where multiple
    letters were changed at once.

    Args:
        pairs: An iterable of (typo, correction) pairs.
        allow_1to2: If True, look for one letter replaced by two (like 'm' to 'rn').
        allow_2to1: If True, look for two letters replaced by one (like 'ph' to 'f').
        include_deletions: If True, also count when letters were added or missed.
        allow_transposition: If True, find swapped letters (like 'teh' to 'the').

    Returns:
        A tuple containing:
        - A dictionary of how often each character replacement happened.
        - The total number of typo-correction pairs found.
    """

    replacement_counts = defaultdict(int)
    total_pairs = 0
    for typo, correction in pairs:
        typo = typo.strip()
        correction = correction.strip()

        # Filter out non-ASCII words
        if not all(ord(c) < 128 for c in typo):
            continue
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
    return replacement_counts, total_pairs


def generate_report(
    replacement_counts: dict[tuple[str, str], int],
    output_file: str | None = None,
    min_occurrences: int = 1,
    sort_by: str = 'count',
    output_format: str = 'arrow',
    limit: int | None = None,
    quiet: bool = False,
    keyboard: bool = False,
    total_pairs: int | None = None,
    total_lines: int | None = None,
    start_time: float | None = None,
    **kwargs,
) -> None:
    """
    Creates a summary report of the found typo patterns.

    This function takes the gathered statistics and presents them in a way
    that is easy to read. It can show a visual dashboard with bar charts
    or export the data to formats like JSON and CSV for other tools to use.

    Args:
        replacement_counts: A dictionary of character replacements and their counts.
        output_file: Where to save the report. If not set, it prints to the screen.
        min_occurrences: Only include patterns that happen at least this many times.
        sort_by: How to order the results ('count', 'typo', or 'correct').
        output_format: The style of the report ('arrow', 'yaml', 'json', or 'csv').
        limit: The maximum number of results to show.
        quiet: If True, hide progress bars and status messages.
        keyboard: If True, highlight mistakes caused by hitting nearby keys.
        total_pairs: Total number of corrections analyzed.
        total_lines: Total number of lines read.
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
    show_color_out = not output_file and _should_enable_color(sys.stdout)
    # standard error colors (used for human-readable headers)
    # If output_file is set, we avoid colors in headers that might be written to the file
    show_color_err = not output_file and _should_enable_color(sys.stderr)

    c_bold = _BOLD if show_color_out else ""
    c_blue = _BLUE if show_color_out else ""
    c_green = _GREEN if show_color_out else ""
    c_yellow = _YELLOW if show_color_out else ""
    c_magenta = _MAGENTA if show_color_out else ""
    c_cyan = _CYAN if show_color_out else ""
    c_red = _RED if show_color_out else ""
    c_reset = _RESET if show_color_out else ""

    # Error-specific colors for the summary section
    c_err_cyan = _CYAN if show_color_err else ""
    c_err_magenta = _MAGENTA if show_color_err else ""
    c_err_green = _GREEN if show_color_err else ""
    c_err_red = _RED if show_color_err else ""
    c_err_reset = _RESET if show_color_err else ""

    if output_format == 'arrow':
        # arrow
        padding = "  "

        enabled_features = []
        if keyboard:
            enabled_features.append("keyboard")
        if kwargs.get('allow_transposition'):
            enabled_features.append("transposition")
        if kwargs.get('allow_1to2'):
            enabled_features.append("1-to-2")
        if kwargs.get('allow_2to1'):
            enabled_features.append("2-to-1")
        if kwargs.get('include_deletions'):
            enabled_features.append("deletions/insertions")

        extra_metrics = {}
        if total_lines is not None:
            extra_metrics["Total lines processed"] = total_lines
        if enabled_features:
            extra_metrics["Enabled features"] = ", ".join(enabled_features)

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
                extra_metrics["Keyboard Adjacency [K]"] = f"{c_err_cyan}{adjacent_count}/{total_single_char} ({percent:.1f}%){c_err_reset}"

        if kwargs.get('allow_transposition'):
            trans_count = 0
            for (c, t), count in replacement_counts.items():
                if len(c) == 2 and len(t) == 2 and c == t[::-1]:
                    trans_count += count
            if trans_count > 0:
                percent = (trans_count / total_typos) * 100 if total_typos > 0 else 0
                extra_metrics["Transpositions [T]"] = f"{c_err_magenta}{trans_count}/{total_typos} ({percent:.1f}%){c_err_reset}"

        if any([kwargs.get('allow_1to2'), kwargs.get('allow_2to1'), kwargs.get('include_deletions')]):
            counts = defaultdict(int)
            color_map = {
                "Insertions [Ins]": c_err_green,
                "1-to-2 replacements [1:2]": c_err_green,
                "Deletions [Del]": c_err_red,
                "2-to-1 replacements [2:1]": c_err_red,
            }
            for (c, t), count in replacement_counts.items():
                if len(c) < len(t):
                    if c in t:
                        counts["Insertions [Ins]"] += count
                    else:
                        counts["1-to-2 replacements [1:2]"] += count
                elif len(c) > len(t):
                    if t in c:
                        counts["Deletions [Del]"] += count
                    else:
                        counts["2-to-1 replacements [2:1]"] += count

            for label, count in counts.items():
                if count > 0:
                    percent = (count / total_typos) * 100 if total_typos > 0 else 0
                    color = color_map.get(label, "")
                    extra_metrics[label] = f"{color}{count}/{total_typos} ({percent:.1f}%){c_err_reset}"

        if unique_filtered != unique_total:
            extra_metrics["Patterns matching criteria"] = unique_filtered

        if unique_filtered > len(sorted_replacements):
            extra_metrics["Showing patterns"] = f"{len(sorted_replacements)} of {unique_filtered}"

        # Generate a list of all replacements to use summary statistics
        all_replacements = [k for k, v in filtered.items() for _ in range(v)]
        summary_lines = _format_analysis_summary(
            total_typos,
            all_replacements,
            item_label="pattern",
            use_color=show_color_err,
            extra_metrics=extra_metrics,
            start_time=start_time,
            total_input_items=total_pairs,
        )

        # Calculate padding for alignment (default to header labels' lengths)
        max_c = max((len(c) for (c, t), count in sorted_replacements), default=10)
        max_c = max(max_c, 10)  # 'Correction' is 10
        max_t = max((len(t) for (c, t), count in sorted_replacements), default=4)
        max_t = max(max_t, 4)  # 'Typo' is 4
        max_n = max((len(str(count)) for (c, t), count in sorted_replacements), default=5)
        max_n = max(max_n, 5)  # 'Count' is 5
        max_p = 6  # Width for percentage (for example, "100.0%")

        # Header row and divider with consistent padding and vertical separators
        # Bold blue for table visual elements
        sep = f"{c_bold}{c_blue}│{c_reset}"
        header_row = (
            f"{padding}{c_bold}{c_blue}{'Typo':<{max_t}}{c_reset} {sep} "
            f"{c_bold}{c_blue}{'Correction':<{max_c}}{c_reset} {sep} "
            f"{c_bold}{c_blue}{'Count':>{max_n}}{c_reset} {sep} "
            f"{c_bold}{c_blue}{'%':>{max_p}}{c_reset}"
        )
        visible_header_len = max_t + max_c + max_n + max_p + 9

        show_attr = any([keyboard, kwargs.get('allow_transposition'), kwargs.get('allow_1to2'), kwargs.get('allow_2to1'), kwargs.get('include_deletions'), kwargs.get('all')])

        if show_attr:
            header_row += f" {sep} {c_bold}{c_blue}{'Attr':<5}{c_reset}"
            visible_header_len += 8

        # Add Visual column header
        max_bar = 20
        header_row += f" {sep} {c_bold}{c_blue}{'Visual':<{max_bar}}{c_reset}"
        visible_header_len += 3 + max_bar

        divider = f"{padding}{c_bold}{c_blue}{'─' * visible_header_len}{c_reset}"

        if not output_file:
            # Move the human-readable header to standard error to keep the main output clean for piping
            if not quiet:
                sys.stderr.write("\n".join(summary_lines) + "\n")
                if sorted_replacements:
                    # Frequency table header
                    table_title = f"{padding}{c_bold}{c_blue}LETTER REPLACEMENTS{c_reset}"
                    sys.stderr.write(f"\n{table_title}\n")
                    sys.stderr.write(f"{padding}{c_bold}{c_blue}{'─' * visible_header_len}{c_reset}\n")
                    sys.stderr.write(f"{header_row}\n")
                    sys.stderr.write(f"{divider}\n")
                sys.stderr.flush()
            report_lines = []
        else:
            report_lines = summary_lines[:]
            if sorted_replacements:
                table_title = f"{padding}LETTER REPLACEMENTS"
                report_lines.extend(["", table_title, f"{padding}{'─' * visible_header_len}", header_row, divider])

        if not sorted_replacements:
            no_results = f"{padding}{c_yellow}No replacements found matching the criteria.{c_reset}"
            if not output_file:
                if not quiet:
                    sys.stderr.write(f"{no_results}\n")
            else:
                report_lines.append(no_results)

        for (correct_char, typo_char), count in sorted_replacements:
            percent = (count / total_typos * 100) if total_typos > 0 else 0

            # Determine error type and color
            marker_text = "     "
            marker_color = c_yellow
            if len(correct_char) == 1 and len(typo_char) == 1:
                if keyboard and typo_char.lower() in adjacent_map.get(correct_char.lower(), set()):
                    marker_text = "[K]"
                    marker_color = c_cyan
            elif len(correct_char) == 2 and len(typo_char) == 2 and correct_char == typo_char[::-1]:
                marker_text = "[T]"
                marker_color = c_magenta
            elif len(correct_char) < len(typo_char):
                # Typo is longer: Insertion [Ins] or 1-to-2 replacement [1:2]
                marker_color = c_green
                if correct_char in typo_char:
                    marker_text = "[Ins]"
                else:
                    marker_text = "[1:2]"
            elif len(correct_char) > len(typo_char):
                # Typo is shorter: Deletion [Del] or 2-to-1 replacement [2:1]
                marker_color = c_red
                if typo_char in correct_char:
                    marker_text = "[Del]"
                else:
                    marker_text = "[2:1]"

            if marker_text == "     ":
                # If no specific classification, use neutral blue for the bar
                marker_color_for_bar = c_blue
            else:
                marker_color_for_bar = marker_color

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
                f"{padding}{c_red}{typo_char:<{max_t}}{c_reset} {sep} "
                f"{c_green}{correct_char:<{max_c}}{c_reset} {sep} "
                f"{c_yellow}{count:>{max_n}}{c_reset} {sep} "
                f"{c_green}{percent:>5.1f}%{c_reset}"
            )
            if show_attr:
                marker = f"{marker_color}{marker_text:<5}{c_reset}"
                row += f" {sep} {marker}"

            row += f" {sep} {marker_color_for_bar}{bar}{c_reset}"
            report_lines.append(row)
        report_content = "\n".join(report_lines)
    elif output_format == 'json':
        adjacent_map = {}
        if keyboard:
            adjacent_map = get_adjacent_keys(include_diagonals=True)

        replacements = []
        for (correct_char, typo_char), count in sorted_replacements:
            item = {
                "typo": typo_char,
                "correct": correct_char,
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
        writer.writerow(['typo', 'correction', 'count'])
        for (correct_char, typo_char), count in sorted_replacements:
            writer.writerow([typo_char, correct_char, count])
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
        # Standardize standard output writing to ensure no duplicate newlines
        if report_content:
            sys.stdout.write(report_content)
            if not report_content.endswith('\n'):
                sys.stdout.write('\n')


def detect_encoding(file_path: str) -> str | None:
    """
    Tries to figure out the text encoding of a file.
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




def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=f"{BOLD}Find common patterns in your typos. This tool analyzes typo corrections and tells you which keys you hit by mistake most often.{RESET}\n\n"
                    f"It supports multiple input formats including standard typo lists (arrow, table, colon, CSV),\n"
                    f"JSON/YAML mapping files, and Markdown lists or tables.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"""{BLUE}Examples:{RESET}
  {GREEN}python typostats.py typos.txt -t{RESET}          # Find swapped letters (like 'teh' -> 'the')
  {GREEN}python typostats.py typos.txt --1to2 --2to1{RESET}  # Find multi-letter mistakes (like 'rn' -> 'm')
  {GREEN}python typostats.py typos.txt -k -n 20{RESET}       # Find top 20 keyboard slips
  {GREEN}python typostats.py typos.txt -a{RESET}             # Run all analysis modes at once
""",
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {VERSION}'
    )

    # Input/Output Group
    io_group = parser.add_argument_group(f"{BLUE}INPUT/OUTPUT OPTIONS{RESET}")
    io_group.add_argument(
        'input_files',
        nargs='*',
        help="One or more files containing typo corrections (txt, csv, json, yaml, md). If empty, it reads from standard input.",
    )
    io_group.add_argument('-o', '--output', help="Save the report to this file instead of printing it.")
    io_group.add_argument(
        '-f',
        '--format',
        choices=['arrow', 'yaml', 'json', 'csv'],
        metavar='FMT',
        default=None,
        help="The format of the report. If not provided, it is automatically detected from the output file extension. (default: arrow).",
    )
    io_group.add_argument('-q', '--quiet', action='store_true', help="Suppress informational log output.")

    # Analysis Options Group
    analysis_group = parser.add_argument_group(f"{BLUE}ANALYSIS OPTIONS{RESET}")
    analysis_group.add_argument('-m', '--min', type=int, default=1, help="Only show patterns that happen at least this many times.")
    analysis_group.add_argument(
        '-s', '--sort',
        choices=['count', 'typo', 'correct'],
        default='count',
        help="How to sort the results: 'count' (most frequent first), 'typo' (alphabetical by the mistake), or 'correct' (alphabetical by the fix)."
    )
    analysis_group.add_argument(
        '-a',
        '--all',
        action='store_true',
        help="Use all analysis features (swapped letters, keys next to each other, and multi-letter changes). This is the default if no other options are picked.",
    )
    analysis_group.add_argument(
        '-2',
        '--allow-two-char',
        dest='allow_two_char',
        action='store_true',
        help="Allow cases where one letter is replaced by two (like 'm' to 'rn') or two letters are replaced by one (like 'ph' to 'f').",
    )
    # Hidden alias for backward compatibility
    parser.add_argument('--allow_two_char', action='store_true', help=argparse.SUPPRESS)

    analysis_group.add_argument(
        '--1to2',
        dest='allow_1to2',
        action='store_true',
        help="Allow cases where one letter is replaced by two (like 'm' to 'rn').",
    )
    analysis_group.add_argument(
        '--2to1',
        dest='allow_2to1',
        action='store_true',
        help="Allow cases where two letters are replaced by one (like 'ph' to 'f').",
    )
    analysis_group.add_argument(
        '--include-deletions',
        action='store_true',
        help="Include cases where you added an extra letter or missed one (like 'aa' to 'a' or 'o' to 'or').",
    )

    analysis_group.add_argument(
        '-t',
        '--transposition',
        action='store_true',
        help="Find cases where you swapped two letters next to each other (like 'teh' instead of 'the').",
    )
    analysis_group.add_argument(
        '-k',
        '--keyboard',
        action='store_true',
        help="Find cases where you hit a key next to the correct one on your keyboard (like 'p' instead of 'o').",
    )
    analysis_group.add_argument(
        '-n',
        '-L',
        '--limit',
        type=int,
        help="Only show the top N results in the report.",
    )
    args = parser.parse_args()

    # If no analysis flags are provided, enable all of them by default
    analysis_flags = [
        'allow_1to2', 'allow_2to1', 'include_deletions',
        'transposition', 'keyboard', 'all', 'allow_two_char'
    ]
    if not any(getattr(args, flag) for flag in analysis_flags):
        args.all = True

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
    if output_format is None:
        allowed_formats = ['arrow', 'yaml', 'json', 'csv']
        output_format = _detect_format_from_extension(output_file, allowed_formats, 'arrow')
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

    start_time = time.perf_counter()
    all_counts = defaultdict(int)
    total_lines_all = 0
    total_pairs_all = 0

    for file_path in input_files:
        # Pre-calculate total lines for statistics and progress tracking
        try:
            if file_path == '-':
                lines = _read_file_lines_robust('-')
                total_lines_all += len(lines)
            else:
                with open(file_path, 'rb') as f:
                    total_lines_all += sum(1 for _ in f)
        except Exception:
            pass

        pairs = _extract_pairs([file_path], quiet=args.quiet)

        file_counts, pairs_count = process_typos(
            pairs,
            allow_1to2=allow_1to2,
            allow_2to1=allow_2to1,
            include_deletions=include_deletions,
            allow_transposition=allow_transposition,
        )
        for k, v in file_counts.items():
            all_counts[k] += v
        total_pairs_all += pairs_count

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
        total_pairs=total_pairs_all,
        total_lines=total_lines_all,
        start_time=start_time,
    )


if __name__ == "__main__":
    main()
