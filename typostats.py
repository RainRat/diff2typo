from collections import defaultdict
import json
import sys
import logging
import csv
import io
import os
import time
from typing import Any, Iterable, List, Mapping, Sequence

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
if (not sys.stdout.isatty() and not os.environ.get('FORCE_COLOR')) or os.environ.get('NO_COLOR'):
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

    # In typostats, raw_count is the number of word pairs, but filtered_items are patterns.
    # We rename labels to be more descriptive of what they actually count.
    raw_label = "Total word pairs encountered:"
    filtered_label = "Total patterns after analysis:"
    if item_label != "replacement":
        raw_label = f"Total {item_label_plural} encountered:"
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

    unique_label = "Unique patterns found:" if item_label == "replacement" else f"Unique {item_label_plural}:"
    report.append(
        f"  {c_bold}{c_blue}{unique_label:<{label_width}}{c_reset} {c_green}{unique_count}{c_reset}"
    )

    # Shortest/Longest and stats
    if filtered_items:

        def format_item(it: Any) -> str:
            if isinstance(it, tuple) and len(it) == 2:
                return f"{it[0]} -> {it[1]}"
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
    if allow_1to2 and len(typo) == len(correction) + 1:
        # Find all positions i where correction[i] is replaced by typo[i:i+2].
        # We use a set to avoid counting identical interpretations (for example for doubled letters) multiple times.
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

    # Two-to-one replacement scenario (for example 'ph' -> 'f')
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


def process_typos(
    lines: Iterable[str],
    allow_1to2: bool = False,
    allow_2to1: bool = False,
    include_deletions: bool = False,
    allow_transposition: bool = False,
) -> tuple[dict[tuple[str, str], int], int, int]:
    """
    Finds common mistake patterns in a list of typo corrections.

    This function reads through your typo list and shows how letters were replaced.
    It can find simple one-letter mistakes, swapped letters, or cases where multiple
    letters were changed at once.

    Args:
        lines: The lines of text containing your typo corrections.
        allow_1to2: If True, look for one letter replaced by two (like 'm' to 'rn').
        allow_2to1: If True, look for two letters replaced by one (like 'ph' to 'f').
        include_deletions: If True, also count when letters were added or missed.
        allow_transposition: If True, find swapped letters (like 'teh' to 'the').

    Returns:
        A tuple containing:
        - A dictionary of how often each character replacement happened.
        - The total number of lines processed.
        - The total number of typo-correction pairs found.
    """

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
    show_color_out = not output_file and sys.stdout.isatty()
    # standard error colors (used for human-readable headers)
    # If output_file is set, we avoid colors in headers that might be written to the file
    show_color_err = not output_file and sys.stderr.isatty()

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
                extra_metrics["Keyboard Adjacency [K]"] = f"{adjacent_count}/{total_single_char} ({percent:.1f}%)"

        if kwargs.get('allow_transposition'):
            trans_count = 0
            for (c, t), count in replacement_counts.items():
                if len(c) == 2 and len(t) == 2 and c == t[::-1]:
                    trans_count += count
            if trans_count > 0:
                percent = (trans_count / total_typos) * 100 if total_typos > 0 else 0
                extra_metrics["Transpositions [T]"] = f"{trans_count}/{total_typos} ({percent:.1f}%)"

        if any([kwargs.get('allow_1to2'), kwargs.get('allow_2to1'), kwargs.get('include_deletions')]):
            counts = defaultdict(int)
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
                    extra_metrics[label] = f"{count}/{total_typos} ({percent:.1f}%)"

        if unique_filtered != unique_total:
            extra_metrics["Patterns matching criteria"] = unique_filtered

        if unique_filtered > len(sorted_replacements):
            extra_metrics["Showing patterns"] = f"{len(sorted_replacements)} of {unique_filtered}"

        # Generate a list of all replacements to use summary statistics
        all_replacements = [k for k, v in replacement_counts.items() for _ in range(v)]
        summary_lines = _format_analysis_summary(
            total_pairs if total_pairs is not None else total_typos,
            all_replacements,
            item_label="replacement",
            use_color=show_color_err,
            extra_metrics=extra_metrics,
            start_time=start_time,
        )

        # Calculate padding for alignment (default to header labels' lengths)
        max_c = max((len(c) for (c, t), count in sorted_replacements), default=7)
        max_c = max(max_c, 7)  # 'CORRECT' is 7
        max_t = max((len(t) for (c, t), count in sorted_replacements), default=4)
        max_t = max(max_t, 4)  # 'TYPO' is 4
        max_n = max((len(str(count)) for (c, t), count in sorted_replacements), default=5)
        max_n = max(max_n, 5)  # 'COUNT' is 5
        max_p = 6  # Width for percentage (for example, "100.0%")

        # Colors for table
        c_bold = BOLD if show_color_out else ""
        c_blue = BLUE if show_color_out else ""
        c_green = GREEN if show_color_out else ""
        c_yellow = YELLOW if show_color_out else ""
        c_red = RED if show_color_out else ""
        c_reset = RESET if show_color_out else ""

        # Header row and divider with consistent padding and vertical separators
        # Bold blue for table visual elements
        sep = f"{c_bold}{c_blue}│{c_reset}"
        header_row = (
            f"{padding}{c_bold}{c_blue}{'CORRECT':>{max_c}}{c_reset} {sep} "
            f"{c_bold}{c_blue}{'TYPO':<{max_t}}{c_reset} {sep} "
            f"{c_bold}{c_blue}{'COUNT':>{max_n}}{c_reset} {sep} "
            f"{c_bold}{c_blue}{'%':>{max_p}}{c_reset}"
        )
        # 3 chars for each " │ " (total 3 * 3 = 9)
        visible_header_len = max_c + max_t + max_n + max_p + 9

        show_attr = any([keyboard, kwargs.get('allow_transposition'), kwargs.get('allow_1to2'), kwargs.get('allow_2to1'), kwargs.get('include_deletions'), kwargs.get('all')])

        if show_attr:
            header_row += f" {sep} {c_bold}{c_blue}{'ATTR':<4}{c_reset}"
            visible_header_len += 7

        # Add Visual column header
        max_bar = 15
        header_row += f" {sep} {c_bold}{c_blue}{'VISUAL':<{max_bar}}{c_reset}"
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
                    sys.stderr.write(f"{padding}{c_bold}{c_blue}───────────────────────────────────────────────────────{c_reset}\n")
                    sys.stderr.write(f"{header_row}\n")
                    sys.stderr.write(f"{divider}\n")
                sys.stderr.flush()
            report_lines = []
        else:
            report_lines = summary_lines[:]
            if sorted_replacements:
                table_title = f"{padding}LETTER REPLACEMENTS"
                report_lines.extend(["", table_title, f"{padding}───────────────────────────────────────────────────────", header_row, divider])

        if not sorted_replacements:
            no_results = f"{padding}{c_yellow}No replacements found matching the criteria.{c_reset}"
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
                f"{padding}{c_green}{correct_char:>{max_c}}{c_reset} {sep} "
                f"{c_red}{typo_char:<{max_t}}{c_reset} {sep} "
                f"{c_yellow}{count:>{max_n}}{c_reset} {sep} "
                f"{c_green}{percent:>5.1f}%{c_reset}"
            )

            if show_attr:
                marker_text = "     "
                if len(correct_char) == 1 and len(typo_char) == 1:
                    if keyboard and typo_char.lower() in adjacent_map.get(correct_char.lower(), set()):
                        marker_text = "[K]"
                elif len(correct_char) == 2 and len(typo_char) == 2 and correct_char == typo_char[::-1]:
                    marker_text = "[T]"
                elif len(correct_char) < len(typo_char):
                    # Typo is longer: Insertion [Ins] or 1-to-2 replacement [1:2]
                    if correct_char in typo_char:
                        marker_text = "[Ins]"
                    else:
                        marker_text = "[1:2]"
                elif len(correct_char) > len(typo_char):
                    # Typo is shorter: Deletion [Del] or 2-to-1 replacement [2:1]
                    if typo_char in correct_char:
                        marker_text = "[Del]"
                    else:
                        marker_text = "[2:1]"
                elif len(correct_char) != len(typo_char):
                    # Fallback for any other multi-letter case
                    marker_text = "[M]"

                marker = f"{c_bold}{marker_text:<5}{c_reset}"
                row += f" {sep} {marker}"

            row += f" {sep} {c_red}{bar}{c_reset}"
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


def load_lines_from_file(file_path: str) -> list[str] | None:
    """
    Reads all lines from a file and handles different text encodings.

    If the file is not in standard UTF-8 format, it tries to detect the
    correct encoding or falls back to a simpler format to prevent crashes.
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
  {GREEN}python typostats.py typos.txt -t{RESET}          # Find swapped letters (like 'teh' -> 'the')
  {GREEN}python typostats.py typos.txt --1to2 --2to1{RESET}  # Find multi-letter mistakes (like 'rn' -> 'm')
  {GREEN}python typostats.py typos.txt -k -n 20{RESET}       # Find top 20 keyboard slips
  {GREEN}python typostats.py typos.txt -a{RESET}             # Run all analysis modes at once
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
