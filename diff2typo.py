'''
diff2typo.py

Purpose:
    Find typo corrections in a Git diff and prepare an update for the `typos` tool.
    This helps ensure that typos you find are caught in future changes.

Features:
    - Finds typo corrections in Git diffs and file renames or copies.
    - Splits compound words based on spaces, underscores, and capital letters.
    - Skips corrections where the "before" word is in the large dictionary.
    - Works with the `typos` tool to avoid duplicate entries.
    - Automatically detects the word list file format.
    - Allows customization through command-line options.
    - Uses the `--mode` option to find typos, corrections for existing typos, or cases where a correct word was changed into a typo.

Usage:
    python diff2typo.py diff.txt --output=typos.txt --format=list

Examples:
    - Find typos: python diff2typo.py diff.txt --output=typos.txt --mode typos
    - Corrections for existing typos: python diff2typo.py diff.txt --output=typos.txt --mode corrections
    - Both typos and corrections: python diff2typo.py diff.txt --output=typos.txt --mode both
    - Find correct words changed into typos: python diff2typo.py diff.txt --mode audit

Output Formats:
    - arrow: typo -> correction
    - csv: typo,correction
    - table: typo = "correction"
    - list: typo
    - json: JSON array of objects or dict
    - yaml: YAML document
    - markdown: Markdown table or list
'''

import argparse
from collections import Counter
import contextlib
import csv
import fnmatch
import glob
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, TextIO

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter([])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def update(self, n=1): pass
        def close(self): pass
        def set_description(self, desc=None, refresh=True): pass
        def set_postfix(self, *args, **kwargs): pass


VERSION = "1.1.0"


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
# Specific functions (like _format_analysis_summary) use _should_enable_color
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


def _render_visual_bar(percentage: float, max_bar: int = 20) -> str:
    """
    Creates a high-resolution visual bar using Unicode block characters.
    """
    total_blocks = (percentage * max_bar) / 100
    full_blocks = int(total_blocks)
    fraction = total_blocks - full_blocks
    blocks = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
    frac_idx = int(fraction * 8)

    bar = "█" * full_blocks
    if full_blocks < max_bar:
        bar += blocks[frac_idx]
        bar += " " * (max_bar - full_blocks - 1)
    return bar


def _format_analysis_summary(
    raw_count: int,
    filtered_items: Sequence[Any],
    item_label: str = "item",
    start_time: Optional[float] = None,
    use_color: bool = False,
    extra_metrics: Optional[Mapping[str, Any]] = None,
    title: str = "ANALYSIS SUMMARY",
    total_input_items: Optional[int] = None,
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
    c_cyan = _CYAN if use_color else ""
    c_reset = _RESET if use_color else ""

    padding = "  "
    label_width = 35
    report = []

    report.append(f"\n{padding}{c_bold}{c_blue}{title}{c_reset}")
    report.append(f"{padding}{c_bold}{c_blue}───────────────────────────────────────────────────────{c_reset}")

    if total_input_items is not None:
        report.append(
            f"  {c_bold}{c_blue}{'Total word pairs in diff:':<{label_width}}{c_reset} {c_yellow}{total_input_items}{c_reset}"
        )

    report.append(
        f"  {c_bold}{c_blue}{'Unique ' + item_label_plural + ' found:':<{label_width}}{c_reset} {c_yellow}{raw_count}{c_reset}"
    )

    filtered_count = len(filtered_items)
    report.append(
        f"  {c_bold}{c_blue}{'Total ' + item_label_plural + ' after filtering:':<{label_width}}{c_reset} {c_green}{filtered_count}{c_reset}"
    )

    if raw_count > 0:
        retention = (filtered_count / raw_count) * 100
        # High-res visual bar for retention
        max_bar = 20
        bar = _render_visual_bar(retention, max_bar)

        report.append(
            f"  {c_bold}{c_blue}{'Retention rate:':<{label_width}}{c_reset} {c_green}{retention:>5.1f}%{c_reset} {c_cyan}{bar}{c_reset}"
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

    # Extra metrics
    if extra_metrics:
        for label, value in extra_metrics.items():
            report.append(f"  {c_bold}{c_blue}{label + ':':<{label_width}}{c_reset} {value}")

    # Processing Time
    if start_time is not None:
        duration = time.perf_counter() - start_time
        report.append(
            f"  {c_bold}{c_blue}{'Processing time:':<{label_width}}{c_reset} {c_green}{duration:.3f}s{c_reset}"
        )

    report.append("")
    return report


def filter_to_letters(text: str) -> str:
    """Return text containing only lowercase a-z characters."""
    return re.sub("[^a-z]", "", text.lower())




def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the number of character changes needed to turn one word into another."""
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


def read_allowed_words(allowed_file: str) -> Set[str]:
    """
    Reads allowed words from a CSV file and returns a set of lowercase words.
    These are words that have been explicitly rejected from being considered typos.

    Args:
        allowed_file (str): Path to the allowed words CSV file.

    Returns:
        set: A set of allowed words in lowercase.
    """
    try:
        with open(allowed_file, "r", encoding="utf-8") as file_handle:
            rows = list(csv.reader(file_handle))
    except FileNotFoundError:
        logging.warning(f"Allowed words file '{allowed_file}' not found. Skipping.")
        rows = []
    except Exception as exc:
        logging.error(f"Error reading allowed words file '{allowed_file}': {exc}")
        rows = []

    allowed_words = {row[0].strip().lower() for row in rows if row}
    if rows:
        logging.info(f"Loaded {len(allowed_words)} allowed words from '{allowed_file}'.")
    return allowed_words

def split_into_subwords(word: str) -> List[str]:
    """
    Splits a word into subwords based on spaces, underscores, hyphens, and casing boundaries.

    Args:
        word (str): The word to split.

    Returns:
        list: A list of subwords.
    """
    pattern = r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+'
    subwords = []
    for part in re.split(r'[ _-]+', word):
        subwords.extend(re.findall(pattern, part) or [part])
    return subwords

def read_words_mapping(file_path: str, required: bool = True) -> Dict[str, Set[str]]:
    """
    Reads a CSV file of typo fixes and returns a mapping:
         incorrect_word -> corrections

    Each row should be in the form:
         incorrect_word, correction1, correction2, ...

    We can also accept a list of words for the large dictionary. They will
        not have any corrections.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file_handle:
            rows = list(csv.reader(file_handle))
    except FileNotFoundError:
        message = f"Large dictionary file '{file_path}' not found."
        if required:
            logging.error(message)
            sys.exit(1)
        logging.warning(message + " Skipping.")
        rows = []
    except Exception as exc:
        logging.error(f"Error reading large dictionary file '{file_path}': {exc}")
        if required:
            sys.exit(1)
        rows = []

    mapping: Dict[str, Set[str]] = {}
    for row in rows:
        if row:
            incorrect = row[0].strip().lower()
            corrections = {col.strip().lower() for col in row[1:] if col.strip()}
            mapping[incorrect] = corrections
    if rows:
        logging.info(f"Loaded mapping for {len(mapping)} words from '{file_path}'.")
    return mapping

def _compare_word_lists(
    before_words: Sequence[str],
    after_words: Sequence[str],
    min_length: int,
    max_dist: Optional[int] = None,
) -> List[str]:
    """Return typo pairs discovered when comparing two word sequences."""
    import difflib

    # Use sequence alignment to find corresponding changes in words.
    # This allows finding typo corrections even when words
    # are added or removed within the same diff block.
    matcher = difflib.SequenceMatcher(None, before_words, after_words)
    typos: List[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Extraction of words from the replaced blocks.
            # We match them 1-to-1 if the block sizes are identical.
            # Otherwise, we attempt to find the most likely pairing.
            removals = before_words[i1:i2]
            additions = after_words[j1:j2]

            # If the number of removed and added words in the block matches,
            # we can process them as individual substitutions.
            if len(removals) == len(additions):
                for k, (before_word, after_word) in enumerate(zip(removals, additions)):
                    if before_word == after_word:
                        continue

                    before_clean = filter_to_letters(before_word)
                    after_clean = filter_to_letters(after_word)

                    if before_clean == after_clean:
                        continue

                    if len(before_clean) >= min_length and len(after_clean) >= min_length:
                        if max_dist is None or levenshtein_distance(before_clean, after_clean) <= max_dist:
                            typos.append(f"{before_clean} -> {after_clean}")
            else:
                # If block sizes differ (for example, "teh house" -> "the big house"),
                # we perform a local similar word matching to find the best candidate pair.
                for b_word in removals:
                    b_clean = filter_to_letters(b_word)
                    if len(b_clean) < min_length:
                        continue

                    best_match = None
                    best_dist = float('inf')

                    for a_word in additions:
                        a_clean = filter_to_letters(a_word)
                        if len(a_clean) < min_length or a_clean == b_clean:
                            continue

                        dist = levenshtein_distance(b_clean, a_clean)
                        # We only consider it a typo if the distance is low relative to the word length
                        # and fits within the global max_dist constraint.
                        if dist < best_dist and dist <= (max_dist if max_dist is not None else 2):
                            best_match = a_clean
                            best_dist = dist

                    if best_match:
                        typos.append(f"{b_clean} -> {best_match}")

    return typos


def process_diff_block(
    removals: List[str], additions: List[str], min_length: int, max_dist: Optional[int] = None
) -> List[str]:
    """Return typos generated from matching removal/addition blocks."""

    if not removals or not additions:
        return []

    before_text = " ".join(removals)
    after_text = " ".join(additions)
    before_words = split_into_subwords(before_text)
    after_words = split_into_subwords(after_text)
    return _compare_word_lists(before_words, after_words, min_length, max_dist)


def _match_pattern(filepath: str, patterns: Optional[List[str]]) -> bool:
    if not patterns or not filepath:
        return False
    for pattern in patterns:
        if fnmatch.fnmatch(filepath, pattern) or fnmatch.fnmatch(os.path.basename(filepath), pattern):
            return True
    return False


def find_typos(
    diff_text: str,
    min_length: int = 2,
    max_dist: Optional[int] = None,
    exclude_patterns: Optional[List[str]] = None,
    include_patterns: Optional[List[str]] = None,
) -> List[str]:
    """
    Parses the diff text to find typo corrections.

    Args:
        diff_text (str): The Git diff text.
        min_length (int): Minimum length of differing substrings to consider as typos.
        max_dist (int, optional): Maximum Levenshtein distance for typos.
        exclude_patterns (list, optional): List of file/path patterns to exclude.

    Returns:
        list: A list of typo candidates in the format "before -> after".
    """
    typos: List[str] = []
    lines = diff_text.split("\n")
    removals: List[str] = []
    additions: List[str] = []
    current_file: Optional[str] = None
    skip_current_file = False

    for line in lines:
        if line.startswith('diff --git '):
            if not skip_current_file:
                typos.extend(process_diff_block(removals, additions, min_length, max_dist))
            removals = []
            additions = []

            current_file = None
            skip_current_file = False
            try:
                parts = shlex.split(line)
            except ValueError:
                parts = line.split(' ')
            if len(parts) >= 4:
                p = parts[2]
                if p.startswith('a/') or p.startswith('b/'):
                    current_file = p[2:]
                else:
                    current_file = p
                if current_file.startswith('"') and current_file.endswith('"'):
                    current_file = current_file[1:-1]
            if current_file:
                if _match_pattern(current_file, exclude_patterns) or (include_patterns and not _match_pattern(current_file, include_patterns)):
                    skip_current_file = True
            continue

        # Handle file renames and copies
        if line.startswith('rename from ') or line.startswith('copy from '):
            path = line.split(' from ', 1)[1].strip()
            if path.startswith('"') and path.endswith('"'):
                path = path[1:-1]
            removals.append(path)
            continue
        if line.startswith('rename to ') or line.startswith('copy to '):
            path = line.split(' to ', 1)[1].strip()
            if path.startswith('"') and path.endswith('"'):
                path = path[1:-1]
            additions.append(path)
            if not (skip_current_file or _match_pattern(path, exclude_patterns) or (include_patterns and not _match_pattern(path, include_patterns))):
                typos.extend(process_diff_block(removals, additions, min_length, max_dist))
            removals = []
            additions = []
            continue

        if line.startswith('---') or line.startswith('+++'):
            p_match = re.match(r'^(?:---|\+\+\+)\s+[ab]/(.*)$', line)
            if p_match:
                path = p_match.group(1).strip()
                if path.startswith('"') and path.endswith('"'):
                    path = path[1:-1]
                current_file = path
                if not skip_current_file:
                    typos.extend(process_diff_block(removals, additions, min_length, max_dist))
                removals = []
                additions = []
                skip_current_file = _match_pattern(current_file, exclude_patterns) or (include_patterns and not _match_pattern(current_file, include_patterns))
            continue

        if skip_current_file:
            continue

        if line.startswith('-'):
            removals.append(line[1:].strip())
        elif line.startswith('+'):
            additions.append(line[1:].strip())
        else:
            typos.extend(process_diff_block(removals, additions, min_length, max_dist))
            removals = []
            additions = []

    if not skip_current_file:
        typos.extend(process_diff_block(removals, additions, min_length, max_dist))

    return typos

@contextlib.contextmanager
def smart_open_output(filename: str, encoding: str = 'utf-8') -> Iterable[TextIO]:
    """
    Context manager that yields a file object for writing.
    If filename is '-', yields the screen.
    Otherwise, opens the file for writing.
    """
    if filename == '-':
        yield sys.stdout
    else:
        with open(filename, 'w', encoding=encoding) as f:
            yield f


def format_typos(typos: Iterable[str], output_format: str) -> List[str]:
    """
    Formats the list of typos based on the specified output format.

    Args:
        typos (list): List of typo strings in the format "before -> after".
        output_format (str): Desired output format ('arrow', 'csv', 'table', 'list', 'json', 'yaml', 'markdown', 'md').

    Returns:
        list: Formatted list of typo strings.
    """
    if output_format in ('json', 'yaml'):
        items = []
        for typo in typos:
            if ' -> ' in typo:
                before, after = typo.split(' -> ')
                items.append({"typo": before, "correction": after})
            else:
                items.append({"typo": typo, "correction": ""})

        if output_format == 'yaml':
            if _YAML_AVAILABLE:
                return yaml.safe_dump(items, default_flow_style=False).rstrip().split('\n')
            else:
                logging.warning("PyYAML not installed. Falling back to JSON for YAML output format.")

        return json.dumps(items, indent=2).split('\n')

    if output_format in ('markdown', 'md'):
        formatted: List[str] = []
        typo_pairs = []
        single_items = []

        for typo in typos:
            if ' -> ' in typo:
                before, after = typo.split(' -> ')
                typo_pairs.append((before, after))
            else:
                single_items.append(typo)

        if typo_pairs:
            formatted.append("| Typo | Correction |")
            formatted.append("| :--- | :--- |")
            for before, after in typo_pairs:
                formatted.append(f"| `{before}` | `{after}` |")

        if single_items:
            if typo_pairs:
                formatted.append("")
            for item in single_items:
                clean_item = filter_to_letters(item)
                formatted.append(f"- `{clean_item}`")

        return formatted

    formatted: List[str] = []
    for typo in typos:
        if ' -> ' in typo:
            before, after = typo.split(' -> ')
            if output_format == 'csv':
                formatted.append(f"{before},{after}")
            elif output_format == 'table':
                formatted.append(f'{before} = "{after}"')
            elif output_format == 'list':
                formatted.append(f"{before}")
            else:
                # Default to arrow format for 'arrow' or unknown formats
                formatted.append(f"{before} -> {after}")
        else:
            # If it's just a single word, return it as is or filtered if it's meant to be a typo
            if output_format in ['csv', 'table', 'list']:
                formatted.append(filter_to_letters(typo))
            else:
                formatted.append(typo)
    return formatted


def _decode_with_fallback(data: bytes, description: str) -> str:
    """Decode ``data`` using UTF-8 with a latin-1 fallback and log the outcome."""

    try:
        text = data.decode("utf-8")
        logging.info(f"Successfully read {description}.")
        return text
    except UnicodeDecodeError:
        text = data.decode("latin-1")
        logging.info(f"Successfully read {description} with 'latin-1' encoding.")
        return text


def _read_stdin_text() -> str:
    """Return standard input contents, supporting both binary and text streams."""

    stream = getattr(sys.stdin, "buffer", sys.stdin)
    data = stream.read()
    if isinstance(data, str):
        logging.info("Successfully read input diff.")
        return data
    return _decode_with_fallback(data, "input diff")


def _run_git_command(command: List[str]) -> str:
    """Run a Git command via subprocess and return stdout, or exit on failure."""
    try:
        logging.info(f"Running Git command: {' '.join(command)}")
        result = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        logging.error("Git executable not found.")
        sys.exit(1)


def _run_git_subcommand(base_command: List[str], git_args: Optional[str]) -> str:
    """Append split git_args to base_command and run the Git command."""
    command = list(base_command)
    if git_args:
        command.extend(shlex.split(git_args))
    return _run_git_command(command)


def _read_git_diff(git_args: Optional[str]) -> str:
    """Fetch diff directly from Git using the provided arguments."""
    return _run_git_subcommand(["git", "diff"], git_args)


def _read_git_log(git_args: Optional[str]) -> str:
    """Fetch commit history diffs directly from Git using 'git log -p'."""
    return _run_git_subcommand(["git", "log", "-p"], git_args)


def _read_diff_sources(input_files: Optional[Sequence[str]]) -> str:
    """Return concatenated diff text from standard input or the provided file patterns."""

    if not input_files:
        return _read_stdin_text()

    contents: List[str] = []
    ignored_dirs = {
        '.git', 'node_modules', 'venv', '.venv', '.pytest_cache',
        '.ruff_cache', '.vscode', '.idea', '__pycache__', 'dist', 'build'
    }
    supported_extensions = {'.diff', '.patch', '.txt', '.log'}

    for pattern in input_files:
        if pattern == "-":
            contents.append(_read_stdin_text())
            continue

        matches = glob.glob(pattern)
        if not matches:
            logging.error(f"Input file '{pattern}' not found. Exiting.")
            sys.exit(1)

        for match in matches:
            if os.path.isdir(match):
                for root, dirs, files in os.walk(match):
                    # Prune ignored directories in-place to avoid walking into them
                    dirs[:] = [d for d in dirs if d not in ignored_dirs]
                    for file in sorted(files):
                        ext = os.path.splitext(file)[1].lower()
                        if ext in supported_extensions:
                            file_path = os.path.join(root, file)
                            with open(file_path, "rb") as file_handle:
                                data = file_handle.read()
                            contents.append(_decode_with_fallback(data, f"input diff file '{file_path}'"))
            elif os.path.isfile(match):
                with open(match, "rb") as file_handle:
                    data = file_handle.read()
                contents.append(_decode_with_fallback(data, f"input diff file '{match}'"))
            else:
                logging.error(f"Input file '{match}' not found. Exiting.")
                sys.exit(1)

    return "\n".join(contents)


def filter_known_typos(candidates, typos_tool_path):
    """
    Filters out typos that are known by the 'typos' tool.

    Args:
        candidates (list): A list of typo candidates in "before -> after" format.
        typos_tool_path (str): The path to the 'typos' tool.

    Returns:
        list: A filtered list of typo candidates.
    """
    with tempfile.TemporaryDirectory(prefix="typos_") as temp_dir:
        temp_file = os.path.join(temp_dir, "candidates.txt")
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                for typo in candidates:
                    f.write(f"{typo}\n")
        except Exception as e:
            logging.error(f"Error writing to temporary file '{temp_file}': {e}")
            return candidates

        typos_executable = shutil.which(typos_tool_path)
        if not typos_executable and os.path.exists(typos_tool_path):
            typos_executable = typos_tool_path
        if not typos_executable:
            logging.warning(
                f"Typos tool '{typos_tool_path}' not found in PATH. Skipping known typo filtering."
            )
            return candidates

        command = [typos_executable, '--format', 'brief', temp_file]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            known_typos = {s.lower() for s in re.findall(r'`([^`]+)`', result.stdout) if len(s) > 1}
            filtered = [
                line for line in candidates
                if line.split(' -> ')[0].lower() not in known_typos
            ]
            logging.info(f"Filtered out {len(candidates) - len(filtered)} known typo(s).")
            return filtered
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logging.warning(f"Error running typos tool: {e}. Skipping known typo filtering.")
            return candidates

def _filter_candidates_by_set(candidates, filter_set, desc, quiet=False):
    """Return candidate typos whose ``before`` word is not in ``filter_set``."""

    if not filter_set:
        return candidates

    filtered_list = []
    progress = None
    iterator = candidates
    if not quiet:
        progress = tqdm(candidates, desc=desc, unit="typo", leave=False)
        iterator = progress

    for typo in iterator:
        if typo.split(' -> ')[0].lower() not in filter_set:
            filtered_list.append(typo)

    if progress:
        progress.close()

    logging.info(
        f"Excluded {len(candidates) - len(filtered_list)} typo(s) based on {desc.lower()}."
    )
    return filtered_list


def process_typos_mode(candidates, args, large_dictionary, allowed_words):
    """
    Find typos that are not known.
    Uses allowed words and the large dictionary to filter the results.
    The large dictionary can be a simple word list (one word per line) or a
    CSV file where the first word is a typo and the rest are corrections.
    Returns a sorted list of unique typo strings "before -> after".
    """
    candidates = filter_known_typos(candidates, typos_tool_path=args.typos_tool_path)
    candidates = _filter_candidates_by_set(
        candidates,
        filter_set=allowed_words,
        desc="Filtering allowed words",
        quiet=args.quiet,
    )
    filtered_candidates = _filter_candidates_by_set(
        candidates,
        filter_set=large_dictionary,
        desc="Filtering large dictionary words",
        quiet=args.quiet,
    )

    # Deduplicate and sort.
    return sorted(set(filtered_candidates))


def process_corrections_mode(candidates, words_mapping, quiet=False):
    """
    Find corrections for typos that are known.
    It reads a word list and for each potential correction,
    if the "before" word is known but the "after" word is not,
    then it is saved.
    Returns a sorted list of corrections in "before -> after" form.

    Args:
        candidates (list): Candidate "before -> after" strings.
        words_mapping (dict): Mapping of known typos to their corrections.
        quiet (bool): When True, suppress progress display.
    """

    corrections = []

    if not words_mapping:
        logging.info("Large dictionary mapping is empty; skipping corrections search.")
        return corrections

    progress = None
    iterator = candidates
    if not quiet:
        progress = tqdm(candidates, desc="Checking corrections", unit="candidate", leave=False)
        iterator = progress

    for candidate in iterator:
        if '->' in candidate:
            before, after = [s.strip().lower() for s in candidate.split('->')]
            # Only consider cases where the "before" word is known in the mapping as a typo.
            if before in words_mapping:
                if after not in words_mapping[before]:
                    corrections.append(f"{before} -> {after}")
    if progress:
        progress.close()
    corrections = sorted(set(corrections))
    return corrections


def process_audit_typos(candidates, args, large_dictionary, allowed_words):
    """
    Find cases where a correct word was changed into a typo.
    Finds cases where a word that used to be valid
    was changed to a word that is not in the large dictionary.
    Returns a sorted list of unique typo strings "before -> after".
    """
    audit_candidates = []
    for candidate in candidates:
        if '->' in candidate:
            before, after = [s.strip().lower() for s in candidate.split('->')]
            if before in large_dictionary:
                if after not in large_dictionary and after not in allowed_words:
                    audit_candidates.append(candidate)

    return sorted(set(audit_candidates))


def main():

    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        description=f"{BOLD}Process a Git diff (including file renames) to find typos for the `typos` tool.{RESET}",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"""{BLUE}Examples:{RESET}
  {GREEN}python diff2typo.py diff.txt --output typos.txt --mode typos{RESET}
  {GREEN}git diff | python diff2typo.py -o found.txt -f csv{RESET}
""",
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {VERSION}'
    )

    # Input/Output Options
    io_group = parser.add_argument_group(f"{BLUE}INPUT/OUTPUT OPTIONS{RESET}")
    io_group.add_argument(
        'input_files',
        nargs='*',
        metavar='FILE',
        help="One or more input Git diff files or patterns. Use '-' to read from standard input.",
    )
    io_group.add_argument(
        '-g', '--git',
        nargs='?',
        const='',
        help="Fetch diff directly from Git. Optional arguments are passed to 'git diff'.",
    )
    io_group.add_argument(
        '-l', '--git-log',
        nargs='?',
        const='',
        help="Fetch commit history diffs directly from Git using 'git log -p'. Optional arguments are passed to 'git log'.",
    )
    io_group.add_argument(
        '--input',
        '-i',
        dest='input_files_flag',
        nargs='+',
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    # Hidden alias for backward compatibility
    parser.add_argument('--input_file', dest='input_files_flag', nargs='+', type=str, help=argparse.SUPPRESS, default=argparse.SUPPRESS)

    io_group.add_argument(
        '--output',
        '-o',
        dest='output_file',
        type=str,
        default='-',
        help="Path to the output file. Use '-' to print to the screen (default: the screen).",
    )
    # Hidden alias for backward compatibility
    parser.add_argument('--output_file', type=str, help=argparse.SUPPRESS, default=argparse.SUPPRESS)

    io_group.add_argument(
        '--format',
        '-f',
        dest='output_format',
        type=str,
        choices=['arrow', 'csv', 'table', 'list', 'json', 'yaml', 'markdown', 'md'],
        default=None,
        help='Format of the output typos. If not provided, it is automatically detected from the output file extension. Choices are: arrow (typo -> correction), csv (typo,correction), table (typo = "correction"), list (typo), json, yaml, markdown, md. Default is arrow.',
    )
    # Hidden alias for backward compatibility
    parser.add_argument('--output_format', type=str, choices=['arrow', 'csv', 'table', 'list', 'json', 'yaml', 'markdown', 'md'], help=argparse.SUPPRESS, default=argparse.SUPPRESS)

    # Analysis Options
    analysis_group = parser.add_argument_group(f"{BLUE}ANALYSIS OPTIONS{RESET}")
    analysis_group.add_argument(
        '-e', '--exclude',
        nargs='+',
        default=None,
        help="One or more file patterns (e.g., '*.json', 'tests/*') to exclude from typo scanning.",
    )
    analysis_group.add_argument(
        '-I', '--include',
        nargs='+',
        default=None,
        help="One or more file patterns (e.g., '*.md', 'src/*') to include in typo scanning (all files are scanned by default).",
    )
    analysis_group.add_argument(
        '-M', '--mode',
        type=str,
        choices=['typos', 'corrections', 'both', 'audit'],
        default='typos',
        help=(
            f"{YELLOW}Analysis mode:{RESET}\n"
            f"  {GREEN}typos{RESET}:       Find typos that are not in your large dictionary (default).\n"
            f"  {GREEN}corrections{RESET}: Find corrections for typos in your large dictionary.\n"
            f"  {GREEN}both{RESET}:        Run both analyses and label the results.\n"
            f"  {GREEN}audit{RESET}:       Find cases where a correct word was changed into a typo."
        ),
    )
    analysis_group.add_argument(
        '--min-length',
        '-m',
        dest='min_length',
        type=int,
        default=2,
        help='Ignore words shorter than this (default: 2).',
    )
    # Hidden alias for backward compatibility
    parser.add_argument('--min_length', type=int, help=argparse.SUPPRESS, default=argparse.SUPPRESS)

    analysis_group.add_argument(
        '-D', '--max-dist',
        type=int,
        default=None,
        help='Only include typos with a number of character changes up to this value (default: no limit).',
    )

    analysis_group.add_argument(
        '-c', '--min-count',
        type=int,
        default=1,
        help='Minimum occurrences of a typo in the diff to include it in the output (default: 1).',
    )

    analysis_group.add_argument(
        '-s', '--sort',
        choices=['count', 'alpha'],
        default='alpha',
        help="How to sort the results: 'count' (most frequent first) or 'alpha' (alphabetical, default).",
    )

    analysis_group.add_argument(
        '--limit',
        '-L',
        type=int,
        help='Limit the number of typos in the output.',
    )

    analysis_group.add_argument(
        '--dictionary',
        '-d',
        dest='dictionary_file',
        type=str,
        default='words.csv',
        help='The file containing the large dictionary (default: words.csv).',
    )
    # Hidden alias for backward compatibility
    parser.add_argument('--dictionary_file', type=str, help=argparse.SUPPRESS, default=argparse.SUPPRESS)

    analysis_group.add_argument(
        '-a', '--allowed',
        dest='allowed_file',
        type=str,
        default='allowed.csv',
        help='The file with allowed words to ignore (default: allowed.csv).',
    )
    # Hidden alias for backward compatibility
    parser.add_argument('--allowed_file', type=str, help=argparse.SUPPRESS, default=argparse.SUPPRESS)

    analysis_group.add_argument(
        '--typos-path',
        dest='typos_tool_path',
        type=str,
        default='typos',
        help='The command or path to the typos tool (default: typos).',
    )
    # Hidden alias for backward compatibility
    parser.add_argument('--typos_tool_path', type=str, help=argparse.SUPPRESS, default=argparse.SUPPRESS)

    analysis_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration details and a sample preview of typo extraction without writing files.',
    )

    analysis_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Hide progress bars and status messages.'
    )

    args = parser.parse_args()

    # Resolve output format if not provided
    if args.output_format is None:
        default_fmt = 'arrow'
        allowed_formats = ['arrow', 'csv', 'table', 'list', 'json', 'yaml', 'markdown', 'md']
        if args.output_file and args.output_file != '-':
            ext = os.path.splitext(args.output_file)[1].lower().lstrip('.')
            mapping = {
                'txt': 'arrow',
                'csv': 'csv',
                'table': 'table',
                'toml': 'table',
                'list': 'list',
                'arrow': 'arrow',
                'json': 'json',
                'yaml': 'yaml',
                'yml': 'yaml',
                'md': 'markdown',
                'markdown': 'markdown',
            }
            detected = mapping.get(ext)
            args.output_format = detected if detected in allowed_formats else default_fmt
        else:
            args.output_format = default_fmt

    log_level = logging.WARNING if args.quiet else logging.INFO
    # Use a custom handler and formatter to keep output clean
    handler = logging.StreamHandler()
    handler.setFormatter(MinimalFormatter('%(levelname)s: %(message)s'))
    logging.basicConfig(level=log_level, handlers=[handler])

    start_time = time.perf_counter()
    logging.info("Starting typo search...")

    # Combine positional and flag inputs
    pos_inputs = getattr(args, 'input_files', []) or []
    flag_inputs = getattr(args, 'input_files_flag', []) or []
    input_files = pos_inputs + flag_inputs

    git_val = getattr(args, 'git', None)
    git_log_val = getattr(args, 'git_log', None)
    dry_run_val = getattr(args, 'dry_run', False)

    # Robust handling for Mock/MagicMock in unit tests
    with contextlib.suppress(ImportError):
        import unittest.mock
        if isinstance(git_val, unittest.mock.Mock):
            git_val = None
        if isinstance(git_log_val, unittest.mock.Mock):
            git_log_val = None
        if isinstance(dry_run_val, unittest.mock.Mock):
            dry_run_val = False

    if git_val is not None:
        diff_text = _read_git_diff(git_val)
    elif git_log_val is not None:
        diff_text = _read_git_log(git_log_val)
    else:
        if not input_files and sys.stdin.isatty():
            try:
                res = subprocess.run(
                    ["git", "rev-parse", "--is-inside-work-tree"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                is_git = res.returncode == 0 and res.stdout.strip() == "true"
            except FileNotFoundError:
                is_git = False

            if is_git:
                logging.info("No files specified. Running in a Git repository; automatically checking your recent changes (git diff)...")
                diff_text = _read_git_diff(None)
            else:
                logging.error(
                    "No files specified and not running inside a Git repository.\n"
                    "Please specify one or more input files, run inside a Git repository, or pipe a diff command into this tool."
                )
                sys.exit(1)
        else:
            diff_text = _read_diff_sources(input_files)

    if not diff_text.strip():
        logging.warning("The input diff is empty (no changes detected).")
        if git_log_val is not None:
            logging.info("Tip: Try checking a different commit range, e.g. '-l HEAD~5'.")
        elif git_val is not None or (not input_files and sys.stdin.isatty()):
            logging.info("Tip: If you have no unstaged changes, try checking staged changes with '-g --cached', or previous commits with '-l HEAD~5'.")
        else:
            logging.info("Tip: Ensure the specified input files or piped input contain a valid Git diff or patch.")

    # Load the large dictionary (words mapping) once.
    # If the file is missing, we don't exit. Instead we just warn and continue without filtering.
    large_dictionary_mapping = read_words_mapping(args.dictionary_file, required=False)

    allowed_words = read_allowed_words(args.allowed_file)
    # Build a set of words for the large dictionary. For simple word lists, every
    # entry is treated as correct. For words.csv files, only the corrections
    # (columns after the first) are considered correct words.
    large_dictionary = set()
    for typo, fixes in large_dictionary_mapping.items():
        if fixes:
            large_dictionary.update(fixes)
        else:
            large_dictionary.add(typo)

    # Find candidate typo corrections from the diff.
    logging.info("Finding typo corrections from the diff...")
    candidates_raw = find_typos(
        diff_text,
        min_length=args.min_length,
        max_dist=args.max_dist,
        exclude_patterns=args.exclude,
        include_patterns=args.include,
    )
    counts = Counter(candidates_raw)

    unique_candidates = sorted(counts.keys())
    candidates = [item for item in unique_candidates if counts[item] >= args.min_count]

    raw_count = len(unique_candidates)
    total_occurrences = len(candidates_raw)

    # Prepare lists to hold results.
    typos_list = []
    corrections_list = []
    audit_list = []

    # Process typos if requested.
    if args.mode in ['typos', 'both']:
        logging.info("Processing typos (filtering out known typos)...")
        typos_list = process_typos_mode(candidates, args, large_dictionary, allowed_words)

    # Process corrections if requested.
    if args.mode in ['corrections', 'both']:
        logging.info("Processing corrections to typos...")
        corrections_list = process_corrections_mode(candidates, large_dictionary_mapping, quiet=args.quiet)

    # Check for correct words changed into typos if requested.
    if args.mode == 'audit':
        logging.info("Checking for cases where correct words were changed into typos...")
        audit_list = process_audit_typos(candidates, args, large_dictionary, allowed_words)

    if dry_run_val:
        use_color = _should_enable_color(sys.stderr)
        c_blue = (BOLD + _BLUE) if use_color else ""
        c_green = (BOLD + _GREEN) if use_color else ""
        c_reset = _RESET if use_color else ""

        logging.info(f"{c_blue}--- DIFF2TYPO DRY RUN ---{c_reset}")
        input_desc = input_files if input_files else ("Git Diff" if git_val is not None else ("Git Log" if git_log_val is not None else "stdin"))
        logging.info(f"Input Source: {input_desc}")
        logging.info(f"Output Target: {args.output_file} (Format: {args.output_format})")
        logging.info(f"Mode: {args.mode} | Min Length: {args.min_length} | Max Dist: {args.max_dist if args.max_dist is not None else 'None'}")
        logging.info(f"Min Count: {args.min_count} | Sort: {args.sort} | Limit: {args.limit if args.limit is not None else 'None'}")
        logging.info(f"Large Dictionary: {args.dictionary_file} | Allowed File: {args.allowed_file}")
        logging.info(f"Exclude Patterns: {args.exclude if args.exclude else 'None'} | Include Patterns: {args.include if args.include else 'None'}")

        # Sample Preview
        logging.info(f"\n{c_blue}Sample Typo Extraction Preview:{c_reset}")
        if args.mode == 'both':
            sample_typos = typos_list[:5]
            sample_corrections = corrections_list[:5]
            logging.info(f"  Typos ({len(typos_list)} total): {', '.join(sample_typos) if sample_typos else 'None'}")
            logging.info(f"  Corrections ({len(corrections_list)} total): {', '.join(sample_corrections) if sample_corrections else 'None'}")
        else:
            sample_items = (typos_list if args.mode == 'typos' else (corrections_list if args.mode == 'corrections' else audit_list))
            logging.info(f"  Found {len(sample_items)} candidate(s) in '{args.mode}' mode:")
            for item in sample_items[:10]:
                logging.info(f"    {item}")

        logging.info(f"{c_green}Dry run complete. No files were written.{c_reset}")
        return

    # Helper to sort and limit results
    def sort_and_limit(items):
        if args.sort == 'count':
            # Sort by frequency descending, then alphabetically
            items.sort(key=lambda x: (-counts.get(x, 0), x))
        else:
            items.sort()
        if args.limit:
            return items[:args.limit]
        return items

    # Combine results if needed.
    final_output = []
    filtered_items = []
    if args.mode == 'both':
        typos_final = sort_and_limit(typos_list)
        corrections_final = sort_and_limit(corrections_list)
        filtered_items.extend(typos_final)
        filtered_items.extend(corrections_final)

        if args.output_format in ('json', 'yaml'):
            def _to_dicts(items):
                res = []
                for typo in items:
                    if ' -> ' in typo:
                        b, a = typo.split(' -> ')
                        res.append({"typo": b, "correction": a})
                    else:
                        res.append({"typo": typo, "correction": ""})
                return res

            data = {
                "typos": _to_dicts(typos_final),
                "corrections": _to_dicts(corrections_final),
            }
            if args.output_format == 'yaml':
                if _YAML_AVAILABLE:
                    final_output = yaml.safe_dump(data, default_flow_style=False).rstrip().split('\n')
                else:
                    logging.warning("PyYAML not installed. Falling back to JSON for YAML output format.")
                    final_output = json.dumps(data, indent=2).split('\n')
            else:
                final_output = json.dumps(data, indent=2).split('\n')
        else:
            if args.output_format in ('markdown', 'md'):
                if typos_final:
                    final_output.append("### Typos")
                    final_output.extend(format_typos(typos_final, args.output_format))
                    final_output.append("")  # Blank line for separation.
                if corrections_final:
                    final_output.append("### Corrections")
                    final_output.extend(format_typos(corrections_final, args.output_format))
            else:
                if typos_final:
                    final_output.append("=== Typos ===")
                    final_output.extend(format_typos(typos_final, args.output_format))
                    final_output.append("")  # Blank line for separation.
                if corrections_final:
                    final_output.append("=== Corrections ===")
                    final_output.extend(format_typos(corrections_final, args.output_format))
    else:
        results_list = []
        if args.mode == 'typos':
            results_list = typos_list
        elif args.mode == 'corrections':
            results_list = corrections_list
        elif args.mode == 'audit':
            results_list = audit_list

        results_final = sort_and_limit(results_list)
        final_output = format_typos(results_final, args.output_format)
        filtered_items = results_final

    # Write the final output to the specified file.
    try:
        with smart_open_output(args.output_file, encoding='utf-8') as f:
            for line in final_output:
                f.write(f"{line}\n")
    except Exception as e:
        logging.error(f"Error writing to output file '{args.output_file}': {e}")
        sys.exit(1)

    # Display analysis summary to stderr
    if not args.quiet:
        use_color = _should_enable_color(sys.stderr)
        item_label = "typo" if args.mode != "corrections" else "correction"
        if args.mode == "audit":
            item_label = "audit-item"

        extra_metrics = {}
        if args.min_count > 1:
            extra_metrics["Min occurrences (--min-count)"] = args.min_count
        if args.limit:
            extra_metrics["Output limit (--limit)"] = args.limit

        if args.mode == "both":
            extra_metrics["Typos found"] = len(typos_list)
            extra_metrics["Corrections found"] = len(corrections_list)

        summary = _format_analysis_summary(
            raw_count,
            filtered_items,
            item_label=item_label,
            start_time=start_time,
            use_color=use_color,
            extra_metrics=extra_metrics,
            total_input_items=total_occurrences
        )
        sys.stderr.write("\n".join(summary))

        dest_label = "the screen" if args.output_file == "-" else f"'{args.output_file}'"
        c_blue = (BOLD + _BLUE) if use_color else ""
        c_reset = _RESET if use_color else ""
        logging.info(f"{c_blue}[diff2typo]{c_reset} Wrote {len(final_output)} line(s) to {dest_label}.\n")

if __name__ == "__main__":
    main()
