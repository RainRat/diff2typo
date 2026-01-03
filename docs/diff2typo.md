# diff2typo.py

**Purpose:** Scans Git diffs to identify typos that have been fixed. It effectively "learns" from your history to help you build a database of typos to avoid in the future.

## Usage

```bash
# Read from files
python diff2typo.py my_changes.diff [OPTIONS]

# Read from stdin
git diff | python diff2typo.py [OPTIONS]
```

## Core Features

1. **Diff Parsing:** Reads standard Git diffs (from file or stdin).
2. **Context Awareness:** Splits compound words (camelCase, snake_case) to find typos within variable names.
3. **Filtering:** Uses dictionary files and "allowed" lists to prevent false positives.
4. **Integration:** Can verify candidates against the external `typos` crate to avoid duplicates.

## Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `INPUT_FILES` | stdin | Positional arguments for input diff file(s). Supports glob patterns (e.g., `*.diff`). |
| `--input_file`, `-i` | None | **Legacy flag.** Alternative to positional arguments. |
| `--output_file` | `output.txt` | Where to write the results. |
| `--mode` | `typos` | **`typos`**: Output the incorrect word.<br>**`corrections`**: Output the fix (if new).<br>**`both`**: Output separated lists. |
| `--output_format` | `arrow` | `arrow` (a->b), `csv` (required for `typostats`), `table`, or `list`. |
| `--dictionary_file` | `words.csv` | A list of valid words to validate corrections against. |
| `--allowed_file` | `allowed.csv` | A list of words to explicitly ignore (false positive suppression). |
| `--min_length` | `2` | Minimum length of a word to be considered. |

## Examples

**Extract new typos from a specific diff file:**

```bash
python diff2typo.py feature.diff --mode typos --output_format list
```

**Pipe directly from Git:**

```bash
git diff | python diff2typo.py --output_file found_typos.txt --mode both
```

**Find patterns with typostats:**

```bash
python diff2typo.py recent_changes.diff --output_format csv --output_file typos.csv
python typostats.py typos.csv
```
