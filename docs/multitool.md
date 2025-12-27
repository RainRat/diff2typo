# Multitool

**Multitool** is a versatile utility for processing text files. It can extract specific data (like columns from a CSV or text inside backticks), compare files, and clean up lists.

## Quick Start

Run the tool with a mode and your input file:

```bash
python multitool.py <MODE> [OPTIONS]
```

Most modes default to reading from `input.txt` if you do not specify an input file.

## Modes

Multitool operates in different "modes," each designed for a specific task.

### 1. Extraction Modes

These modes help you pull specific data out of a messy file.

- **`arrow`**
  - **What it does:** Extracts the left side of an arrow (`typo -> correction`). Useful for getting a clean list of typos from a log.
  - **Example:** `python multitool.py arrow --input typos.log`

- **`backtick`**
  - **What it does:** Extracts text found inside backticks (like \`code\`). It is smart enough to pick the most relevant item from lines that contain error messages or warnings.
  - **Example:** `python multitool.py backtick --input build.log`

- **`csv`**
  - **What it does:** Extracts columns from a CSV file. By default, it skips the first column (often an ID or key) and keeps the rest. Use `--first-column` to keep *only* the first column.
  - **Example:** `python multitool.py csv --input data.csv`

- **`line`**
  - **What it does:** Reads a file line by line. Use this to simply clean or filter a text file without special extraction logic.
  - **Example:** `python multitool.py line --input raw_words.txt`

- **`sample`**
  - **What it does:** Picks a random set of lines from a file. You can choose a specific number (`--n 100`) or a percentage (`--percent 10`).
  - **Example:** `python multitool.py sample --input big_log.txt --n 50`

### 2. Analysis & Comparison Modes

These modes help you analyze your data or compare multiple files.

- **`check`**
  - **What it does:** Finds words that appear as both a typo *and* a correction. This is useful for spotting errors in your typo database (loops).
  - **Example:** `python multitool.py check --input mappings.csv`

- **`combine`**
  - **What it does:** Merges multiple files into one list, removes duplicates, and sorts the result alphabetically.
  - **Example:** `python multitool.py combine --input file1.txt file2.txt`

- **`count`**
  - **What it does:** Counts how many times each word appears in a file and sorts them by frequency.
  - **Example:** `python multitool.py count --input all_typos.txt`

- **`filterfragments`**
  - **What it does:** Removes words from your input file if they appear anywhere inside a second file (`--file2`).
  - **Example:** `python multitool.py filterfragments --input candidates.txt --file2 dictionary.txt`

- **`set_operation`**
  - **What it does:** Compares two files using standard set logic:
    - `intersection`: Finds lines common to both files.
    - `union`: Combines all lines from both files.
    - `difference`: Finds lines in the first file that are not in the second.
  - **Example:** `python multitool.py set_operation --input a.txt --file2 b.txt --operation difference`

## Common Options

These options work with most modes:

- `--input`: The file(s) to read. Defaults to `input.txt` if omitted (except for `combine` which needs explicit files).
- `--output`: The file to write results to. Defaults to printing to the screen.
- `--min-length`: Skip words shorter than this length (default: 3).
- `--process-output`: Automatically sort, deduplicate, and lowercase the final list.
- `--quiet`: Hide progress bars and log messages.
