# Multitool

**Multitool** is a versatile utility for processing text files. It can extract specific data (like columns from a CSV or text inside backticks), compare files, and clean up lists.

## Quick Start

Run the tool with a mode and your input files:

```bash
python multitool.py <MODE> [INPUT_FILES...] [OPTIONS]
```

Most modes default to reading from **standard input (stdin)** if you do not specify an input file. This makes it easy to pipe data into Multitool.

## Modes

Multitool operates in different "modes," each designed for a specific task.

### 1. Extraction Modes

These modes help you pull specific data out of a messy file.

- **`arrow`**
  - **What it does:** Extracts the left side of an arrow (`typo -> correction`). Useful for getting a clean list of typos from a log. You can also extract the right side (the correction) by adding the `--right` flag.
  - **Example:** `python multitool.py arrow typos.log --right`

- **`table`**
  - **What it does:** Extracts the key or value from a table entry (`key = "value"`). Saves the key by default. Use the `--right` flag to extract the quoted value instead.
  - **Example:** `python multitool.py table typos.toml --right`

- **`backtick`**
  - **What it does:** Extracts text found inside backticks (like \`code\`). It is smart enough to pick the most relevant item from lines that contain error messages or warnings.
  - **Example:** `python multitool.py backtick build.log`

- **`csv`**
  - **What it does:** Extracts columns from a CSV file. By default, it extracts **all columns except the first one**. Use `--first-column` to keep *only* the first column.
  - **Example:** `python multitool.py csv data.csv`

- **`markdown`**
  - **What it does:** Extracts items from Markdown bulleted lists (lines starting with `- `, `* `, or `+ `). It can also split items by `:` or `->` to extract one side of a pair (use the `--right` flag for the second part).
  - **Example:** `python multitool.py markdown notes.md --right`

- **`json`**
  - **What it does:** Extracts values from a JSON file based on a specific key. You can use dots to access nested keys (e.g., `user.name`). It automatically handles lists.
  - **Example:** `python multitool.py json report.json --key replacements.typo`

- **`yaml`**
  - **What it does:** Extracts values from a YAML file based on a key path. Like JSON mode, it supports dot notation (e.g., `config.items`) and handles lists.
  - **Example:** `python multitool.py yaml config.yaml --key config.items`

- **`line`**
  - **What it does:** Reads a file line by line. Use this to simply clean or filter a text file without special extraction logic.
  - **Example:** `python multitool.py line raw_words.txt`

- **`regex`**
  - **What it does:** Extracts text matching a Python regular expression pattern. Unlike other modes, it **preserves exact text** (it does not convert to lowercase or remove punctuation) by default.
  - **Example:** `python multitool.py regex inputs.txt --pattern 'user_\w+'`

### 2. Manipulation Modes

These modes help you transform or combine your data.

- **`combine`**
  - **What it does:** Merges multiple files (or standard input) into one list, removes duplicates, and sorts the result alphabetically.
  - **Note:** This mode has built-in sorting and deduplication; the `--process-output` flag is not needed.
  - **Example:** `python multitool.py combine file1.txt file2.txt`

- **`filterfragments`**
  - **What it does:** Removes words from your input file if they appear anywhere inside a second file (`--file2`).
  - **Example:** `python multitool.py filterfragments candidates.txt --file2 dictionary.txt`

- **`map`**
  - **What it does:** Replaces items in your list with values from a mapping file. Supports CSV, Arrow, Table (`typo = "correction"`), JSON, and YAML formats. By default, it keeps items that are not in the mapping. The `--min-length` and `--max-length` filters are **re-applied** to items after they are transformed. Use `--drop-missing` to remove unmatched items.
  - **Example:** `python multitool.py map input.txt --mapping corrections.csv`

- **`sample`**
  - **What it does:** Picks a random set of lines from a file. You can choose a specific number (`--n 100`) or a percentage (`--percent 10`).
  - **Example:** `python multitool.py sample big_log.txt --n 50`

- **`set_operation`**
  - **What it does:** Compares two files using standard set logic:
    - `intersection`: Finds lines common to both files.
    - `union`: Combines all lines from both files.
    - `difference`: Finds lines in the first file that are not in the second.
  - **Example:** `python multitool.py set_operation a.txt --file2 b.txt --operation difference`

- **`zip`**
  - **What it does:** Combines two files line-by-line into a paired format. It applies `--min-length` and `--max-length` filters to **both items in each pair**. If the files have a different number of lines, the output will stop at the end of the shortest file.
  - **Supported Formats:** `line`, `json`, `csv`, `markdown`, `arrow`, `table`, and `yaml`.
  - **Example:** `python multitool.py zip typos.txt --file2 corrections.txt --output-format arrow`

- **`swap`**
  - **What it does:** Reverses the order of elements in paired data (e.g., `typo -> correction` becomes `correction -> typo`).
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `json`, and `yaml`.
  - **Example:** `python multitool.py swap mappings.csv --output-format arrow`

- **`pairs`**
  - **What it does:** Processes paired data (like `typo -> correction`) from any supported format and writes it to the specified output format. This is the primary way to convert between paired formats (e.g., from JSON to CSV) while applying cleaning and length filters.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `json`, and `yaml`.
  - **Example:** `python multitool.py pairs typos.json --output-format csv`

### 3. Analysis Modes

These modes help you analyze your data.

- **`check`**
  - **What it does:** Finds words that appear as both a typo *and* a correction. This is useful for spotting errors in your typo database (loops).
  - **Example:** `python multitool.py check mappings.csv`

- **`conflict`**
  - **What it does:** Identifies typos that are associated with more than one unique correction. Use this to find inconsistencies in your typo lists.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `json`, and `yaml`.
  - **Example:** `python multitool.py conflict mappings.csv`

- **`count`**
  - **What it does:** Counts how many times each word appears in a file and sorts them by frequency (most frequent first).
  - **Options:** Use `--min-count` and `--max-count` to filter results by their frequency.
  - **Note:** This mode has built-in sorting; the `--process-output` flag is not needed.
  - **Example:** `python multitool.py count all_typos.txt --min-count 5`

- **`near_duplicates`**
  - **What it does:** Identifies pairs of words in your list that are very similar (within a small edit distance). This is useful for finding potential typos or unintended duplicates in a project.
  - **Options:** Use `--min-dist` and `--max-dist` to control the edit distance threshold, and `--show-dist` to see the distance in the output.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `json`, and `yaml`.
  - **Example:** `python multitool.py near_duplicates words.txt --max-dist 1 --show-dist`

- **`similarity`**
  - **What it does:** Filters paired data (like `typo -> correction`) based on the number of changes (edit distance) needed to turn one word into another. Use this to remove noise or find specific types of typos.
  - **Options:** Use `--min-dist` and `--max-dist` to set the distance range, and `--show-dist` to include the distance in the output.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `json`, and `yaml`.
  - **Example:** `python multitool.py similarity typos.txt --max-dist 2 --show-dist`

- **`stats`**
  - **What it does:** Provides a high-level summary of your dataset. It reports counts, unique items, and length distributions. If the `--pairs` flag is used, it additionally analyzes the file as paired data (typos/corrections) and reports conflicts (one typo to multiple corrections), overlaps (words that are both typos and corrections), and edit distance statistics.
  - **Example:** `python multitool.py stats typos.csv --pairs`

## Common Options

These options work with most modes:

- `[INPUT_FILES...]`: One or more files to read. Defaults to **standard input (stdin)** if omitted.
- `--output`: The file to write results to. Defaults to printing to the screen.
- `--output-format`: The format of the output. Options include `line` (default), `json`, `csv`, `markdown`, `arrow`, and `table`.
- `--min-length`: Skip words shorter than this length (default: 3).
- `--max-length`: Skip words longer than this length (default: 1000).
- `--process-output`: Sorts the final list and removes duplicates. Use this to organize your output or remove redundant entries.
- `--raw`: Keep punctuation and capitalization. By default, most tools convert everything to lowercase and remove all characters except for lowercase **a through z**. Use this flag if you need to preserve numbers, punctuation, or capitalization.
- `--quiet`: Hide progress bars and log messages.
