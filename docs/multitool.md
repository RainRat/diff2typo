# Multitool

**Multitool** is a multipurpose tool for processing text files. It can get specific data (like columns from a CSV or text inside backticks), compare files, and clean up lists.

## Quick Start

Run the tool with a mode and your input files:

```bash
python multitool.py <MODE> [INPUT_FILES...] [OPTIONS]
```

Use the **`help`** subcommand to see all available modes or get detailed information about a specific one:

```bash
python multitool.py help        # Show summary of all modes
python multitool.py help count  # Show details for 'count' mode
```

Most modes default to reading from **standard input** if you do not specify an input file. This makes it easy to send data from other commands into Multitool.

## Modes

Multitool operates in different "modes," each designed for a specific task.

### 1. GETTING DATA

These modes help you pull specific data out of a messy file.

- **`arrow`**
  - **What it does:** Gets text from lines that use arrows (for example, `typo -> correction`). Useful for getting a clean list of typos from a log. It saves the left side by default. You can also get the right side (the correction) by adding the `--right` flag.
  - **Example:** `python multitool.py arrow typos.log --right`

- **`table`**
  - **What it does:** Gets keys or values from entries like `key = "value"`. It saves the key by default. Use the `--right` flag to get the quoted value instead.
  - **Example:** `python multitool.py table typos.toml --right`

- **`backtick`**
  - **What it does:** Gets text found inside backticks (for example, \`code\`). It picks items near words like 'error' or 'warning' to find the most useful data.
  - **Example:** `python multitool.py backtick build.log`

- **`quoted`**
  - **What it does:** Gets text found inside double (`"`) or single (`'`) quotes. It handles backslash escaping (like `\"` or `\'`) to correctly extract strings from code or data files.
  - **Example:** `python multitool.py quoted source.py`

- **`between`**
  - **What it does:** Gets text found between two markers. It is useful for extracting data from templating languages, logs, or custom file formats. You can use the `--multi-line` flag to capture content that spans multiple lines.
  - **Options:** Requires `--start` and `--end` to define the markers.
  - **Example:** `python multitool.py between input.txt --start '{{' --end '}}'`

- **`csv`**
  - **What it does:** Gets columns from a CSV file. By default, it picks **every column except the first one**. Use `--first-column` to get *only* the first column, or `--column` (or `-c`) followed by one or more numbers to get specific columns. Use `--delimiter` (or `-d`) to pick a different column separator (for example, `;`).
  - **Example:** `python multitool.py csv data.csv --column 2`

- **`markdown`**
  - **What it does:** Gets items from Markdown bulleted lists (lines starting with `- `, `* `, or `+ `). It can also split items by `:` or `->` to get one side of a pair (use the `--right` flag for the second part).
  - **Example:** `python multitool.py markdown notes.md --right`

- **`md-table`**
  - **What it does:** Gets text from Markdown tables. It saves the first column by default. Use the `--right` flag to save the second column instead, or `--column` (or `-c`) followed by one or more numbers to get specific columns. It automatically skips header and divider rows.
  - **Example:** `python multitool.py md-table readme.md --column 2`

- **`json`**
  - **What it does:** Gets values from a JSON file using a specific key. You can use dots for keys inside other keys (for example, `user.name`). If you do not provide a key, it gets items from the top level. It automatically handles lists and objects.
  - **Example:** `python multitool.py json report.json --key replacements.typo`

- **`yaml`**
  - **What it does:** Gets values from a YAML file using a key path. Like JSON mode, it supports dot notation (for example, `config.items`). If you do not provide a key, it gets items from the top level.
  - **Example:** `python multitool.py yaml config.yaml --key config.items`

- **`line`**
  - **What it does:** Reads a file line by line. Use this to simply clean or filter a text file without special getting logic.
  - **Example:** `python multitool.py line raw_words.txt`

- **`words`**
  - **What it does:** Gets individual words from a file. It splits lines by whitespace by default, but you can pick a custom character using `--delimiter` (or `-d`). Use the `--smart` (or `-S`) flag to also split by symbols and capital letters (for example, splitting "CamelCase" into "Camel" and "Case").
  - **Example:** `python multitool.py words report.txt --smart`

- **`ngrams`**
  - **What it does:** Gets sequences of N words from a file. This is useful for finding common phrases or context around typos. It supports sequences across line boundaries.
  - **Options:** Use `-n` to pick the number of words in each sequence (default is 2). Like the `words` mode, it supports custom delimiters and smart word splitting.
  - **Example:** `python multitool.py ngrams report.txt -n 2 --smart`

- **`regex`**
  - **What it does:** Finds and gets text that matches a Python regular expression pattern. Unlike other modes, it **keeps the original text** (it does not convert to lowercase or remove punctuation) by default.
  - **Example:** `python multitool.py regex inputs.txt --pattern 'user_\w+'`

### 2. CHANGING DATA

These modes help you transform or combine your data.

- **`combine`**
  - **What it does:** Merges multiple files (or standard input) into one list, removes duplicates, and sorts the result alphabetically.
  - **Note:** This mode has built-in sorting and deduplication; the `--process-output` flag is not needed.
  - **Example:** `python multitool.py combine file1.txt file2.txt`

- **`unique`**
  - **What it does:** Removes duplicate items from your list while **keeping the original order**. This is useful when the sequence of items is important.
  - **Example:** `python multitool.py unique raw_typos.txt`

- **`resolve`**
  - **What it does:** Finds and shortens chains of typo corrections. For example, if your mapping file contains `A -> B` and `B -> C`, this mode will resolve them to `A -> C` and `B -> C`. This ensures that your mappings always point directly to the final correct word, making them more efficient for fixing typos and analysis.
  - **Example:** `python multitool.py resolve mappings.csv`

- **`rename`**
  - **What it does:** Changes file and folder names using a mapping file or extra pairs. It is useful for fixing typos in filenames across your entire project. It handles nested renames by processing files before their parent folders.
  - **Options:**
    - Supports `--in-place` renaming and `--dry-run` preview.
    - Use the `--add` flag to provide extra mapping pairs (for example, `--add old_name:new_name`) directly on the command line.
    - Use the `--smart-case` flag to automatically match the casing of the original filename.
  - **Example:** `python multitool.py rename src/ --mapping corrections.csv --in-place`

- **`diff`**
  - **What it does:** Finds added, removed, and changed items between two files. It can compare simple lists of words or (with the `--pairs` flag) find changes in typo-correction mappings.
  - **Supported Formats:** Color-coded terminal output (default) and structured JSON output.
  - **Example:** `python multitool.py diff old_list.txt new_list.txt`
  - **Pairs Example:** `python multitool.py diff old_typos.csv new_typos.csv --pairs`

- **`filterfragments`**
  - **What it does:** Removes words from your input file if they appear anywhere inside a second file (`--file2`).
  - **Example:** `python multitool.py filterfragments candidates.txt --file2 dictionary.txt`

- **`map`**
  - **What it does:** Changes items in your list using values from a mapping file or extra pairs. Supports CSV, Arrow, Table (`typo = "correction"`), JSON, and YAML formats. By default, it keeps items that are not in the mapping. The `--min-length` and `--max-length` filters are **re-applied** to items after they are changed. Use `--drop-missing` to remove unmatched items.
  - **Options:**
    - Use the `--add` flag to provide extra mapping pairs (for example, `--add teh:the`) directly on the command line.
  - **Example:** `python multitool.py map input.txt --add teh:the`

- **`sample`**
  - **What it does:** Picks a random set of lines from a file. You can pick a specific number (`--n 100`) or a percentage (`--percent 10`).
  - **Example:** `python multitool.py sample big_log.txt --n 50`

- **`set_operation`**
  - **What it does:** Compares two files using standard set logic:
    - `intersection`: Finds lines common to both files.
    - `union`: Combines all lines from both files.
    - `difference`: Finds lines in the first file that are not in the second.
    - `symmetric_difference`: Finds lines that are unique to each file (items in either file, but not both).
  - **Example:** `python multitool.py set_operation a.txt --file2 b.txt --operation symmetric_difference`

- **`zip`**
  - **What it does:** Joins two files line-by-line into a paired format. It applies `--min-length` and `--max-length` filters to **both items in each pair**. If the files have a different number of lines, the output will stop at the end of the shortest file.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py zip typos.txt --file2 corrections.txt --output-format arrow`

- **`swap`**
  - **What it does:** Flips the order of elements in paired data (for example, `typo -> correction` becomes `correction -> typo`).
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py swap mappings.csv --output-format arrow`

- **`scrub`**
  - **What it does:** Fixes typos in your text files using a mapping file or extra pairs. It tries to keep the surrounding context (punctuation, whitespace) while fixing errors. It automatically handles compound words like `CamelCase` and `snake_case` variables.
  - **Supported Formats:** Supports CSV, Arrow, Table, JSON, and YAML mapping formats.
  - **Options:**
    - Use the `--add` flag to provide extra mapping pairs (for example, `--add teh:the`) directly on the command line.
  - **In-Place Editing:** Use the `--in-place` flag to modify files directly. If you provide an extension (for example, `--in-place .bak`), the tool will create a backup of each file before modifying it.
  - **Dry Run:** Use the `--dry-run` flag to see how many fixes would be made without actually changing any files.
  - **Diff Preview:** Use the `--diff` flag to see a unified diff of the changes that would be made. This is useful for reviewing fixes before applying them in-place.
  - **Smart Casing:** Use the `--smart-case` flag to automatically match the casing of the original word. For example, if the mapping is `teh -> the`, then `Teh` will be replaced with `The`, and `TEH` with `THE`.
  - **Example:** `python multitool.py scrub input.txt --add teh:the --diff`
  - **In-Place Example:** `python multitool.py scrub file1.txt file2.txt --mapping corrections.csv --in-place`

- **`standardize`**
  - **What it does:** Fixes inconsistent casing by using the most frequent form. It analyzes your files to find words used with different capitalization (for example, 'database' vs 'Database'). It then automatically replaces all less frequent versions with the most popular one across the entire project. This ensures naming consistency without needing a manual mapping file.
  - **Options:**
    - Supports `--in-place` editing and `--dry-run` preview.
    - **Diff Preview:** Use the `--diff` flag to see a unified diff of the changes that would be made.
    - Works with standard filters like `--min-length` and `--max-length`.
  - **Example:** `python multitool.py standardize . --diff --min-length 4`

- **`highlight`**
  - **What it does:** Searches for words from a list, mapping, or extra pairs and colors them in the output. This is useful as a preview before using the `scrub` mode to make permanent changes.
  - **Options:**
    - Use the `--mapping` flag to provide a file with typos or words to find.
    - Use the `--add` flag to provide extra mapping pairs (for example, `--add teh:the`) or words to match directly on the command line.
    - The `--smart` flag allows for coloring subwords within larger compound words (like variable names).
  - **Example:** `python multitool.py highlight input.txt --add teh:the`

- **`pairs`**
  - **What it does:** Works with and converts paired data (like `typo -> correction`) from any supported format and writes it to the specified output format. This is the primary way to convert between paired formats (for example, from JSON to CSV) while applying cleaning and length filters.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py pairs typos.json --output-format csv`

### 3. CHECKING DATA

These modes help you analyze your data.

- **`check`**
  - **What it does:** Finds words that appear as both a typo *and* a correction. This is useful for spotting errors in your typo database (loops).
  - **Example:** `python multitool.py check mappings.csv`

- **`classify`**
  - **What it does:** Groups typo corrections based on their error type. It labels each pair with a code like `[K]` (Keyboard), `[T]` (Transposition), `[D]` (Deletion), `[I]` (Insertion), `[R]` (Replacement), or `[M]` (Multiple letters).
  - **Options:** Use `--show-dist` to include the number of character changes in the output labels.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py classify typos.txt --show-dist --output labeled.txt`

- **`conflict`**
  - **What it does:** Finds typos that are associated with more than one unique correction. Use this to find inconsistencies in your typo lists.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py conflict mappings.csv`

- **`count`**
  - **What it does:** Counts how often each word appears in a file and sorts them by frequency (most frequent first).
  - **Options:**
    - `--min-count` and `--max-count`: Filter results by their frequency.
    - `-d`, `--delimiter`: The character to split words by (default: whitespace).
    - `-S`, `--smart`: Split by symbols and capital letters (for example, splitting "CamelCase" into "Camel" and "Case").
    - `-p`, `--pairs`: Count frequencies of word pairs (for example, `typo -> correction`) instead of single words.
    - `-B`, `--by-file`: Count how many files contain each item instead of total occurrences. This is useful for finding words that are common across your entire project.
  - **Visual Report:** Use `--output-format arrow` to generate a rich visual report. This includes an **ANALYSIS SUMMARY** dashboard with metrics like retention rate, an aligned frequency table, and high-resolution bar charts.
  - **Supported Formats:** `arrow`, `json`, `csv`, `markdown`, `md-table`, and `line`.
  - **Note:** This mode has built-in sorting; the `--process-output` flag is not needed.
  - **Example:** `python multitool.py count all_typos.txt --min-count 5 -f arrow --smart`
  - **By-File Example:** `python multitool.py count src/*.py --by-file --output-format arrow`
  - **Pairs Example:** `python multitool.py count typos.log --pairs --output-format arrow`

- **`cycles`**
  - **What it does:** Finds loops in typo-correction pairs (for example, "A" maps to "B" and "B" maps back to "A"). These repeated loops can cause issues when automatically fixing text and often indicate errors in your data.
  - **Example:** `python multitool.py cycles typos.csv --output-format arrow`

- **`fuzzymatch`**
  - **What it does:** Finds words in your list that are similar to words in a second list (large dictionary). Use this to find likely corrections for typos.
  - **Options:** Use `--min-dist` and `--max-dist` to control the number of changes allowed, and `--show-dist` to see the number of changes in the output.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py fuzzymatch typos.txt words.csv --max-dist 1 --show-dist`

- **`near_duplicates`**
  - **What it does:** Finds pairs of words in your list that are very similar (only a few characters are different). This is useful for finding potential typos or unintended duplicates.
  - **Options:** Use `--min-dist` and `--max-dist` to control the number of changes allowed, and `--show-dist` to see the number of changes in the output.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py near_duplicates words.txt --max-dist 1 --show-dist`

- **`similarity`**
  - **What it does:** Filters paired data (like `typo -> correction`) based on the number of character changes needed to turn one word into another. Use this to remove extra data or find specific types of typos.
  - **Options:** Use `--min-dist` and `--max-dist` to set the range of allowed changes, and `--show-dist` to include the number of changes in the output.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py similarity typos.txt --max-dist 2 --show-dist`

- **`stats`**
  - **What it does:** Shows a high-level summary of your dataset. It reports counts, unique items, and statistics. If the `--pairs` flag is used, it additionally analyzes the file as paired data (typos/corrections) and reports conflicts (one typo to multiple corrections), overlaps (words that are both typos and corrections), and character change statistics.
  - **Supported Formats:** `json`, `yaml`, `markdown`, `md-table`, and `line` (human-readable).
  - **Example:** `python multitool.py stats typos.csv --pairs`

- **`discovery`**
  - **What it does:** Automatically finds potential typos in a text by finding rare words that are very similar to frequent words. It assumes that frequent words are likely correct and rare versions are likely typos. This is a powerful way to find errors in a dataset without needing a dictionary.
  - **Options:**
    - `--rare-max`: Maximum frequency for a word to be considered a potential typo (default: 1).
    - `--freq-min`: Minimum frequency for a word to be considered a potential correction (default: 5).
    - `--min-dist` and `--max-dist`: Control the number of allowed character changes between the typo and the correction.
    - `--show-dist`: Include the number of character changes in the output.
    - `-d`, `--delimiter`: The character to split words by (default: whitespace).
    - `-S`, `--smart`: Split by symbols and capital letters (for example, splitting "CamelCase" into "Camel" and "Case").
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py discovery code.py --smart --rare-max 2 --freq-min 10 --max-dist 1`

- **`casing`**
  - **What it does:** Finds words that appear in your files with multiple different casing styles (for example, 'hello', 'Hello', 'HELLO'). This is useful for seeing inconsistent naming or typos that differ only by case.
  - **Options:**
    - `-d`, `--delimiter`: The character to split words by (default: whitespace).
    - `-S`, `--smart`: Split by symbols and capital letters (for example, splitting "CamelCase" into "Camel" and "Case").
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py casing report.txt --smart --output-format arrow`

- **`repeated`**
  - **What it does:** Finds consecutive identical words (for example, "the the"). It's a common typing error that is often missed by standard spell-checkers.
  - **Options:**
    - `-d`, `--delimiter`: The character to split words by (default: whitespace).
    - `-S`, `--smart`: Split by symbols and capital letters (for example, splitting "CamelCase" into "Camel" and "Case").
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py repeated report.txt --smart --output-format arrow`

- **`search`**
  - **What it does:** A typo-aware search tool. It searches for a query in your files and can find similar words (typos) or subword matches.
  - **Options:**
    - `-Q`, `--query`: The word or pattern to search for.
    - `--max-dist`: Maximum number of character changes for similar word matching (default: 0).
    - `-S`, `--smart`: Search for subwords within larger items (for example, finding "teh" inside "tehWord").
    - `--line-numbers`: Show the filename and line number for each match.
  - **Example:** `python multitool.py search report.txt -Q 'teh' --max-dist 1 --line-numbers`

- **`scan`**
  - **What it does:** Like a batch version of the `search` mode. It searches for every word in a mapping file, list, or extra pairs and reports all matches with filename, line number, and highlighting. This is the recommended way to check your project for a set of known typos before performing replacements.
  - **Options:**
    - Use the `--mapping` flag to provide a file with typos or words to find.
    - Use the `--add` flag to provide extra mapping pairs (for example, `--add teh:the`) or words to match directly on the command line.
    - The `--smart` flag allows for finding subwords within larger compound words.
  - **Example:** `python multitool.py scan . --add teh:the --smart`

- **`verify`**
  - **What it does:** Finds which entries in a mapping file or extra pairs are present in the provided input files. It provides a high-level summary of which typos were found and which were missing.
  - **Options:**
    - Use the `--mapping` flag to provide the file containing typos to check.
    - Use the `--add` flag to provide extra mapping pairs (for example, `--add teh:the`) or words to match directly on the command line.
    - Use the `--prune` flag to output a new mapping file containing only the typos that were actually found in your project.
    - Use the `--smart` flag to also find subword matches (for example, finding "teh" inside "tehWord").
  - **Example:** `python multitool.py verify . --add teh:the --prune`

## Common Options

These options work with most modes:

- `[INPUT_FILES...]`: One or more files to read. Defaults to **standard input** if not provided.
- `--output`: The file to write results to. Defaults to printing to the screen.
- `--output-format`: The format of the output. Options include `line` (default), `json`, `yaml`, `csv`, `markdown`, `md-table`, `arrow`, and `table`.
- `--min-length`: Skip items shorter than this length (default: 1 for most modes, 3 for word extraction modes like 'words' and 'count').
- `--max-length`: Skip words longer than this length (default: 1000).
- `--process-output`: Sorts the final list and removes duplicates. Use this to organize your output or remove redundant entries.
- `--limit`, `-L`: Limit the number of items in the output.
- `--raw`: Keep punctuation and capitalization. By default, most tools convert everything to lowercase and remove all characters except for lowercase **a through z**. Use this flag if you need to preserve numbers, punctuation, or capitalization.
- `--quiet`: Hide progress bars and log messages.
