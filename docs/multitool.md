# Multitool

**Multitool** is a versatile tool for processing text files. It can extract specific data (like columns from a CSV or text inside backticks), compare files, and clean up lists.

## Quick Start

Run the tool with a mode and your input files:

```bash
python multitool.py <MODE> [INPUT_FILES...] [OPTIONS]
```

Most modes default to reading from **standard input** if you do not specify an input file. This makes it easy to pipe data into Multitool.

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
  - **What it does:** Extracts columns from a CSV file. By default, it extracts **all columns except the first one**. Use `--first-column` to keep *only* the first column, or `--column` (or `-c`) followed by one or more numbers to extract specific columns. Use `--delimiter` (or `-d`) to specify a different column separator (for example, `;`).
  - **Example:** `python multitool.py csv data.csv --column 2`

- **`markdown`**
  - **What it does:** Extracts items from Markdown bulleted lists (lines starting with `- `, `* `, or `+ `). It can also split items by `:` or `->` to extract one side of a pair (use the `--right` flag for the second part).
  - **Example:** `python multitool.py markdown notes.md --right`

- **`md-table`**
  - **What it does:** Extracts text from Markdown tables. It saves the first column by default. Use the `--right` flag to save the second column instead, or `--column` (or `-c`) followed by one or more numbers to extract specific columns. It automatically skips header and divider rows.
  - **Example:** `python multitool.py md-table readme.md --column 2`

- **`json`**
  - **What it does:** Extracts values from a JSON file based on a specific key. You can use dots to access nested keys (for example, `user.name`). If you do not provide a key, it extracts keys from the top level object (or all items if the top level is a list). It automatically handles lists and objects.
  - **Example:** `python multitool.py json report.json --key replacements.typo`

- **`yaml`**
  - **What it does:** Extracts values from a YAML file based on a key path. Like JSON mode, it supports dot notation (for example, `config.items`). If you do not provide a key, it extracts keys from the top level object (or all items if the top level is a list).
  - **Example:** `python multitool.py yaml config.yaml --key config.items`

- **`line`**
  - **What it does:** Reads a file line by line. Use this to simply clean or filter a text file without special extraction logic.
  - **Example:** `python multitool.py line raw_words.txt`

- **`words`**
  - **What it does:** Extracts individual words from a file. It splits lines by whitespace by default, but you can specify a custom character using `--delimiter` (or `-d`). Use the `--smart` (or `-S`) flag to also split by symbols and capital letters (for example, splitting "CamelCase" into "Camel" and "Case").
  - **Example:** `python multitool.py words report.txt --smart`

- **`ngrams`**
  - **What it does:** Extracts sequences of N words from a file. This is useful for finding common phrases or context around typos. It supports sequences across line boundaries.
  - **Options:** Use `-n` to specify the number of words in each sequence (default is 2). Like the `words` mode, it supports custom delimiters and smart word splitting.
  - **Example:** `python multitool.py ngrams report.txt -n 2 --smart`

- **`regex`**
  - **What it does:** Extracts text matching a Python regular expression pattern. Unlike other modes, it **preserves exact text** (it does not convert to lowercase or remove punctuation) by default.
  - **Example:** `python multitool.py regex inputs.txt --pattern 'user_\w+'`

### 2. Manipulation Modes

These modes help you transform or combine your data.

- **`combine`**
  - **What it does:** Merges multiple files (or standard input) into one list, removes duplicates, and sorts the result alphabetically.
  - **Note:** This mode has built-in sorting and deduplication; the `--process-output` flag is not needed.
  - **Example:** `python multitool.py combine file1.txt file2.txt`

- **`unique`**
  - **What it does:** Removes duplicate items from your list while **preserving the original order**. This is useful when the sequence of items is important.
  - **Example:** `python multitool.py unique raw_typos.txt`

- **`resolve`**
  - **What it does:** Identifies and flattens chains of typo corrections. For example, if your mapping file contains `A -> B` and `B -> C`, this mode will resolve them to `A -> C` and `B -> C`. This ensures that your mappings always point directly to the final correct word, making them more efficient for scrubbing and analysis.
  - **Example:** `python multitool.py resolve mappings.csv`

- **`rename`**
  - **What it does:** Renames files and directories using a mapping file. It is useful for fixing typos in filenames across your entire project. It handles nested renames by processing files before their parent directories.
  - **Options:**
    - Supports `--in-place` renaming and `--dry-run` preview.
    - Use the `--smart-case` flag to automatically match the casing of the original filename.
  - **Example:** `python multitool.py rename src/**/* --mapping corrections.csv --in-place`

- **`diff`**
  - **What it does:** Identifies added, removed, and changed items between two files. It can compare simple lists of words or (with the `--pairs` flag) identify changes in typo-correction mappings.
  - **Supported Formats:** Color-coded terminal output (default) and structured JSON output.
  - **Example:** `python multitool.py diff old_list.txt new_list.txt`
  - **Pairs Example:** `python multitool.py diff old_typos.csv new_typos.csv --pairs`

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
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py zip typos.txt --file2 corrections.txt --output-format arrow`

- **`swap`**
  - **What it does:** Reverses the order of elements in paired data (for example, `typo -> correction` becomes `correction -> typo`).
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py swap mappings.csv --output-format arrow`

- **`scrub`**
  - **What it does:** Performs replacements of typos in your text files using a mapping file. It tries to preserve the surrounding context (punctuation, whitespace) while fixing errors. It automatically handles compound words like `CamelCase` and `snake_case` variables.
  - **Supported Formats:** Supports CSV, Arrow, Table, JSON, and YAML mapping formats.
  - **In-Place Editing:** Use the `--in-place` flag to modify files directly. If you provide an extension (for example, `--in-place .bak`), the tool will create a backup of each file before modifying it.
  - **Dry Run:** Use the `--dry-run` flag to see how many replacements would be made without actually changing any files.
  - **Smart Casing:** Use the `--smart-case` flag to automatically match the casing of the original word. For example, if the mapping is `teh -> the`, then `Teh` will be replaced with `The`, and `TEH` with `THE`.
  - **Example:** `python multitool.py scrub input.txt --mapping corrections.csv --output fixed.txt`
  - **In-Place Example:** `python multitool.py scrub file1.txt file2.txt --mapping corrections.csv --in-place`

- **`standardize`**
  - **What it does:** Fixes inconsistent casing by using the most frequent form. It analyzes your files to find words used with different capitalization (for example, 'database' vs 'Database'). It then automatically replaces all less frequent versions with the most popular one across the entire project. This ensures naming consistency without needing a manual mapping file.
  - **Options:**
    - Supports `--in-place` editing and `--dry-run` preview.
    - Works with standard filters like `--min-length` and `--max-length`.
  - **Example:** `python multitool.py standardize . --in-place --min-length 4`

- **`highlight`**
  - **What it does:** Searches for words from a list or mapping and highlights them with color in the output. This is useful as a non-destructive preview before using the `scrub` mode to make permanent changes.
  - **Options:** Use the `--mapping` flag to provide a file with typos or words to find. The `--smart` flag allows for highlighting subwords within larger compound words (like variable names).
  - **Example:** `python multitool.py highlight input.txt --mapping corrections.csv`

- **`pairs`**
  - **What it does:** Processes paired data (like `typo -> correction`) from any supported format and writes it to the specified output format. This is the primary way to convert between paired formats (for example, from JSON to CSV) while applying cleaning and length filters.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py pairs typos.json --output-format csv`

### 3. Analysis Modes

These modes help you analyze your data.

- **`check`**
  - **What it does:** Finds words that appear as both a typo *and* a correction. This is useful for spotting errors in your typo database (loops).
  - **Example:** `python multitool.py check mappings.csv`

- **`classify`**
  - **What it does:** Categorizes typo corrections based on their error type. It labels each pair with a code like `[K]` (Keyboard), `[T]` (Transposition), `[D]` (Deletion), `[I]` (Insertion), `[R]` (Replacement), or `[M]` (Multiple letters).
  - **Options:** Use `--show-dist` to include the number of character changes in the output labels.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py classify typos.txt --show-dist --output labeled.txt`

- **`conflict`**
  - **What it does:** Identifies typos that are associated with more than one unique correction. Use this to find inconsistencies in your typo lists.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py conflict mappings.csv`

- **`count`**
  - **What it does:** Counts how many times each word appears in a file and sorts them by frequency (most frequent first).
  - **Options:**
    - `--min-count` and `--max-count`: Filter results by their frequency.
    - `-d`, `--delimiter`: The character to split words by (default: whitespace).
    - `-S`, `--smart`: Split by symbols and capital letters (for example, splitting "CamelCase" into "Camel" and "Case").
    - `-p`, `--pairs`: Count frequencies of word pairs (for example, `typo -> correction`) instead of single words.
  - **Visual Report:** Use `--output-format arrow` to generate a rich visual report. This includes an **ANALYSIS SUMMARY** dashboard with metrics like retention rate, an aligned frequency table, and high-resolution bar charts.
  - **Supported Formats:** `arrow`, `json`, `csv`, `markdown`, `md-table`, and `line`.
  - **Note:** This mode has built-in sorting; the `--process-output` flag is not needed.
  - **Example:** `python multitool.py count all_typos.txt --min-count 5 -f arrow --smart`
  - **Pairs Example:** `python multitool.py count typos.log --pairs --output-format arrow`

- **`cycles`**
  - **What it does:** Identifies loops in typo-correction pairs (for example, "A" maps to "B" and "B" maps back to "A"). These repeated loops can cause issues when automatically fixing text and often indicate errors in your data.
  - **Example:** `python multitool.py cycles typos.csv --output-format arrow`

- **`fuzzymatch`**
  - **What it does:** Identifies words in your list that are similar to words in a second list (dictionary). Use this to find likely corrections for typos.
  - **Options:** Use `--min-dist` and `--max-dist` to control the number of changes allowed, and `--show-dist` to see the number of changes in the output.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py fuzzymatch typos.txt dictionary.txt --max-dist 1 --show-dist`

- **`near_duplicates`**
  - **What it does:** Identifies pairs of words in your list that are very similar (only a few characters are different). This is useful for finding potential typos or unintended duplicates.
  - **Options:** Use `--min-dist` and `--max-dist` to control the number of changes allowed, and `--show-dist` to see the number of changes in the output.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py near_duplicates words.txt --max-dist 1 --show-dist`

- **`similarity`**
  - **What it does:** Filters paired data (like `typo -> correction`) based on the number of character changes needed to turn one word into another. Use this to remove extra data or find specific types of typos.
  - **Options:** Use `--min-dist` and `--max-dist` to set the range of allowed changes, and `--show-dist` to include the number of changes in the output.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, and `yaml`.
  - **Example:** `python multitool.py similarity typos.txt --max-dist 2 --show-dist`

- **`stats`**
  - **What it does:** Provides a high-level summary of your dataset. It reports counts, unique items, and statistics. If the `--pairs` flag is used, it additionally analyzes the file as paired data (typos/corrections) and reports conflicts (one typo to multiple corrections), overlaps (words that are both typos and corrections), and character change statistics.
  - **Supported Formats:** `json`, `yaml`, `markdown`, `md-table`, and `line` (human-readable).
  - **Example:** `python multitool.py stats typos.csv --pairs`

- **`discovery`**
  - **What it does:** Automatically finds potential typos in a text by identifying rare words that are very similar to frequent words. It assumes that frequent words are likely correct and rare variations are likely typos. This is a powerful way to find errors in a dataset without needing a dictionary.
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
  - **What it does:** Identifies words that appear in your files with multiple different casing styles (for example, 'hello', 'Hello', 'HELLO'). This is useful for identifying inconsistent naming or typos that differ only by case.
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
  - **What it does:** Like a batch version of the `search` mode. It searches for every word in a mapping file or list and reports all matches with filename, line number, and highlighting. This is the recommended way to audit your project for a set of known typos before performing replacements.
  - **Options:** Use the `--mapping` flag to provide a file with typos or words to find. The `--smart` flag allows for finding subwords within larger compound words.
  - **Example:** `python multitool.py scan . --mapping typos.csv --smart`

## Common Options

These options work with most modes:

- `[INPUT_FILES...]`: One or more files to read. Defaults to **standard input** if omitted.
- `--output`: The file to write results to. Defaults to printing to the screen.
- `--output-format`: The format of the output. Options include `line` (default), `json`, `yaml`, `csv`, `markdown`, `md-table`, `arrow`, and `table`.
- `--min-length`: Skip words shorter than this length (default: 3).
- `--max-length`: Skip words longer than this length (default: 1000).
- `--process-output`: Sorts the final list and removes duplicates. Use this to organize your output or remove redundant entries.
- `--limit`, `-L`: Limit the number of items in the output.
- `--raw`: Keep punctuation and capitalization. By default, most tools convert everything to lowercase and remove all characters except for lowercase **a through z**. Use this flag if you need to preserve numbers, punctuation, or capitalization.
- `--quiet`: Hide progress bars and log messages.
