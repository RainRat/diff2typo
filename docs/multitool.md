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

### GET DATA

These modes help you pull specific data out of a messy file.

- **`arrow`**
  - **What it does:** Extracts text from arrow lines (for example, `typo -> correction`). Useful for getting a clean list of typos from a log. It saves the left side by default. You can also get the right side (the correction) by adding the `--right` flag.
  - **Example:** `python multitool.py arrow typos.log --right`

- **`table`**
  - **What it does:** Extracts text from key=value entries like `key = "value"`. It saves the key by default. Use the `--right` flag to get the quoted value instead.
  - **Example:** `python multitool.py table typos.toml --right`

- **`backtick`**
  - **What it does:** Extracts text inside backticks (for example, \`code\`). It picks items near words like 'error' or 'warning' to find the most useful data.
  - **Example:** `python multitool.py backtick build.log`

- **`quoted`**
  - **What it does:** Extracts text inside quotes. Finds text inside double (`"`) or single (`'`) quotes. It handles backslash escaping (like `\"` or `\'`) to correctly extract strings from code or data files.
  - **Example:** `python multitool.py quoted source.py`

- **`between`**
  - **What it does:** Extracts text between markers. Finds text between a starting marker and an ending marker. It is useful for extracting data from templating languages, logs, or custom file formats. You can use the `--multi-line` flag to capture content that spans multiple lines.
  - **Options:** Requires `--start` and `--end` to define the markers.
  - **Example:** `python multitool.py between input.txt --start '{{' --end '}}'`

- **`csv`**
  - **What it does:** Extracts columns from CSV. Gets data from CSV files. By default, it picks **every column except the first one**. Use `--first-column` to get *only* the first column, or `--column` (or `-c`) followed by one or more numbers to get specific columns. Use `--delimiter` (or `-d`) to pick a different column separator (for example, `;`).
  - **Example:** `python multitool.py csv data.csv --column 2`

- **`markdown`**
  - **What it does:** Extracts Markdown list items. Finds text in lines starting with `- `, `* `, or `+ `. It can also split items by `:` or `->` to get one side of a pair (use the `--right` flag for the second part).
  - **Example:** `python multitool.py markdown notes.md --right`

- **`md-table`**
  - **What it does:** Extracts Markdown table items. Finds text in cells of a Markdown table. It saves the first column by default. Use the `--right` flag to save the second column instead, or `--column` (or `-c`) followed by one or more numbers to get specific columns. It automatically skips header and divider rows.
  - **Example:** `python multitool.py md-table readme.md --column 2`

- **`json`**
  - **What it does:** Extracts JSON values by key. Finds values for a specific key in a JSON file. You can use dots for keys inside other keys (for example, `user.name`). If you do not provide a key, it gets items from the top level. It automatically handles lists and objects.
  - **Example:** `python multitool.py json report.json --key replacements.typo`

- **`yaml`**
  - **What it does:** Extracts YAML values by key. Finds values for a specific key in a YAML file using a key path. Like JSON mode, it supports dot notation (for example, `config.items`). If you do not provide a key, it gets items from the top level.
  - **Example:** `python multitool.py yaml config.yaml --key config.items`

- **`toml`**
  - **What it does:** Extracts TOML values by key. Finds values for a specific key in a TOML file using a key path. Like JSON and YAML modes, it supports dot notation (for example, `tool.poetry.dependencies`). If you do not provide a key, it gets items from the top level. It automatically handles nested tables.
  - **Example:** `python multitool.py toml pyproject.toml --key tool.poetry.dependencies`

- **`line`**
  - **What it does:** Extracts every line from a file. Use this to simply clean or filter a text file without special getting logic.
  - **Example:** `python multitool.py line raw_words.txt`

- **`words`**
  - **What it does:** Extracts words from a file. Splits a file into individual words using whitespace or a custom delimiter. It's the standard way to get a list of every word used in a document. Use the `--smart` (or `-S`) flag to also split by symbols and capital letters (for example, splitting "CamelCase" into "Camel" and "Case").
  - **Example:** `python multitool.py words report.txt --smart`

- **`ngrams`**
  - **What it does:** Extracts sequences of words from a file. This is useful for finding common phrases or context around typos. It supports sequences across line boundaries.
  - **Options:** Use `-n` to pick the number of words in each sequence (default is 2). Like the `words` mode, it supports custom delimiters and smart word splitting.
  - **Example:** `python multitool.py ngrams report.txt -n 2 --smart`

- **`regex`**
  - **What it does:** Extracts text matching a pattern. Finds and gets all text that matches a Python regular expression pattern. Unlike other modes, it **keeps the original text** (it does not convert to lowercase or remove punctuation) by default.
  - **Example:** `python multitool.py regex inputs.txt --pattern 'user_\w+'`

### CHANGE DATA

These modes help you transform or combine your data.

- **`combine`**
  - **What it does:** Merges multiple files into one. Merges multiple files (or standard input) into one list, removes duplicates, and sorts the result alphabetically.
  - **Note:** This mode has built-in sorting and deduplication; the `--process-output` flag is not needed.
  - **Example:** `python multitool.py combine file1.txt file2.txt`

- **`unique`**
  - **What it does:** Removes duplicates, keeps order. Removes duplicate items from your list while **keeping the original order**. This is useful when the sequence of items is important.
  - **Example:** `python multitool.py unique raw_typos.txt`

- **`resolve`**
  - **What it does:** Shortens typo correction chains. Finds and shortens chains of corrections (for example, 'A' -> 'B' and 'B' -> 'C' becomes 'A' -> 'C'). This ensures that your mapping files always point directly to the final correct word, which improves the efficiency of fixing typos and analysis.
  - **Example:** `python multitool.py resolve mappings.csv`

- **`align`**
  - **What it does:** Aligns typo-correction pairs. Extracts typo-correction pairs from any supported format (CSV, Arrow, Markdown lists/tables) and outputs them in perfectly aligned columns by automatically calculating the maximum width of the left column. This is the recommended way to "beautify" your typo lists for human readability.
  - **Options:** Use the `--sep` flag to customize the separator string between columns (default is ` -> `).
  - **Example:** `python multitool.py align typos.csv --sep ' | '`

- **`rename`**
  - **What it does:** Batch renames files and folders. Renames files or directories based on a typo mapping or extra pairs provided via --add. It is useful for fixing typos in filenames across your entire project. It handles nested renames by processing files before their parent folders.
  - **Options:**
    - Supports `--in-place` renaming and `--dry-run` preview.
    - Use the `--add` flag to provide extra mapping pairs (for example, `--add old_name:new_name`) directly on the command line.
    - Use the `--smart-case` flag to automatically match the casing of the original filename.
  - **Example:** `python multitool.py rename src/ --mapping corrections.csv --in-place`

- **`diff`**
  - **What it does:** Shows differences between files. Finds added, removed, and changed items between two files. It can compare simple lists of words or (with the `--pairs` flag) find changes in typo-correction mappings.
  - **Supported Formats:** Color-coded terminal output (default) and structured JSON output.
  - **Example:** `python multitool.py diff old_list.txt new_list.txt`
  - **Pairs Example:** `python multitool.py diff old_typos.csv new_typos.csv --pairs`

- **`filterfragments`**
  - **What it does:** Removes words found inside others. Removes words from your input file if they appear anywhere inside a second file (`--file2`).
  - **Example:** `python multitool.py filterfragments candidates.txt --file2 dictionary.txt`

- **`map`**
  - **What it does:** Replaces items using a mapping file or extra pairs provided via --add. Supports CSV, Arrow, Table (`typo = "correction"`), JSON, YAML, and TOML formats. By default, it keeps items that are not in the mapping. The `--min-length` and `--max-length` filters are **re-applied** to items after they are changed. Use `--drop-missing` to remove unmatched items.
  - **Options:**
    - Use the `--add` flag to provide extra mapping pairs (for example, `--add teh:the`) directly on the command line.
  - **Example:** `python multitool.py map input.txt --add teh:the`

- **`sample`**
  - **What it does:** Picks a random set of lines. Picks a random set of lines from a file. You can pick a specific number (`--n 100`) or a percentage (`--percent 10`).
  - **Example:** `python multitool.py sample big_log.txt --n 50`

- **`set_operation`**
  - **What it does:** Compares files using set logic. Compares two files using standard set logic:
    - `intersection`: Finds lines common to both files.
    - `union`: Combines all lines from both files.
    - `difference`: Finds lines in the first file that are not in the second.
    - `symmetric_difference`: Finds lines that are unique to each file (items in either file, but not both).
  - **Example:** `python multitool.py set_operation a.txt --file2 b.txt --operation symmetric_difference`

- **`zip`**
  - **What it does:** Pairs lines from two files. Joins two files line-by-line into a paired format. It applies `--min-length` and `--max-length` filters to **both items in each pair**. If the files have a different number of lines, the output will stop at the end of the shortest file.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, and `toml`.
  - **Example:** `python multitool.py zip typos.txt --file2 corrections.txt --output-format arrow`

- **`swap`**
  - **What it does:** Reverses the order of pairs (for example, `typo -> correction` becomes `correction -> typo`).
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, and `toml`.
  - **Example:** `python multitool.py swap mappings.csv --output-format arrow`

- **`scrub`**
  - **What it does:** Fixes typos in text files using a mapping file or extra pairs provided via --add. It tries to preserve the surrounding context (punctuation, whitespace) while fixing errors. It automatically handles compound words like 'CamelCase' and 'snake_case' variables.
  - **Supported Formats:** CSV, Arrow, Table, JSON, YAML, and TOML mapping formats.
  - **Options:**
    - Use the `--add` flag to provide extra mapping pairs (for example, `--add teh:the`) directly on the command line.
  - **In-Place Editing:** Use the `--in-place` flag to modify files directly. If you provide an extension (for example, `--in-place .bak`), the tool will create a backup of each file before modifying it.
  - **Dry Run:** Use the `--dry-run` flag to see how many fixes would be made without actually changing any files.
  - **Diff Preview:** Use the `--diff` flag to see a unified diff of the changes that would be made. This is useful for reviewing fixes before applying them in-place.
  - **Smart Casing:** Use the `--smart-case` flag to automatically match the casing of the original word. For example, if the mapping is `teh -> the`, then `Teh` will be replaced with `The`, and `TEH` with `THE`.
  - **Example:** `python multitool.py scrub input.txt --add teh:the --diff`
  - **In-Place Example:** `python multitool.py scrub file1.txt file2.txt --mapping corrections.csv --in-place`

- **`standardize`**
  - **What it does:** Fixes casing/spelling project-wide. Analyzes your files to find words used with different capitalization (for example, 'database' vs 'Database') or similar spelling (for example, 'teh' vs 'the'). It then automatically replaces all less frequent versions with the most popular one across the entire project. This ensures consistency without needing a manual mapping file.
  - **Options:**
    - Supports `--in-place` editing and `--dry-run` preview.
    - **Similar Word Matching:** Use `--fuzzy` to set the maximum character distance for matching similar words.
    - **Frequency Ratio:** Use `--threshold` to set the minimum frequency ratio required to consider a rare word a typo (default: 10.0).
    - **Diff Preview:** Use the `--diff` flag to see a unified diff of the changes that would be made.
    - Works with standard filters like `--min-length` and `--max-length`.
  - **Example:** `python multitool.py standardize . --diff --min-length 4 --fuzzy 1`

- **`highlight`**
  - **What it does:** Color-codes words from a list. Searches for words from a mapping file or extra pairs provided via --add and highlights them with color in the output. This is useful as a non-destructive preview before using 'scrub'.
  - **Options:**
    - Use the `--mapping` flag to provide a file with typos or words to find.
    - Use the `--add` flag to provide extra mapping pairs (for example, `--add teh:the`) or words to match directly on the command line.
    - The `--smart` flag allows for coloring subwords within larger compound words (like variable names).
  - **Example:** `python multitool.py highlight input.txt --add teh:the`

- **`pairs`**
  - **What it does:** Converts paired data formats. Reads pairs (like 'typo -> correction') from any supported format and writes them to the specified output format. Useful for cleaning, filtering, and format conversion.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, and `toml`.
  - **Example:** `python multitool.py pairs typos.json --output-format csv`

### CHECK & ANALYZE

These modes help you analyze your data.

- **`check`**
  - **What it does:** Finds words used as both typos and fixes. Checks for words that appear in both the typo and correction columns of a file. Use this to find errors in your typo lists.
  - **Example:** `python multitool.py check mappings.csv`

- **`classify`**
  - **What it does:** Groups typos by error type. Labels typo pairs with error codes like [K] Keyboard, [T] Transposition, [D] Deletion, [I] Insertion, [R] Replacement, and [M] Multiple letters.
  - **Options:** Use `--show-dist` to include the number of character changes in the output labels.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, and `toml`.
  - **Example:** `python multitool.py classify typos.txt --show-dist --output labeled.txt`

- **`conflict`**
  - **What it does:** Finds typos with multiple fixes. Finds typos in your paired data that are associated with more than one unique correction. Use this to find inconsistencies in your typo lists.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, and `toml`.
  - **Example:** `python multitool.py conflict mappings.csv`

- **`count`**
  - **What it does:** Counts how often items appear. Counts how often each word, pair, line, or character appears and sorts the list by frequency.
  - **Options:**
    - `--min-count` and `--max-count`: Filter results by their frequency.
    - `-d`, `--delimiter`: The character to split words by (default: whitespace).
    - `-S`, `--smart`: Split by symbols and capital letters (for example, splitting "CamelCase" into "Camel" and "Case").
    - `-p`, `--pairs`: Count frequencies of word pairs (for example, `typo -> correction`) instead of single words.
    - `-l`, `--lines`: Count frequencies of raw lines instead of individual words.
    - `-c`, `--chars`: Count frequencies of individual characters.
    - `-B`, `--by-file`: Count how many files contain each item instead of total matches. This is useful for finding words that are common across your entire project.
    - **Checking:** Use the `--mapping` flag or `--add` to count matches of specific typos across your files.
  - **Visual Report:** Use `--output-format arrow` to generate a rich visual report. This includes an **ANALYSIS SUMMARY** dashboard with metrics like retention rate, an aligned frequency table, and high-resolution bar charts.
  - **Supported Formats:** `arrow`, `json`, `csv`, `markdown`, `md-table`, and `line`.
  - **Note:** This mode has built-in sorting; the `--process-output` flag is not needed.
  - **Example:** `python multitool.py count all_typos.txt --min-count 5 -f arrow --smart`
  - **By-File Example:** `python multitool.py count src/*.py --by-file --output-format arrow`
  - **Pairs Example:** `python multitool.py count typos.log --pairs --output-format arrow`

- **`cycles`**
  - **What it does:** Finds loops in typo pairs (for example, "A" maps to "B" and "B" maps back to "A"). Repeated loops can cause issues when automatically fixing typos and represent logic errors in your data.
  - **Example:** `python multitool.py cycles typos.csv --output-format arrow`

- **`fuzzymatch`**
  - **What it does:** Finds similar words in two lists. Finds words in your list that are similar to words in a second list (large dictionary). Use this to find likely corrections for typos.
  - **Options:** Use `--min-dist` and `--max-dist` to control the number of changes allowed, and `--show-dist` to see the number of changes in the output.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, and `toml`.
  - **Example:** `python multitool.py fuzzymatch typos.txt words.csv --max-dist 1 --show-dist`

- **`near_duplicates`**
  - **What it does:** Finds similar words in one list. Finds pairs of words in your list that are very similar (only a few characters are different). Use this to find potential typos or unintended duplicates in a project.
  - **Options:** Use `--min-dist` and `--max-dist` to control the number of changes allowed, and `--show-dist` to see the number of changes in the output.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, and `toml`.
  - **Example:** `python multitool.py near_duplicates words.txt --max-dist 1 --show-dist`

- **`similarity`**
  - **What it does:** Filters pairs by changes. Filters pairs (typo -> correction) based on the number of character changes needed to turn one word into another. Use this to remove extra data or find specific types of typos.
  - **Options:** Use `--min-dist` and `--max-dist` to set the range of allowed changes, and `--show-dist` to include the number of changes in the output.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, and `toml`.
  - **Example:** `python multitool.py similarity typos.txt --max-dist 2 --show-dist`

- **`stats`**
  - **What it does:** Shows statistics for a list. Provides a detailed overview of your dataset. It reports counts, unique items, statistics, and (optionally) paired data stats like conflicts, overlaps, and the number of changes between words.
  - **Supported Formats:** `json`, `yaml`, `markdown`, `md-table`, and `line` (human-readable).
  - **Example:** `python multitool.py stats typos.csv --pairs`

- **`discovery`**
  - **What it does:** Finds typos in rare words. Automatically finds potential typos in a text by seeing rare words that are very similar to frequent words. It assumes that frequent words are likely correct and rare variations are likely typos. This is a powerful way to find errors without needing a dictionary.
  - **Options:**
    - `--rare-max`: Maximum frequency for a word to be considered a potential typo (default: 1).
    - `--freq-min`: Minimum frequency for a word to be considered a potential correction (default: 5).
    - `--min-dist` and `--max-dist`: Control the number of allowed character changes between the typo and the correction.
    - `--show-dist`: Include the number of character changes in the output.
    - `-d`, `--delimiter`: The character to split words by (default: whitespace).
    - `-S`, `--smart`: Split by symbols and capital letters (for example, splitting "CamelCase" into "Camel" and "Case").
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, and `toml`.
  - **Example:** `python multitool.py discovery code.py --smart --rare-max 2 --freq-min 10 --max-dist 1`

- **`casing`**
  - **What it does:** Finds inconsistent casing. Finds words that appear in your files with multiple different casing styles (for example, 'hello', 'Hello', 'HELLO'). This is useful for seeing inconsistent naming or typos that differ only by case.
  - **Options:**
    - `-d`, `--delimiter`: The character to split words by (default: whitespace).
    - `-S`, `--smart`: Split by symbols and capital letters (for example, splitting "CamelCase" into "Camel" and "Case").
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, and `toml`.
  - **Example:** `python multitool.py casing report.txt --smart --output-format arrow`

- **`repeated`**
  - **What it does:** Finds doubled words. Finds doubled words (for example, 'the the') in your text. It outputs the duplicated pair and the suggested fix. Use --smart to handle CamelCase or punctuation.
  - **Options:**
    - `-d`, `--delimiter`: The character to split words by (default: whitespace).
    - `-S`, `--smart`: Split by symbols and capital letters (for example, splitting "CamelCase" into "Camel" and "Case").
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, and `toml`.
  - **Example:** `python multitool.py repeated report.txt --smart --output-format arrow`

- **`search`**
  - **What it does:** Searches for words or patterns. A typo-aware search tool. It searches for a query in your files and can find similar words (typos) or subword matches. It supports highlighting, line numbers, and context lines.
  - **Options:**
    - `-Q`, `--query`: The word or pattern to search for.
    - `--max-dist`: Maximum number of character changes for similar word matching (default: 0).
    - `-S`, `--smart`: Search for subwords within larger items (for example, finding "teh" inside "tehWord").
    - `--line-numbers`: Show the filename and line number for each match.
  - **Example:** `python multitool.py search report.txt -Q 'teh' --max-dist 1 --line-numbers`

- **`scan`**
  - **What it does:** Scans project for known typos. Like a batch version of the 'search' mode. It searches for every word in a mapping file or provided via --add and reports all matches with filename, line number, and highlighting. It also supports context lines.
  - **Options:**
    - Use the `--mapping` flag to provide a file with typos or words to find.
    - Use the `--add` flag to provide extra mapping pairs (for example, `--add teh:the`) or words to match directly on the command line.
    - The `--smart` flag allows for finding subwords within larger compound words.
  - **Example:** `python multitool.py scan . --add teh:the --smart`

- **`verify`**
  - **What it does:** Checks if typos exist in project. Finds which entries in a mapping file or extra pairs are present in the provided input files. It provides a high-level summary of which typos were found and which were missing.
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
- `--output-format`: The format of the output. Options include `line` (default), `json`, `yaml`, `toml`, `csv`, `markdown`, `md-table`, `arrow`, and `table`.
- `--min-length`: Skip items shorter than this length (default: 1 for most modes, 3 for word extraction modes like 'words' and 'count').
- `--max-length`: Skip words longer than this length (default: 1000).
- `--process-output`: Sorts the final list and removes duplicates. Use this to organize your output or remove redundant entries.
- `--limit`, `-L`: Limit the number of items in the output.
- `--raw`: Keep punctuation and capitalization. By default, most tools convert everything to lowercase and remove all characters except for lowercase **a through z**. Use this flag if you need to preserve numbers, punctuation, or capitalization.
- `--quiet`: Hide progress bars and log messages.
