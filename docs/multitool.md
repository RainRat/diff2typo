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

### Input Files and Directories

You can provide one or more files or directories as input. If you provide a directory, Multitool will automatically find and process every file inside it recursively.

To keep things fast and clean, the tool automatically ignores common system and environment folders, including:
- **`.git`**
- **`node_modules`**
- **`venv`** and **`.venv`**
- **`.pytest_cache`** and **`.ruff_cache`**
- **`.vscode`** and **`.idea`**
- **`__pycache__`**
- **`dist`** and **`build`**

## Modes

Multitool operates in different "modes," each designed for a specific task.

### GET DATA

Use these modes to pull specific data from a file.

- **`arrow`**
  - Extracts text from lines using arrows (for example, `typo -> correction`). It saves the left side by default. Use the `--right` flag to get the correction instead.
  - **Example:** `python multitool.py arrow typos.log --right`

- **`table`**
  - Extracts keys or values from entries like `key = "value"`. It saves the key by default. Use the `--right` flag to get the quoted value instead.
  - **Example:** `python multitool.py table typos.toml --right`

- **`backtick`**
  - Extracts text found inside backticks (for example, \`code\`). It prioritizes items near words like 'error' or 'warning'.
  - **Example:** `python multitool.py backtick build.log`

- **`quoted`**
  - Extracts text found inside double (`"`) or single (`'`) quotes. It handles backslash escaping (like `\"` or `\'`) to correctly extract strings from code or data files.
  - **Example:** `python multitool.py quoted source.py`

- **`between`**
  - Extracts text between markers. This is useful for templating languages, logs, or custom formats. Use the `--multi-line` flag to capture content that spans multiple lines.
  - **Options:** Requires `--start` and `--end` markers.
  - **Example:** `python multitool.py between input.txt --start '{{' --end '}}'`

- **`csv`**
  - Extracts columns from CSV files. By default, it picks **every column except the first one**. Use `--first-column` to get *only* the first column, or `--column` (or `-c`) to pick specific columns (starting from 0). Use `--delimiter` (or `-d`) to set a different separator (for example, `;`).
  - **Example:** `python multitool.py csv data.csv --column 1  # Get the second column`

- **`markdown`**
  - Extracts Markdown list items (lines starting with `-`, `*`, or `+`). You can split items by `:` or `->` to get one side of a pair (use the `--right` flag for the second part).
  - **Example:** `python multitool.py markdown notes.md --right`

- **`frontmatter`**
  - Extracts YAML frontmatter from Markdown files (text between '---' delimiters at the start of the file). Use dots for nested keys (like 'metadata.tags'). If you don't provide a key, it gets items from the top level.
  - **Example:** `python multitool.py frontmatter post.md --key 'tags'`

- **`md-table`**
  - Extracts text from cells in a Markdown table. It saves the first column by default. Use the `--right` flag for the second column, or `--column` (or `-c`) to pick specific columns (starting from 0). It automatically skips header and divider rows.
  - **Example:** `python multitool.py md-table readme.md --column 1  # Get the second column`

- **`headings`**
  - Extracts headings from Markdown files (lines starting with `#`). It saves the heading text by default.
  - **Options:** Use `--level` (1-6) to filter by a specific heading level. Use `--pairs` (or `-p`) to see both the level and the text.
  - **Example:** `python multitool.py headings readme.md --level 1`

- **`toc`**
  - Creates a clickable, nested Table of Contents from Markdown headings. It handles duplicate headings by adding numeric suffixes.
  - **Options:** Use `--level` (1-6) to filter by a specific heading level. Use `--no-links` to generate a simple indented list without clickable links.
  - **Example:** `python multitool.py toc readme.md --level 2`

- **`links`**
  - Extracts links (`[text](url)`) and images (`![alt](url)`) from Markdown files. It saves the link text by default.
  - **Options:** Use `--right` to save the URL instead, or `--pairs` (or `-p`) to see both.
  - **Example:** `python multitool.py links readme.md --right`

- **`codeblocks`**
  - Extracts fenced code blocks from Markdown files (lines starting with \` \` \` or `~~~`). It saves the code content by default.
  - **Options:** Use `--language` (or `-l`) to filter by a specific language (for example, `python`). Use `--pairs` (or `-p`) to see both the language and the code content.
  - **Example:** `python multitool.py codeblocks readme.md --language python`

- **`comments`**
  - Extracts comments from source files. It identifies single-line comments (`#`, `//`, `--`) and multi-line comments (`/* */`, `<!-- -->`, and triple quotes) in various programming and markup languages.
  - **Example:** `python multitool.py comments src/ --output comments.txt`

- **`json`**
  - Extracts values from a JSON file by key. Use dot notation for nested keys (for example, `user.name`). If you do not provide a key, it extracts items from the top level. It automatically handles lists and objects.
  - **Example:** `python multitool.py json report.json --key replacements.typo`

- **`yaml`**
  - Extracts values from a YAML file by key. Like JSON mode, it supports dot notation (for example, `config.items`). If you do not provide a key, it extracts items from the top level.
  - **Example:** `python multitool.py yaml config.yaml --key config.items`

- **`toml`**
  - Extracts values from a TOML file by key. Like JSON and YAML modes, it supports dot notation (for example, `tool.poetry.dependencies`). If you do not provide a key, it extracts items from the top level.
  - **Example:** `python multitool.py toml pyproject.toml --key tool.poetry.dependencies`

- **`xml`**
  - Extracts values from XML files using a tag name or XPath expression. If you don't provide a key, it extracts text from every element.
  - **Example:** `python multitool.py xml data.xml -k './/item/name'`

- **`paths`**
  - Extracts path components (basename, dirname, extension) from file and directory paths. It also supports smart splitting to find words within filenames.
  - **Options:** Use `--basename`, `--dirname`, or `--extension` to pick specific parts. Use `--smart` to split components into words.
  - **Example:** `python multitool.py paths src/ --basename --smart`

- **`flatten`**
  - Transforms nested JSON, YAML, or TOML structures into dot-separated `key.path = value` pairs. It supports multi-document YAML and JSON Lines (JSONL). The default output format is `arrow`.
  - **Options:** Use the `-k` (or `--key`) flag to set an optional starting path.
  - **Example:** `python multitool.py flatten config.json --output-format table`

- **`line`**
  - Extracts every line from a file. It reads every line, cleans the text, and writes it to the output.
  - **Example:** `python multitool.py line raw_words.txt`

- **`words`**
  - Extracts individual words from a file using whitespace or a custom delimiter. Use the `--smart` (or `-S`) flag to also split by symbols and capital letters (for example, "CamelCase" becomes "Camel" and "Case").
  - **Example:** `python multitool.py words report.txt --smart`

- **`sentences`**
  - Extracts individual sentences from a file using a regex-based heuristic. It handles multi-line sentences by joining lines with spaces before splitting.
  - **Example:** `python multitool.py sentences report.txt --output sentences.txt`

- **`paragraphs`**
  - Extracts blocks of text separated by one or more blank lines. It automatically joins multi-line blocks into single lines and cleans up extra whitespace.
  - **Example:** `python multitool.py paragraphs report.txt --output paragraphs.txt`

- **`ngrams`**
  - Extracts sequences of words. This is useful for finding phrases or context around typos. It supports sequences across line boundaries.
  - **Options:** Use `-n` to pick the number of words in each sequence (default is 2). Like the `words` mode, it supports custom delimiters and smart word splitting.
  - **Example:** `python multitool.py ngrams report.txt -n 2 --smart`

- **`regex`**
  - Extracts text matching a Python regular expression pattern. Unlike other modes, it **keeps the original text** (no lowercase or punctuation removal) by default.
  - **Example:** `python multitool.py regex inputs.txt --pattern 'user_\w+'`

### CHANGE DATA

Use these modes to transform or combine your data.

- **`combine`**
  - Merges multiple files into one list, removes duplicates, and sorts the results alphabetically.
  - **Note:** This mode has built-in sorting and deduplication; the `--process-output` flag is not needed.
  - **Example:** `python multitool.py combine file1.txt file2.txt`

- **`unique`**
  - Removes duplicate items from your list while **keeping the original order**.
  - **Example:** `python multitool.py unique raw_typos.txt`

- **`sort`**
  - Sorts items in a list by alphabetical order, length, or numeric value. It supports reverse sorting and deduplication.
  - **Options:**
    - Use `--by` to choose the sorting method: `alpha` (alphabetical, default), `length` (string length), or `numeric` (numeric value).
    - Use `--reverse` to sort in descending order.
    - Use the `-u` (or `--unique`) flag to remove duplicates before sorting.
  - **Note:** For numeric sorting, the tool extracts the first number found in each item. Numeric sorting works best with the `--raw` flag to prevent digits from being stripped.
  - **Example:** `python multitool.py sort wordlist.txt --by length --reverse`

- **`resolve`**
  - Shortens typo correction chains. For example, if your mapping file contains `A -> B` and `B -> C`, this mode resolves them to `A -> C` and `B -> C`.
  - **Example:** `python multitool.py resolve mappings.csv`

- **`align`**
  - Aligns typo-correction pairs into perfectly aligned columns.
  - **Options:** Use the `--sep` flag to customize the separator string (default is ` -> `).
  - **Example:** `python multitool.py align typos.csv --sep ' | '`

- **`rename`**
  - Batch renames files and folders using a mapping file or extra pairs. It handles nested renames by processing files before their parent folders.
  - **Options:**
    - Supports `--in-place` renaming and `--dry-run` preview.
    - Use the `--add` flag to provide extra mapping pairs (for example, `--add old_name:new_name`) directly on the command line.
    - Use the `--smart-case` flag to match the casing of the original filename.
    - Use the `--regex` (or **`-r`**) flag to treat patterns as regular expressions. This supports capturing groups and backreferences (for example, `\1`).
  - **Example:** `python multitool.py rename . --regex --add 'test_(.*)\.py:spec_\1.py' --dry-run`

- **`diff`**
  - Shows differences between two files, including added, removed, and changed items.
  - **Supported Formats:** Color-coded terminal output (default) and structured JSON output.
  - **Example:** `python multitool.py diff old_list.txt new_list.txt`
  - **Pairs Example:** `python multitool.py diff old_typos.csv new_typos.csv --pairs`

- **`filterfragments`**
  - Removes words from your list if they appear as a fragment inside words in a second file.
  - **Example:** `python multitool.py filterfragments candidates.txt --file2 dictionary.txt`

- **`map`**
  - Replaces items in your list using a mapping file or extra pairs. By default, it keeps items that are not in the mapping.
  - **Options:**
    - Use the `--add` flag to provide extra mapping pairs directly on the command line.
  - **Example:** `python multitool.py map input.txt --add teh:the`

- **`case`**
  - Changes word casing to styles like `snake_case`, `camelCase`, or `PascalCase`. It identifies sub-words within compound words and preserves the overall structure.
  - **Options:**
    - Use the `--to` flag to choose the target style: `lower`, `upper`, `snake`, `camel`, `pascal`, `kebab`, `title`, `constant`, or `sentence`.
    - Use the `--pairs` (or `-p`) flag to see both the original and converted words.
  - **Example:** `python multitool.py case wordlist.txt --to snake --pairs`

- **`sample`**
  - Picks a random set of lines. You can pick a specific number (`--n 100`) or a percentage (`--percent 10`).
  - **Example:** `python multitool.py sample big_log.txt --n 50`

- **`shuffle`**
  - Randomly shuffles the lines in your input files.
  - **Example:** `python multitool.py shuffle wordlist.txt -o randomized.txt`

- **`unflatten`**
  - Reconstructs nested structures from dot-separated `key.path = value` pairs. The default output format is `json`.
  - **Options:** Use the `-k` (or `--key`) flag to unflatten only paths starting with a specific key.
  - **Supported Formats:** `json`, `yaml`, `toml`, and `xml`.
  - **Example:** `python multitool.py unflatten data.txt --output-format json`

- **`convert`**
  - Transforms structured data between JSON, YAML, TOML, and XML while preserving nested structures. It supports extracting sub-keys using dot notation (for example, `metadata.items`).
  - **Options:** Use the `-k` (or `--key`) flag to set an optional starting path.
  - **Example:** `python multitool.py convert input.json --key 'items' --output-format yaml`

- **`replace`**
  - Performs text substitution across multiple files using literal strings or regular expressions.
  - **Options:**
    - Provide the **`OLD`** pattern and **`NEW`** text as positional arguments or use the `--old` and `--new` flags.
    - Use the `--regex` (or **`-r`**) flag to treat the pattern as a regular expression.
    - Use the `--ignore-case` flag for case-insensitive matching.
    - Use the `--smart-case` (or **`-S`**) flag to automatically match the original casing pattern (for example, `Teh` -> `The`).
    - Supports `--in-place` (or **`-I`**), `--dry-run`, and `--diff` (or **`-D`**) flags.
  - **Example:** `python multitool.py replace 'old-tag' 'new-tag' . --smart-case --in-place`

- **`set_operation`**
  - Compares two files using set logic:
    - `intersection`: Finds items present in both files.
    - `union`: Combines all unique items from both files.
    - `difference`: Finds items present only in the first file.
    - `symmetric_difference`: Finds items present in either file, but not both.
  - **Example:** `python multitool.py set_operation a.txt --file2 b.txt --operation symmetric_difference`

- **`zip`**
  - Joins two files line-by-line into a paired format.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, `toml`, and `xml`.
  - **Example:** `python multitool.py zip typos.txt --file2 corrections.txt --output-format arrow`

- **`unzip`**
  - Splits paired data into two lists by extracting one side. It saves the left side by default.
  - **Options:** Use the `--right` flag to save the right side instead.
  - **Example:** `python multitool.py unzip typos.csv --right --output corrections.txt`

- **`swap`**
  - Reverses the order of elements in paired data (for example, `typo -> correction` becomes `correction -> typo`).
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, `toml`, and `xml`.
  - **Example:** `python multitool.py swap mappings.csv --output-format arrow`

- **`scrub`**
  - Fixes typos in text files using a mapping file or extra pairs. It preserves surrounding context while fixing errors.
  - **Supported Formats:** CSV, Arrow, Table, JSON, YAML, TOML, and XML mapping formats.
  - **Options:**
    - Use the `--add` flag to provide extra mapping pairs directly on the command line.
    - Supports `--in-place`, `--dry-run`, `--diff`, and `--smart-case`.
  - **Example:** `python multitool.py scrub input.txt --add teh:the --diff`

- **`standardize`**
  - Fixes inconsistent casing or spelling project-wide by using the most frequent form.
  - **Options:**
    - Supports `--in-place`, `--dry-run`, and `--diff`.
    - Use `--fuzzy` to set the maximum character distance for matching.
    - Use `--keyboard` or `--transposition` to filter for likely typing errors.
    - Use `--threshold` to set the minimum frequency ratio to consider a rare word a typo (default: 10.0).
  - **Example:** `python multitool.py standardize . --diff --min-length 4 --fuzzy 1 --keyboard`

- **`highlight`**
  - Color-codes words from a list or mapping in the output. This is useful as a preview before using `scrub` mode.
  - **Options:**
    - Use the `--mapping` flag or `--add` to provide words to find.
    - The `--smart` flag allows for coloring subwords within larger compound words.
  - **Example:** `python multitool.py highlight input.txt --add teh:the`

- **`pairs`**
  - Converts paired data (like `typo -> correction`) between any supported format.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, `toml`, and `xml`.
  - **Example:** `python multitool.py pairs typos.json --output-format csv`

### CHECK & ANALYZE

Use these modes to analyze your data.

- **`check`**
  - Finds words that appear in both the typo and correction columns. This helps spot errors in your typo lists.
  - **Example:** `python multitool.py check mappings.csv`

- **`classify`**
  - Groups typos by error type using codes like **[K]** Keyboard, **[T]** Transposition, **[Del]** Deletion, and **[Ins]** Insertion.
  - **Options:** Use `--show-dist` to include the number of character changes.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, `toml`, and `xml`.
  - **Example:** `python multitool.py classify typos.txt --show-dist --output labeled.txt`

- **`conflict`**
  - Finds typos associated with more than one unique correction.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, `toml`, and `xml`.
  - **Example:** `python multitool.py conflict mappings.csv`

- **`count`**
  - Counts how often each word, pair, line, character, or paragraph appears. It sorts the list by frequency.
  - **Options:**
    - `--min-count` and `--max-count`: Filter results by frequency.
    - `-d`, `--delimiter`: The character to split words by (default: whitespace).
    - `-S`, `--smart`: Split by symbols and capital letters (for example, "CamelCase").
    - `-p`, `--pairs`: Count frequencies of word pairs and classify them.
    - `-l`, `--lines`: Count frequencies of raw lines.
    - `-c`, `--chars`: Count frequencies of individual characters.
    - `-E`, `--sentences`: Count frequencies of individual sentences.
    - `-G`, `--paragraphs`: Count frequencies of individual paragraphs.
    - `-B`, `--by-file`: Count how many files contain each item.
  - **Visual Report:** Use `--output-format arrow` for a rich report with metrics and bar charts.
  - **Supported Formats:** `arrow`, `json`, `csv`, `markdown`, `md-table`, `line`, and `xml`.
  - **Note:** This mode has built-in sorting.
  - **Example:** `python multitool.py count all_typos.txt --min-count 5 -f arrow --smart`

- **`cycles`**
  - Finds loops in typo-correction pairs (for example, "A" maps to "B" and "B" maps back to "A").
  - **Example:** `python multitool.py cycles typos.csv --output-format arrow`

- **`fuzzymatch`**
  - Finds words in your list that are similar to words in a second list (large dictionary).
  - **Options:** Use `--min-dist` and `--max-dist` to control the allowed changes.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, `toml`, and `xml`.
  - **Example:** `python multitool.py fuzzymatch typos.txt words.csv --max-dist 1 --show-dist`

- **`near_duplicates`**
  - Finds pairs of words in your list that are very similar.
  - **Options:** Use `--min-dist` and `--max-dist` to control the allowed changes.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, `toml`, and `xml`.
  - **Example:** `python multitool.py near_duplicates words.txt --max-dist 1 --show-dist`

- **`similarity`**
  - Filters paired data based on the number of character changes between words.
  - **Options:** Use `--min-dist` and `--max-dist` to set the range of allowed changes.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, `toml`, and `xml`.
  - **Example:** `python multitool.py similarity typos.txt --max-dist 2 --show-dist`

- **`stats`**
  - Provides a detailed overview of your dataset, including counts and unique items.
  - **Supported Formats:** `json`, `yaml`, `markdown`, `md-table`, and `line` (human-readable).
  - **Example:** `python multitool.py stats typos.csv --pairs`

- **`discovery`**
  - Automatically finds potential typos by identifying rare words similar to frequent ones.
  - **Options:**
    - `--rare-max`: Maximum frequency for a word to be considered a typo (default: 1).
    - `--freq-min`: Minimum frequency for a word to be considered a correction (default: 5).
    - `--min-dist` and `--max-dist`: Control the number of allowed character changes.
    - `-S`, `--smart`: Split by symbols and capital letters.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, `toml`, and `xml`.
  - **Example:** `python multitool.py discovery code.py --smart --rare-max 2 --freq-min 10 --max-dist 1`

- **`casing`**
  - Finds words that appear in your files with multiple different casing styles (for example, 'hello', 'Hello').
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, `toml`, and `xml`.
  - **Example:** `python multitool.py casing report.txt --smart --output-format arrow`

- **`repeated`**
  - Finds consecutive identical words (for example, "the the").
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, `toml`, and `xml`.
  - **Example:** `python multitool.py repeated report.txt --smart --output-format arrow`

- **`anomalies`**
  - Finds words with structural irregularities like accidental capital letters or numbers in the middle of words (for example, 'HEllow' or 'w0rd'). This helps you find common typing errors without needing a dictionary.
  - **Options:**
    - Use the `--delimiter` (or `-d`) flag to set a different separator (default: whitespace).
    - Use the `--smart` (or `-S`) flag to split items by symbols and capital letters.
  - **Supported Formats:** `arrow`, `table`, `csv`, `markdown`, `md-table`, `json`, `yaml`, `toml`, and `xml`.
  - **Example:** `python multitool.py anomalies src/ --output-format arrow`

- **`search`**
  - Searches for a query in your files and identifies similar words or subword matches. It supports highlighting and context lines.
  - **Options:**
    - `QUERY`: The word or pattern to search for.
    - `--max-dist`: Maximum character changes for similar matching.
    - `-S`, `--smart`: Search for subwords within larger items.
    - `-k`, `--keyboard` or `-t`, `--transposition`: Filter for likely typing errors.
    - `-n`, `--line-numbers` and `-C`, `--context N`: Show location and surrounding lines.
    - `--heading` and `--no-heading`: Control whether results are grouped under filename headers (default is on for terminals).
  - **Example:** `python multitool.py search 'teh' report.txt --keyboard --line-numbers -C 1`

- **`scan`**
  - Searches for every word in a mapping file and reports matches with filename, line number, and highlighting.
  - **Options:**
    - `MAPPING`: The file containing typos to search for.
    - `-a`, `--add`: Provide extra mapping pairs directly on the command line.
    - `-S`, `--smart`: Scan for subwords within compound words.
    - `-n`, `--line-numbers` and `-C`, `--context N`: Show location and surrounding lines.
    - `--heading` and `--no-heading`: Control whether results are grouped under filename headers (default is on for terminals).
  - **Example:** `python multitool.py scan . --add teh:the --smart -A 1`

- **`verify`**
  - Checks which entries in a mapping file are present in your project.
  - **Options:**
    - Use the `--mapping` flag or `--add` to provide typos to check.
    - Use the `--prune` flag to output a new mapping file containing only the found typos.
    - Use the `--smart` flag to find subword matches.
  - **Example:** `python multitool.py verify . --add teh:the --prune`

- **`brokenlinks`**
  - Checks Markdown files for broken internal anchors (like `#missing-heading`) and missing local file references. It builds a project-wide map of all headings to correctly validate cross-file links.
  - **Example:** `python multitool.py brokenlinks docs/ --output-format arrow`

- **`orphans`**
  - Identifies unreferenced files and unused Markdown reference definitions. It helps you find dead assets (like images that aren't used) and redundant labels in your documentation.
  - **Example:** `python multitool.py orphans docs/ --output-format arrow`

- **`fileinfo`**
  - Gathers metadata such as file size, number of lines, word count, and detected encoding for the specified files.
  - **Example:** `python multitool.py fileinfo . --output-format arrow`

- **`duplicates`**
  - Identifies files with identical content by computing SHA-256 hashes. It first groups files by size to quickly filter out unique files, then performs full content hashing on potential candidates. It supports large-scale scans and handles standard input.
  - **Example:** `python multitool.py duplicates . --min-length 1024`

## Common Options

These options work with most modes:

- `[INPUT_FILES...]`: One or more files to read. Defaults to **standard input** if not provided.
- `--output` (or **`-o`**): The file to write results to. Defaults to printing to the screen.
- `--output-format` (or **`-f`**): The format of the output. Options include `line` (default), `json`, `yaml`, `toml`, `csv`, `markdown`, `md-table`, `arrow`, `table`, and `xml`. The tool automatically detects the format from the output file extension.
- `--min-length` (or **`-m`**): Skip items shorter than this length (default: 1 for most modes, 3 for word extraction modes like 'words' and 'count', 10 for sentence-based modes, and 20 for paragraph-based modes).
- `--max-length` (or **`-M`**): Skip words longer than this length (default: 1000).
- `--process-output` (or **`-P`**): Sorts the final list and removes duplicates. Use this to organize your output or remove redundant entries.
- `--limit` (or **`-L`**): Limit the number of items in the output.
- `--raw` (or **`-R`**): Keep punctuation and capitalization. By default, most tools convert everything to lowercase and remove all characters except for lowercase **a through z**. Use this flag if you need to preserve numbers, punctuation, or capitalization.
- `--quiet` (or **`-q`**): Hide progress bars and status messages.
