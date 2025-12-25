# multitool.py

**Purpose:** A Swiss Army knife for text file processing. It contains multiple subcommands (modes) to handle specific formatting, extraction, and set operations.

## Usage

```bash
python multitool.py <MODE> [OPTIONS]
```

## Modes

### `arrow`

Extracts the "typo" portion from `typo -> correction` lines.

```bash
python multitool.py arrow --input data.txt --output cleaned.txt
```

### `backtick`

Extracts text found between backticks (`\``). It includes heuristics to prioritize compiler error messages over warning messages.

```bash
python multitool.py backtick --input build.log
```

### `csv`

Extracts columns from CSV files.

- `--first-column`: Extract only the first column.
- Default: Extracts all columns *after* the first.

```bash
python multitool.py csv --input data.csv --delimiter ","
```

### `combine`

Merges multiple text files into a single, deduplicated, sorted file.

```bash
python multitool.py combine --input file1.txt file2.txt --output combined.txt
```

### `count`

Counts word frequency in a file. Useful for prioritizing which typos to fix first.

```bash
python multitool.py count --input all_typos.txt
```

### `check`

Identifies words that are both a typo and a correction in a CSV. This helps find errors in your typo database.

```bash
python multitool.py check --input mappings.csv
```

### `filterfragments`

Removes words from Input A if they appear as substrings inside Input B.

```bash
python multitool.py filterfragments --input candidates.txt --file2 dictionary.txt
```

### `set_operation`

Performs set operations on file contents.

- `--operation`: `union`, `intersection`, or `difference`.

```bash
python multitool.py set_operation --input a.txt --file2 b.txt --operation difference
```

## Common Options

Most modes support:

- `--min-length`: Ignore short strings.
- `--process-output`: Sort, deduplicate, and lowercase results.
