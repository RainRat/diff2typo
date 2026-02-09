# typostats.py

**Purpose:** Finds common patterns in your typos. It looks at your list of corrections and tells you which keys you hit by mistake most often (like hitting `p` instead of `o`).

## Usage

You can process one or more files, or pipe data directly from standard input.

```bash
# Read from a file
python typostats.py my_typos.txt [OPTIONS]

# Read from multiple files
python typostats.py file1.txt file2.txt [OPTIONS]

# Read from standard input
git diff | python diff2typo.py | python typostats.py [OPTIONS]
```

## Input Format

The tool automatically recognizes three input formats:
1. **Arrow (Default):** `typo -> correction`
2. **CSV:** `typo, correction`
3. **Table:** `typo = "correction"`

> **Tip:** You can pipe the output from `diff2typo.py` directly into `typostats.py`.

## Options

### Analysis Options
- `-m`, `--min`: **(Default: 1)** Only show replacements that happen at least this many times.
- `-s`, `--sort`: How to sort the results. Options: `count` (most frequent first), `typo`, or `correct`.
- `-n`, `--limit`: Only show the top N results.
- `-2`, `--allow-two-char`: Look for cases where one character is replaced by two (like `m` becoming `rn`), or two characters are replaced by one (like `ph` becoming `f`).
- `-t`, `--transposition`: Detect when you swap two adjacent letters (like `teh` instead of `the`).

### Output Options
- `-f`, `--format`: Choose the output format:
  - `arrow` (Default): Easy to read (`a -> b: 5`).
  - `csv`: Standard comma-separated values.
  - `json`: Machine-readable data.
  - `yaml`: Simple list format.
- `-o`, `--output`: Save the report to a file instead of printing it to the screen.
- `-q`, `--quiet`: Hide informational messages.

## Examples

**Find your top 5 most common mistakes, including transpositions:**
```bash
python typostats.py my_data.txt --limit 5 --transposition
```

**Find typos that happened at least 5 times and save the report as JSON:**
```bash
python typostats.py my_data.txt --format json --min 5 --output report.json
```
