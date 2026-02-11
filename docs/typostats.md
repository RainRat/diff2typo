# typostats.py

**Purpose:** Finds common patterns in your typos. It analyzes your list of corrections and identifies which keys you hit by mistake most often (like hitting `p` instead of `o`).

## Usage

Process one or more files or pipe data directly from standard input.

```bash
# Read from a file
python typostats.py my_typos.txt

# Read from multiple files
python typostats.py file1.txt file2.txt

# Read from standard input
git diff | python diff2typo.py | python typostats.py
```

## Input Format

The tool automatically recognizes three input formats:
1. **Arrow (Default):** `typo -> correction`
2. **CSV:** `typo, correction`
3. **Table:** `typo = "correction"`

> **Tip:** You can pipe the output from `diff2typo.py` directly into `typostats.py`.

## Options

### Analysis Options
- `-m`, `--min`: Only show replacements that happen at least this many times (Default: 1).
- `-s`, `--sort`: How to sort the results. Choose `count` (most frequent first), `typo`, or `correct`.
- `-n`, `--limit`: Only show the top N results.
- `-2`, `--allow-two-char`: Look for multi-character replacements (like `m` becoming `rn` or `ph` becoming `f`).
- `-t`, `--transposition`: Detect when you swap two adjacent letters (like `teh` instead of `the`).

### Output Options
- `-f`, `--format`: Choose the output format:
  - `arrow` (Default): Easy to read.
  - `csv`: Standard comma-separated values.
  - `json`: Machine-readable data.
  - `yaml`: Simple list format.
- `-o`, `--output`: Save the report to a file instead of printing it to the screen.
- `-q`, `--quiet`: Hide progress bars and informational messages.

## Understanding the Report

When using the default **arrow** format, the report displays results in a table:

```text
 CORRECT    TYPO   COUNT
------------------------
       o -> p    :     15
       i -> u    :     12
```

In this table, the left side is the **expected** character and the right side is the **mistake** you made. For example, `o -> p` means you hit the `p` key when you meant to hit `o`.

## Pro Tips

### Clean Output Strategy
`typostats.py` splits its output into two streams:
- **Standard Error (stderr):** Displays human-readable titles, table headers, and progress bars.
- **Standard Output (stdout):** Contains only the actual data rows.

This design allows you to pipe the report into other tools (like `grep` or `awk`) without needing to manually strip the headers first.

### Visual Feedback
The tool automatically detects if you are viewing the report in a terminal. If so, it uses **ANSI colors** to highlight correct characters in green and typos in red. It automatically disables these colors when you save the report to a file or pipe it to another command.

## Examples

**Find your top 5 most common mistakes, including transpositions:**
```bash
python typostats.py my_data.txt --limit 5 --transposition
```

**Find typos that happened at least 5 times and save the report as JSON:**
```bash
python typostats.py my_data.txt --format json --min 5 --output report.json
```
