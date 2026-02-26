# typostats.py

**Purpose:** Find common patterns in your typos. This tool analyzes your list of corrections and tells you which keys you hit by mistake most often (like hitting `p` instead of `o`).

## Usage

Process one or more files or send data directly to the tool using a pipe.

```bash
# Read from a file
python typostats.py my_typos.txt

# Read from multiple files
python typostats.py file1.txt file2.txt

# Pipe from another tool
git diff | python diff2typo.py | python typostats.py
```

## Input Format

The tool automatically recognizes three common ways of listing typos:
1. **Arrow (Default):** `typo -> correction`
2. **CSV:** `typo, correction`
3. **Table:** `typo = "correction"`

> **Tip:** You can send the output from `diff2typo.py` directly into `typostats.py`.

## Options

### Analysis Options
- `-m`, `--min`: Only show patterns that appear at least this many times (Default: 1).
- `-s`, `--sort`: How to sort the results. Choose `count` (most frequent first), `typo` (alphabetical by typo), or `correct` (alphabetical by fix).
- `-n`, `--limit`: Only show the top N results.
- `-2`, `--allow-two-char`: Look for cases where one letter is replaced by two (like `m` becoming `rn`) or two by one (like `ph` becoming `f`).
- `-t`, `--transposition`: Find swapped letters (like `teh` instead of `the`).
- `-k`, `--keyboard`: Identify typos caused by hitting keys next to each other on the keyboard.

### Output Options
- `-f`, `--format`: Choose the output format:
  - `arrow` (Default): Easy to read.
  - `csv`: Standard comma-separated values.
  - `json`: Data for other programs.
  - `yaml`: Simple list format.
- `-o`, `--output`: Save the report to a file instead of showing it on the screen.
- `-q`, `--quiet`: Hide progress bars and status messages.

## Understanding the Report

When using the default **arrow** format, the report displays results in a table:

```text
 Most Frequent Letter Replacements in Typos:

 Total replacements analyzed: 15
 Keyboard Adjacency: 12/15 (80.0%)

 CORRECT    TYPO   COUNT
 -----------------------
       o -> p    :     15 [K]
       i -> u    :     12 [K]
```

In this table, the left side is the **correct** character and the right side is the **mistake** you made. For example, `o -> p` means you hit the `p` key when you meant to hit `o`.

### Keyboard Analysis
When you use the `--keyboard` flag, the report adds two extra pieces of information:
- **[K] Marker:** This appears next to mistakes where the two keys are next to each other on a standard QWERTY keyboard.
- **Keyboard Adjacency Summary:** A percentage showing how many of your typos were likely caused by physical slips on the keyboard.

## Pro Tips

### Clean Output Strategy
`typostats.py` separates its output to keep your data clean:
- **Status Messages:** Titles, table headers, and progress bars are sent to the "error stream."
- **Data Rows:** The actual results are sent to the "standard output stream."

This design lets you pipe the report into other tools (like `grep` or `awk`) to process the data without having to remove the headers yourself.

### Visual Feedback
The tool detects if you are viewing the report in a terminal. If so, it uses colors to highlight correct characters in green and mistakes in red. It automatically turns off these colors when you save the report to a file or send it to another command.

## Examples

**Find your top 5 most common mistakes, including swapped letters:**
```bash
python typostats.py my_data.txt --limit 5 --transposition
```

**Find typos that happened at least 5 times and save the report as JSON:**
```bash
python typostats.py my_data.txt --format json --min 5 --output report.json
```

**See which typos were likely caused by hitting keys next to each other:**
```bash
python typostats.py my_data.txt --keyboard
```
