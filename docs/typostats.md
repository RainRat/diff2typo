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
- `-a`, `--all`: Shorthand to enable all analysis features at once (transpositions, keyboard adjacency, 1-to-2/2-to-1 replacements, and deletions/insertions).
- `-n`, `-L`, `--limit`: Only show the top N results.
- `-2`, `--allow-two-char`: Look for cases where one letter is replaced by two (like `m` -> `rn`) or two by one (like `ph` -> `f`).
- `--1to2`: Specifically look for single-to-double letter replacements.
- `--2to1`: Specifically look for double-to-single letter replacements.
- `--include-deletions`: Include cases where you skipped a letter or typed an extra one.
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

When using the default **arrow** format, the report displays results in two sections: a summary dashboard and a detailed replacement table.

```text
  ANALYSIS SUMMARY
  ───────────────────────────────────────────────────────
  Total lines processed:              10
  Total pairs processed:              15
  Replacements identified:            15
  Retention rate:                     100.0%
  Unique patterns identified:         2
  Enabled features:                   keyboard, transposition
  Keyboard Adjacency [K]:             12/15 (80.0%)
  Transpositions [T]:                 3/15 (20.0%)

  LETTER REPLACEMENTS
  ───────────────────────────────────────────────────────
  CORRECT │ TYPO │ COUNT │      % │ ATTR │ VISUAL
  ───────────────────────────────────────────────────────
        o │ p    │    12 │  80.0% │ [K]  │ ████████████
       th │ ht   │     3 │  20.0% │ [T]  │ ███
```

### Analysis Summary
The dashboard at the top gives you an overview of your typo history:
- **Total lines/pairs processed:** How much data was analyzed.
- **Replacements identified:** How many actual mistakes were found.
- **Unique patterns:** How many different types of mistakes were found.
- **Keyboard Adjacency [K]:** Percentage of typos caused by hitting a key next to the correct one.
- **Transpositions [T]:** Percentage of typos caused by swapping two letters.
- **Multi-character [M]:** Percentage of typos involving multiple letters (like `m` -> `rn`).

### Letter Replacements Table
This section breaks down every mistake:
- **CORRECT:** The character you intended to type.
- **TYPO:** The mistake you actually made.
- **COUNT:** How many times this specific mistake happened.
- **%:** What percentage of all identified replacements this mistake represents.
- **ATTR:** Special markers identifying the type of mistake (e.g., `[K]` for keyboard slip).
- **VISUAL:** A small bar chart for quick comparison.

For example, a row showing `o │ p` means you typed `p` when you meant to type `o`.

### Typo Attributes (ATTR)
When you enable analysis features, the tool identifies specific patterns in the **ATTR** column:
- **[K]**: Keyboard slip (the keys are next to each other on a QWERTY layout).
- **[T]**: Transposition (swapped letters, like `teh` instead of `the`).
- **[M]**: Multi-character replacement (e.g., `m` to `rn` or `ph` to `f`).

### Visual Bar
The **VISUAL** column provides a small bar chart to help you quickly see which mistakes are the most frequent.

## Pro Tips

### Clean Output Strategy
`typostats.py` separates its output to keep your data clean:
- **Status Messages:** Titles, table headers, and progress bars are sent to **standard error**.
- **Data Rows:** The actual results are sent to the **main output**.

This design lets you pipe the report into other tools to process the data without having to remove the headers yourself.

### Visual Feedback
The tool detects if you are viewing the report on your screen. If so, it uses colors to highlight correct characters in green and mistakes in red. It automatically turns off these colors when you save the report to a file or send it to another command.

## Examples

**Find your top 5 most common mistakes, including swapped letters:**
```bash
python typostats.py my_data.txt --limit 5 --transposition
```

**Run all analysis modes at once:**
```bash
python typostats.py my_data.txt -a
```

**Find typos that happened at least 5 times and save the report as JSON:**
```bash
python typostats.py my_data.txt --format json --min 5 --output report.json
```

**See which typos were likely caused by hitting keys next to each other:**
```bash
python typostats.py my_data.txt --keyboard
```
