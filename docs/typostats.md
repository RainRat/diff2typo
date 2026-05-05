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
- `-a`, `--all`: Enable all analysis features at once. This is the default if no other analysis options are chosen.
- `-n`, `-L`, `--limit`: Only show the top N results.
- `-2`, `--allow-two-char`: Look for cases where one letter is replaced by two (like `m` -> `rn`) or two by one (like `ph` -> `f`).
- `--1to2`: Specifically look for single-to-double letter replacements.
- `--2to1`: Specifically look for double-to-single letter replacements.
- `--include-deletions`: Include cases where you added an extra letter or missed one (like `aa` -> `a`).
- `-t`, `--transposition`: Find swapped letters (like `teh` instead of `the`).
- `-k`, `--keyboard`: Find typos caused by hitting keys next to each other on the keyboard.

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
  Total word pairs analyzed:          4
  Total patterns analyzed:            4
  Total patterns after filtering:     4
  Retention rate:                     100.0% ████████████████████
  Unique patterns:                    3
  Min/Max/Avg length:                 7 / 8 / 7.5
  Shortest pattern:                   'm -> rn' (length: 7)
  Longest pattern:                    'eh -> he' (length: 8)
  Min/Max/Avg changes:                1 / 2 / 1.8
  Total lines processed:              4
  Enabled features:                   keyboard, transposition, 1-to-2, 2-to-1, deletions/insertions
  Transpositions [T]:                 2/4 (50.0%)
  2-to-1 replacements [2:1]:          1/4 (25.0%)
  Insertions [Ins]:                   1/4 (25.0%)
  Processing time:                    0.001s

  LETTER REPLACEMENTS
  ───────────────────────────────────────────────────────
  TYPO │ CORRECT │ COUNT │      % │ ATTR  │ VISUAL
  ───────────────────────────────────────────────────────
  eh   │ he      │     2 │  50.0% │ [T]   │ ███████▌
  m    │ rn      │     1 │  25.0% │ [2:1] │ ███▊
  he   │ h       │     1 │  25.0% │ [Ins] │ ███▊
```

### Analysis Summary
The dashboard at the top gives you an overview of your typo history:
- **Total word pairs analyzed:** The number of typo-correction pairs found in the input.
- **Total patterns analyzed:** The total number of character-level replacements extracted.
- **Total patterns after filtering:** The number of patterns that remain after applying filters (like `--min`).
- **Retention rate:** A visual bar showing the percentage of items kept after filtering.
- **Unique patterns:** The number of distinct character-level mistakes found.
- **Min/Max/Avg length:** Statistics on the length of the extracted patterns.
- **Min/Max/Avg changes:** Statistics on the number of character changes (edit distance) per typo.
- **Enabled features:** Which analysis modes (like keyboard adjacency or transpositions) were active.

### Letter Replacements Table
This section breaks down every mistake:
- **TYPO:** The mistake you actually made.
- **CORRECT:** The character you intended to type.
- **COUNT:** How many times this specific mistake happened.
- **%:** What percentage of all found replacements this mistake represents.
- **ATTR:** Special markers showing the type of mistake (for example, `[K]` for keyboard slip).
- **VISUAL:** A high-resolution bar chart for quick comparison.

For example, a row showing `eh │ he` means you swapped the letters `h` and `e`.

### Typo Attributes (ATTR)
When you enable analysis features, the tool finds specific patterns in the **ATTR** column:
- **[K]**: Keyboard slip (the keys are next to each other on a QWERTY layout).
- **[T]**: Transposition (swapped letters, like `teh` instead of `the`).
- **[1:2]**: One-to-two replacement (for example, typing `rn` instead of `m`).
- **[2:1]**: Two-to-one replacement (for example, typing `f` instead of `ph`).
- **[Ins]**: Insertion (typing an extra letter).
- **[Del]**: Deletion (missing a letter).

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
