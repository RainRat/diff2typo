# typostats.py

**Purpose:** Analyzes a list of typo corrections to determine statistical patterns in keyboard slips. It identifies which characters are most frequently swapped.

## Usage

```bash
python typostats.py input.txt [OPTIONS]
```

## Input Format

The tool expects a file where lines are comma-separated:
`typo, correction`

## Options

- `--allow_two_char`: Checks for one-to-two character replacements (e.g., `m` vs `rn`).
- `--format`:
  - `arrow`: Human readable (`a -> b: 5`).
  - `json`: Machine readable.
  - `yaml`: For configuration generation.
  - `csv`: For spreadsheet import.
- `--sort`: Sort results by `count`, `typo`, or `correct`.

## Example

Generate a JSON report of the most frequent typos that occur at least 5 times:

```bash
python typostats.py my_data.txt --format json --min 5
```
