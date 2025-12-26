# typostats.py

**Purpose:** Finds common patterns in your typos. It looks at your list of corrections and tells you which keys you hit by mistake most often (like hitting `p` instead of `o`).

## Usage

```bash
python typostats.py input.txt [OPTIONS]
```

## Input Format

The tool expects a simple text file where each line has a typo and its correction, separated by a comma:
`typo, correction`

> **Tip:** If you are feeding output from `diff2typo.py`, you **must** run it with `--output_format csv`. The default arrow format (`typo -> correction`) is not compatible.

## Options

- `--min`: **(Default: 1)** Show only typos that happen at least this many times. Useful for ignoring one-off mistakes.
- `--allow_two_char`: Look for cases where one letter is replaced by two (like `m` becoming `rn`).
- `--sort`: Organize the results. Options: `count` (most frequent first), `typo` (alphabetical), or `correct`.
- `--format`: Choose how the output looks:
  - `arrow`: Easy to read (`a -> b: 5`).
  - `json`: Good for other programs.
  - `yaml`: Good for config files.
  - `csv`: Open in a spreadsheet.
- `--output`: Save the results to a file instead of printing them to the screen.

## Example

Find typos that happened at least 5 times and save the report as JSON:

```bash
python typostats.py my_data.txt --format json --min 5 --output report.json
```
