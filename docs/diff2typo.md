# diff2typo.py

**Purpose:** Scans your Git history to find typos you have already fixed. This helps you build a list of common mistakes to avoid in the future.

## Usage

```bash
# Read from a file
python diff2typo.py my_changes.diff [OPTIONS]

# Read from standard input (piping)
git diff | python diff2typo.py [OPTIONS]
```

## Core Features

1. **Diff Parsing:** Reads Git diff files or data sent directly from other commands.
2. **Variable Support:** Automatically splits compound words like `camelCase` and `snake_case` to find typos hidden inside variable names.
3. **Smart Filtering:** Uses a dictionary of valid words and a list of "allowed" words to prevent the tool from reporting correct words as typos.
4. **Integration:** Can check your findings against the external `typos` tool to ensure your list only contains new mistakes.

## Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `FILE` | standard input | One or more input Git diff files. Use `-` to read from standard input. |
| `--output`, `-o` | standard output | Path to the output file. Use `-` to print to the screen. |
| `--format`, `-f` | `arrow` | Choose the output format: `arrow` (typo -> fix), `csv` (typo,fix), `table` (typo = "fix"), or `list` (typo only). |
| `--mode` | `typos` | **`typos`**: Find new typos that are not in your dictionary (default).<br>**`corrections`**: Find new ways to fix typos that are already in your dictionary.<br>**`both`**: Run both checks and label the results. |
| `--min-length`, `-m` | `2` | Ignore words shorter than this length. |
| `--dictionary`, `-d` | `words.csv` | A file containing valid words. The tool uses this to make sure the "fix" is a real word. |
| `--allowed` | `allowed.csv` | A list of words to explicitly ignore, even if they look like typos. |
| `--typos-path` | `typos` | The path to the external `typos` tool executable. |
| `--quiet`, `-q` | Off | Hide progress bars and status messages. |

## Examples

**Extract new typos from a specific diff file:**

```bash
python diff2typo.py feature.diff --mode typos --format list
```

**Pipe directly from Git and save to a file:**

```bash
git diff | python diff2typo.py --output found_typos.txt --mode both
```

**Find patterns with typostats:**

```bash
python diff2typo.py recent_changes.diff --format csv --output typos.csv
python typostats.py typos.csv
```
