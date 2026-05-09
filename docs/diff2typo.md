# diff2typo.py

**Purpose:** Scans your Git history to find typos you have fixed. This helps you build a list of common mistakes to avoid in the future.

## Usage

```bash
# Read from a file
python diff2typo.py my_changes.diff [OPTIONS]

# Read from standard input
git diff | python diff2typo.py [OPTIONS]
```

## Core Features

1. **Find typos in diffs:** Reads Git diff files or data sent directly from other commands to find words you have corrected.
2. **Variable Support:** Automatically splits compound words like `camelCase` and `snake_case` to find typos hidden inside variable names.
3. **Smart Filtering:** Uses a large dictionary of correct words and a list of "allowed" words to prevent the tool from reporting correct words as typos.
4. **Integration:** Can check your findings against the external `typos` tool to ensure your list only contains mistakes.

## Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `FILE` | standard input | One or more input Git diff files. Use `-` to read from standard input. |
| `--git` | None | Fetch diff directly from Git. Optional arguments are passed to `git diff` (for example, `--git "HEAD~3"`). |
| `--output`, `-o` | the screen | Path to the output file. Use `-` to print to the screen. |
| `--format`, `-f` | `arrow` | Choose the output format: `arrow` (typo -> fix), `csv` (typo,fix), `table` (typo = "fix"), or `list` (typo only). |
| `--mode` | `typos` | **`typos`**: Find typos that are not in your large dictionary (default).<br>**`corrections`**: Find corrections for typos in your large dictionary.<br>**`both`**: Run both checks and label the results.<br>**`audit`**: Find cases where a correct word was changed into a typo. |
| `--min-length`, `-m` | `2` | Ignore words shorter than this length. |
| `--max-dist` | None | Only include typos with a number of character changes up to this value. Useful for filtering out intentional word changes. |
| `--dictionary`, `-d` | `words.csv` | A file containing the large dictionary of correct words. The tool uses this to make sure the "fix" is a real word. |
| `--allowed` | `allowed.csv` | A list of words to explicitly ignore, even if they look like typos. |
| `--typos-path` | `typos` | The path to the external `typos` tool. |
| `--quiet`, `-q` | Off | Hide progress bars and status messages. |

## Examples

**Extract typos from a specific diff file:**

```bash
python diff2typo.py feature.diff --mode typos --format list
```

**Find cases where a correct word was changed into a typo:**

```bash
python diff2typo.py recent_changes.diff --mode audit
```

**Fetch recent changes directly from Git:**

```bash
python diff2typo.py --git "HEAD~5" --output recent_typos.txt
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
