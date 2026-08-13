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

1. **Find typos in diffs:** Reads Git diff files or data sent directly from other commands to find words you have corrected. This includes finding typos corrected by renaming or copying files.
2. **Variable Support:** Automatically splits compound words like `camelCase` and `snake_case` to find typos hidden inside variable names.
3. **Smart Filtering:** Uses a large dictionary of correct words and a list of "allowed" words to prevent the tool from reporting correct words as typos.
4. **Integration:** Can check your findings against the external `typos` tool to ensure your list only contains mistakes.

## Recursive Directory Scanning

If you provide a directory path as input, the tool automatically searches it recursively for supported files.

### Supported File Extensions
- `.diff`
- `.patch`
- `.txt`
- `.log`

### Automatically Ignored Folders
To keep scanning fast and clean, the tool automatically skips common system and development folders:
- `.git`
- `node_modules`
- `venv` and `.venv`
- `.pytest_cache` and `.ruff_cache`
- `.vscode` and `.idea`
- `__pycache__`
- `dist` and `build`

## Automatic Git Fallback

If you run the tool without specifying any input files or piping any changes, it will automatically check if you are inside a Git repository.

- **Inside a Git Repository:** The tool automatically runs `git diff` to get your unstaged changes and scans them for typos. This means you do not need to generate a diff file or pipe the output yourself.
- **Outside a Git Repository:** The tool prints a helpful error message and exits. It will ask you to provide input files, pipe diff data, or run the command inside a Git repository.

## Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `FILE` | standard input | One or more input Git diff files, directories, or glob patterns. Use `-` to read from standard input. |
| `--git`, `-g` | None | Fetch diff directly from Git. Optional arguments are passed to `git diff` (for example, `-g "HEAD~3"`). |
| `--git-log`, `-l` | None | Fetch commit history diffs directly from Git. Optional arguments are passed to `git log` (for example, `-l "HEAD~3"`). |
| `--output`, `-o` | the screen | Path to the output file. Use `-` to print to the screen. |
| `--format`, `-f` | `arrow` | Choose the output format: `arrow` (typo -> fix), `csv` (typo,fix), `table` (typo = "fix"), or `list` (typo only). |
| `--mode`, `-M` | `typos` | **`typos`**: Find typos that are not in your large dictionary (default).<br>**`corrections`**: Find corrections for typos in your large dictionary.<br>**`both`**: Run both checks and label the results.<br>**`audit`**: Find cases where a correct word was changed into a typo. |
| `--exclude`, `-e` | None | One or more file patterns (e.g., `*.json`, `tests/*`) to exclude from typo scanning. |
| `--include`, `-I` | None | One or more file patterns (e.g., `*.md`, `src/*`) to include in typo scanning (all files are scanned by default). |
| `--min-length`, `-m` | `2` | Ignore words shorter than this length. |
| `--max-dist`, `-D` | None | Only include typos with a number of character changes up to this value. Useful for filtering out intentional word changes. |
| `--min-count`, `-c` | `1` | Minimum occurrences of a typo in the diff to include it in the output. |
| `--sort` | `alpha` | How to sort the results: `count` (most frequent first) or `alpha` (alphabetical). |
| `--limit`, `-L` | None | Limit the number of typos in the output. |
| `--dictionary`, `-d` | `words.csv` | A file containing the large dictionary of correct words. The tool uses this to make sure the "fix" is a real word. |
| `--allowed` | `allowed.csv` | A list of words to explicitly ignore, even if they look like typos. |
| `--typos-path` | `typos` | The path to the external `typos` tool. |
| `--quiet`, `-q` | Off | Hide progress bars and status messages. |

## Examples

**Extract typos from a specific diff file:**

```bash
python diff2typo.py feature.diff --mode typos --format list
```

**Scan a directory recursively for diff and patch files:**

```bash
python diff2typo.py my_patches_dir/ --output found_typos.txt
```

**Find cases where a correct word was changed into a typo:**

```bash
python diff2typo.py recent_changes.diff --mode audit
```

**Fetch recent changes directly from Git:**

```bash
python diff2typo.py --git "HEAD~5" --output recent_typos.txt
```

**Fetch commit history directly from Git log:**

```bash
python diff2typo.py --git-log "HEAD~5" --output recent_typos.txt
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
