# paths mode

Extract components from file and directory paths.

## Summary

The `paths` mode is a feature of the [Multitool](multitool.md) utility. It helps you analyze your project structure by extracting specific parts of file and folder paths. You can use it to get the filename, the folder path, or the file extension.

## Usage

Run the command with the `paths` mode and provide your files or folders:

```bash
python multitool.py paths [FILES...] [OPTIONS]
```

If you pipe data into the command, the tool reads from standard input. If you run the command in an interactive terminal without providing any files or folders, it automatically scans the current directory (`.`) recursively.

## Features

### Recursive Scanning

When you provide a folder as input, the tool automatically scans it recursively. It extracts path components from both folders and files.

### Automatically Ignored Folders

To keep things fast and clean, the tool automatically skips common development and system folders:
- `.git`
- `node_modules`
- `venv` and `.venv`
- `.pytest_cache` and `.ruff_cache`
- `.vscode` and `.idea`
- `__pycache__`
- `dist` and `build`

### Default Text Cleanup

By default, Multitool cleans up all output text. It removes punctuation and converts everything to lowercase letters. For example, the path `docs/paths.md` becomes `docspathsmd`.

If you want to keep the slashes, dots, and original capital letters, always add the `--raw` (or `-R`) flag.

## Options

| Flag | Description |
| :--- | :--- |
| `--basename` | Extract the final part of the path (the filename). |
| `--dirname` | Extract the directory part of the path. |
| `--extension` | Extract the file extension. |
| `-S`, `--smart` | Split path parts by symbols and capital letters. |
| `-R`, `--raw` | Keep the original text. This preserves capital letters and punctuation. |
| `-P`, `--process-output` | Sort the final list and remove duplicate lines. |
| `-m`, `--min-length` | Skip path items shorter than this character length. |
| `-M`, `--max-length` | Skip path items longer than this character length. |
| `-L`, `--limit` | Limit the number of extracted path items in the output. |
| `-o`, `--output` | Save the results to this file instead of printing to the screen. |
| `-f`, `--format` | Choose the output format (`line`, `csv`, `json`, `yaml`, `toml`, `markdown`, `md-table`, `arrow`, `table`, `xml`). Automatically detected from file extension. |
| `-q`, `--quiet` | Hide progress bars and status summary messages. |

## Examples

### List all filenames in a folder

Get only the filenames and keep their original casing and dots:

```bash
python multitool.py paths src/ --basename --raw
```

### Extract all unique file extensions

Find every file extension used in your project, sort them, and remove duplicates:

```bash
python multitool.py paths . --extension --process-output
```

### Find all words used in filenames

Use smart splitting to break up filenames (like `camelCase` or `snake_case`) into individual words:

```bash
python multitool.py paths src/ --basename --smart --process-output
```

### Get unique folder names in a project

Extract folder paths, sort them, and remove duplicates while keeping the original casing:

```bash
python multitool.py paths . --dirname --process-output --raw
```

### Save unique file extensions to a CSV file

Extract file extensions, remove duplicates, and save the output directly to a CSV file:

```bash
python multitool.py paths . --extension --raw --process-output -o extensions.csv
```
