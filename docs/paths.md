# paths mode

Extract components from file and directory paths.

## Summary
The `paths` mode allows you to list and analyze the structure of your project by extracting specific parts of file and folder paths. It supports getting the filename (basename), the directory path (dirname), or the file extension. It also integrates with the suite's smart splitting functionality to identify words within filenames.

## Usage
```bash
python multitool.py paths [FILES...] [OPTIONS]
```

## Options

### Path Extraction Options
| Flag | Description |
| :--- | :--- |
| `--basename` | Extract the final component of the path (the filename). |
| `--dirname` | Extract the directory part of the path. |
| `--extension` | Extract the file extension. |
| `-S`, `--smart` | Split path components by symbols and capital letters (e.g., "CamelCase" or "snake_case"). |

### Processing & Filtering Options
| Flag | Description |
| :--- | :--- |
| `-m`, `--min-length` | Skip items shorter than this length (default: 1). |
| `-M`, `--max-length` | Skip items longer than this length (default: 1000). |
| `-R`, `--raw` | Keep the original text. Do not change it to lowercase or remove punctuation. |
| `-L`, `--limit` | Limit the number of items in the output. |
| `-P`, `--process-output` | Sort the results and remove duplicates. |

### Input/Output & General Options
| Flag | Description |
| :--- | :--- |
| `-i`, `--input` | Path(s) to the input file(s). Supports multiple files. |
| `-o`, `--output` | Where to save the results. Use `-` to print to the screen (default: the screen). |
| `-f`, `--format` | Choose the format for the output. Choices: `line`, `json`, `csv`, `markdown`, `md-table`, `arrow`, `table`, `yaml`, `toml`, `xml`. |
| `-q`, `--quiet` | Hide progress bars and status messages. |

## Examples

### List all filenames in a directory
```bash
python multitool.py paths src/ --basename --raw
```

### Extract all unique file extensions
```bash
python multitool.py paths . --extension --process-output
```

### Find all words used in filenames (Smart Splitting)
```bash
python multitool.py paths src/ --basename --smart --process-output
```

### Get unique folder names in a project
```bash
python multitool.py paths . --dirname --basename --process-output --raw
```
