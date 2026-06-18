# paths mode

Extract components from file and directory paths.

## Summary
The `paths` mode allows you to list and analyze the structure of your project by extracting specific parts of file and folder paths. It supports getting the filename (basename), the directory path (dirname), or the file extension. It also integrates with the suite's smart splitting functionality to identify words within filenames.

## Usage
```bash
python multitool.py paths [FILES...] [OPTIONS]
```

## Options
| Flag | Description |
| :--- | :--- |
| `--basename` | Extract the final component of the path (the filename). |
| `--dirname` | Extract the directory part of the path. |
| `--extension` | Extract the file extension. |
| `-S`, `--smart` | Split path components by symbols and capital letters. |
| `-R`, `--raw` | Keep original text (preserve casing and punctuation in paths). |
| `-P`, `--process-output` | Sort the results and remove duplicates. |

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
