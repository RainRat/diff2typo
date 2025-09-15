# diff2typo Suite

The diff2typo Suite is a set of simple command-line tools that help you find, fix, and manage typos in your code. It works by analyzing Git diffs and text files and now includes several new features to make typo detection even easier.

## Table of Contents

- [Overview](#overview)
- [New Features](#new-features)
- [Tools in the Suite](#tools-in-the-suite)
- [Installation](#installation)
- [Usage](#usage)
- [Command-Line Arguments Summary](#command-line-arguments-summary)
- [Example Workflow](#example-workflow)
- [License](#license)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Overview

The diff2typo Suite helps you:
- **Extract typos** that you've already fixed in your code by scanning Git diffs (to prevent them from happening again along with the `typos` utility).
- **Generate synthetic typos** from word lists using configurable rules (to help find even more typos in your projects).
- **Process text files** with various modes to extract, count, or simply filter text (misc tasks that came up while processing this data, but use them for whatever you need them for).

Each tool can run on its own or be integrated into your development workflow.

## New Features

- **Multiple Modes in diff2typo.py:**  
  Choose from three modes:
  - **typos** – list new typos.
  - **corrections** – list new corrections for known typos.
  - **both** – combine the above two.

- **New gentypos.py Tool:**  
  Create many synthetic typos from a word list using rules such as keyboard adjacency, deletion, transposition, and duplication. The tool is fully configurable with a YAML file.

- **New multitool.py:**  
  Performs various tasks related to processing one text file into a new one. Various modes (arrow, backtick, csv, count, line) along with sorting and deduping, and filtering by length.

- **New cmdrunner.py:**  
  Run shell commands in every subdirectory of a base folder using a simple YAML configuration. This makes it easy to run commands (like `git diff`) across multiple projects.

- **New typostats.py:**  
  Analyze existing typo corrections to report common letter replacements. This tool can help customize `gentypos.yml`.

## Core Tools in the Suite

### diff2typo.py

Scans a Git diff file to find typo corrections. It:
- Splits compound words (by spaces, underscores, or changes in letter case).
- Ignores corrections when the original word is valid.
- Works with an external typos tool (if available) to skip duplicates.
- Supports three modes: `typos`, `corrections`, and `both`.
- Supports dictionary files as simple word lists or `words.csv` files; when a
  `words.csv` file is used, the correction columns are treated as valid words.

| Argument            | Description                                                                                       | Default       |
|---------------------|---------------------------------------------------------------------------------------------------|---------------|
| `--input_file`      | Path to the input Git diff file.                                                                   | `diff.txt`    |
| `--output_file`     | Path to the output typos file.                                                                     | `output.txt`   |
| `--output_format`   | Format of the output typos. Choices: `arrow`, `csv`, `table`, `list`.                             | `arrow`       |
| `--mode`            | Extract new typos, corrections to existing ones, or both. Choices: `typos`, `corrections`, `both` | `typos`       |
| `--typos_tool_path` | Path to the `typos` tool executable.                                                               | `typos`       |
| `--allowed_file`    | CSV file containing allowed words to exclude from typos.                                         | `allowed.csv` |
| `--min_length`      | Minimum length of differing substrings to consider as typos.                                      | `2`           |
| `--dictionary_file` | Path to the dictionary file for filtering valid words. Automatically detects format.               | `words.csv`   |

**Output Formats**

Choose the output format based on what you are using the data for.

- **arrow**: `typo -> correction`
  
  ```plaintext
  performanve -> performance
  teh -> the
  ```

- **csv**: `typo,correction`
  
  ```csv
  performanve,performance
  teh,the
  ```

- **table**: `typo = "correction"`
  
  ```plaintext
  performanve = "performance"
  teh = "the"
  ```

- **list**: `typo`
  
  ```plaintext
  performanve
  teh
  ```

**Example:**

```bash
python diff2typo.py --input_file=diff.txt --output_file=typos.txt --output_format=list --mode both --typos_tool_path=/path/to/typos --allowed_file=allowed.csv --dictionary_file=words.csv --min_length 2
```

### gentypos.py

Generates synthetic typos from a word list using configurable rules from a YAML file (`gentypos.yaml` by default). It can apply:
- Replacement (using adjacent keys or custom rules)
- Deletion and transposition
- Duplication

**Example:**

```bash
python gentypos.py -c gentypos.yaml -v
```

### cmdrunner.py

Runs a shell command in every subdirectory of a base folder. Configure:
- The base directory.
- The command to run.
- Folders to exclude.

**Example:**

```bash
python cmdrunner.py config.yaml
```

*Sample `config.yaml`:*

```yaml
base_directory: "/path/to/git/projects"
command_to_run: "git diff >> ../diff.txt"
excluded_folders:
  - "node_modules"
  - "venv"
```


## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/diff2typo.git
   cd diff2typo
   ```

2. **Python Requirement:**

   Make sure you have Python 3.6 or higher:

   ```bash
   python --version
   ```

3. **Install Dependencies:**

   Most tools use standard Python libraries. Some tools (like gentypos.py) need extra packages:

   ```bash
   pip install tqdm pyyaml chardet
   ```

4. **Optional: External Typos Tool**

   To integrate with the external typos tool:

   ```bash
   git clone https://github.com/crate-ci/typos.git
   cd typos
   # Follow the instructions to build/install.
   ```

## Usage

### Basic workflow

1. **Generate a diff file with Git:**

```bash
git diff abc123^ abc123 > diff.txt
git diff def456^ def456 >> diff.txt
```

2. **Run `diff2typo.py`**

It comes with some sensible defaults, so you just might need to point it in the right direction.

```bash
diff2typo.py --input_file=diff.txt
```

`output.txt` will be created in the `arrow` format and can be pasted into the monthly typo corrections thread if desired.

## Command-Line Arguments Summary

Each tool has its own set of options. For example:

- **diff2typo.py:**
  - `--input_file`: Path to the Git diff file.
  - `--output_file`: File to write results.
  - `--output_format`: Format (arrow, csv, table, list).
  - `--typos_tool_path`: Path to the external typos tool.
  - `--allowed_file`: CSV file with words to ignore.
  - `--dictionary_file`: File to filter out valid words.
  - `--min_length`: Minimum word length.
  - `--mode`: Choose `typos`, `corrections`, or `both`.


- **gentypos.py:**
  - `-c, --config`: YAML configuration file.
  - `-v, --verbose`: Enable detailed logging.

For more details, run any tool with the `--help` flag.

## Advanced Workflow

Customize steps to your workflow. Mix and match however you like:

 1. Generate list of possible synthetic typos. Use a small wordlist as a basis for the synthetic typos, and a large wordlist for excluding actual valid words. Configure `gentypos.yaml` and run:
   ```bash
   python gentypos.py -c gentypos.yaml
   ```
 2. Use the synthetic typo list with the typos tool to scan your projects. The synthetic typo list will be too large with too many false positives for regular use, but the typos tool can work with it.
   ```bash
typos --config typos_mega.toml --format brief GitHub > typos.txt
   ```
 3. Fix the typos that actually occurred in your project(s)
 4. --- If you don't want to use synthetic typo generation, just start here. ---
 5. Use cmdrunner.py to generate a diff that contains all your typo fixes.
    a. the default cmdrunner.yaml is intended to be run at the root folder of your git projects. Configure cmdrunner.yaml to your needs and run `cmdrunner.py cmdrunner.yaml`
    b. you can concatenate multiple diffs if you want to go back and process multiple past revisions
    c. or, just take a single diff of your project.
 6. Use diff2typo.py to generate a list of typos that actually occurred and you fixed. Make the resulting typo data a permanent part of your workflow, or submit it to the typos maintainer to be added. Example; adjust to your needs.
   ```
diff2typo.py --input_file=diff.txt --typos_tool_path=typos --mode both --dictionary_file=words.csv
   ```

## Side Utilities

### multitool.py

 Handles small tasks that come up. All of the modes take an input file and output file.

- **Command-line options**
  - `--mode`: Choose one of: arrow, backtick, count, csv, line, filterfragments, or check.
  - `--input`: Input file path.
  - `--output`: Output file path.
  - `--min-length` / `--max-length`: Length filters.
  - `--process-output`: Convert output to lowercase, sort, and remove duplicates.

 - arrow mode: Extract text before ' -> ', usually if you want to subject your list of possible typo corrections to further processing, ie.
Input:
teh -> the

Output:
teh

 -backtick mode: Extract strings between the first pair of backticks, usually to extract the typos from the output of the typos utility. If there were a lot of candidate typos you might want to sort by the typo and work through them methodically.

Input:
example.py:9:2: `teh` -> `the`

Output:
teh


 -csv mode: Extract the second word onward from each line of a csv file. Use `--first-column` to extract the first column instead. If you don't have a wordlist of valid words (for the synthetic typo generation) you can take the words.csv file from the typos data files, and make a wordlist out of that.
 
Input:
teh,the,ten

Output:
the
ten

-check mode: Scan a CSV of typos and corrections and list any words that appear in both the typo column and the correction columns.

-count mode: Take a list of words, and return them sorted descended by frequency. Typically a list of typos and you can prioritize whichever is most numerous. Ignores process-output.

Input:
teh
thier
teh


Output:
teh: 2
thier: 1


-line: leave each line exactly how it is (to expose file to max-length, min-length, and process-output)

### typostats.py

Analyzes typo corrections to report frequent letter replacements. This is to generate the `custom_substitutions` section of `gentypos.yaml`, which already comes with a `custom_substitutions`, so you probably won't need it, but this tool is available to experiment with.

**Example:**

```bash
python typostats.py input_file -o report.txt -m 2 -s count -f arrow -2
```

## License

This project is available under the [MIT License](LICENSE) and the [Apache 2.0 License](LICENSE-APACHE). It is not officially affiliated with the [typos](https://github.com/crate-ci/typos) project.

## Contributing

Contributions are welcome! To contribute:
1. **Fork the Repository.**
2. **Create a Feature Branch:**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes:**

   ```bash
   git commit -m "Add Your Feature"
   ```

4. **Push the Branch:**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request.**

## Acknowledgements

- [typos](https://github.com/crate-ci/typos): An inspiration for this project.
- All contributors and users who help improve this suite.

---

*Feel free to reach out with any questions or suggestions!*
