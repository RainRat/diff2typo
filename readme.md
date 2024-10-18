# diff2typo.py

**diff2typo.py** is a Python utility designed to streamline the process of extracting and managing typo corrections from Git diffs. By analyzing changes that fix multiple typos, this tool generates a structured list of known typos, ensuring they are recognized and handled consistently in future codebases.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Generating a Diff File](#generating-a-diff-file)
  - [Running diff2typo.py](#running-diff2typopy)
- [Command-Line Arguments](#command-line-arguments)
- [Output Formats](#output-formats)
- [Example Workflow](#example-workflow)
- [License](#license)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Features

- **Extract Typos from Git Diffs**: Identifies typo corrections from Git diffs efficiently.
- **Compound Word Splitting**: Splits compound words based on spaces, underscores, and casing boundaries to accurately detect individual typos.
- **Dictionary Filtering**: Excludes corrections where the "before" word is a valid dictionary word, ensuring only true typos are captured.
- **Integration with `typos` Tool**: Avoids duplicate typo entries by integrating with the `typos` spell-checking utility.
- **Flexible Output Formats**: Supports multiple output formats (`arrow`, `csv`, `table`, and `list`) to suit various integration needs.
- **Customizable via Command-Line Options**: Offers a range of options to tailor the extraction and formatting process.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/diff2typo.git
   cd diff2typo
   ```

2. **Ensure Python is Installed**

   Make sure you have Python 3.6 or higher installed on your system. You can verify your Python version with:

   ```bash
   python --version
   ```

3. **Install Dependencies**

   The script uses standard Python libraries, so no additional installations are required. However, ensure you have the `typos` tool installed if you intend to use its integration features.

   ```bash
   # Example installation of typos tool
   git clone https://github.com/crate-ci/typos.git
   cd typos
   # Follow the typos tool's installation instructions
   ```

## Usage

### Generating a Diff File

Before running `diff2typo.py`, you need to generate a diff file that captures the changes fixing typos. Here's how you can do it:

1. **Identify the Commit(s) Fixing Typos**

   Find the commit(s) in your repository that primarily fix typos.

2. **Generate the Diff**

   Use `git diff` to create a diff file. You can include multiple commits or even diffs from different repositories into a single file.

   ```bash
   # Single commit diff
   git diff <commit_hash>^ <commit_hash> > diff.txt

   # Multiple commits or repositories
   git diff [parameters] >> diff.txt
   ```

   **Example:**

   ```bash
   git diff abc123^ abc123 > diff.txt
   git diff def456^ def456 >> diff.txt
   ```

### Running diff2typo.py

Once you have your `diff.txt` file, run the `diff2typo.py` script to extract and process the typos.

```bash
python diff2typo.py \
    --input_file=diff.txt \
    --output_file=typos.txt \
    --output_format=list \
    --typos_tool_path=/path/to/typos \
    --allowed_file=allowed.csv \
    --dictionary_file=/path/to/dictionary.txt \
    --min_length=2
```

## Command-Line Arguments

| Argument            | Description                                                                                       | Default       |
|---------------------|---------------------------------------------------------------------------------------------------|---------------|
| `--input_file`      | Path to the input Git diff file.                                                                   | `diff.txt`    |
| `--output_file`     | Path to the output typos file.                                                                     | `typos.txt`   |
| `--output_format`   | Format of the output typos. Choices: `arrow`, `csv`, `table`, `list`.                             | `arrow`       |
| `--typos_tool_path` | Path to the `typos` tool executable.                                                               | `typos`       |
| `--allowed_file`    | CSV file containing allowed words to exclude from typos.                                         | `allowed.csv` |
| `--min_length`      | Minimum length of differing substrings to consider as typos.                                      | `2`           |
| `--dictionary_file` | Path to the dictionary file for filtering valid words. Automatically detects format.               | `words.csv`   |

## Output Formats

Choose the output format that best suits your integration needs:

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

## Example Workflow

1. **Generate the Diff File**

   ```bash
   git diff abc123^ abc123 > diff.txt
   git diff def456^ def456 >> diff.txt
   ```

2. **Run diff2typo.py**

   ```bash
   python diff2typo.py \
       --input_file=diff.txt \
       --output_file=typos_list.txt \
       --output_format=list \
       --typos_tool_path=typos/typos \
       --allowed_file=allowed.csv \
       --dictionary_file=words.csv \
       --min_length=2
   ```

3. **Review the Output**

   The `typos_list.txt` file will contain a list of typos, one per line:

   ```plaintext
   performanve
   teh
   ```

4. **Integrate with `typos` Tool**

   Use the generated `typos.txt` or `typos_list.txt` as needed in your workflows or documentation.

## License

This project is licensed under the [MIT License](LICENSE) and [Apache 2.0 License](LICENSE-APACHE) for compatibility with the parent [typos](https://github.com/crate-ci/typos) tool. There is no official affiliation with the `typos` project.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add Your Feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## Acknowledgements

- [typos](https://github.com/crate-ci/typos): The spell-checking tool that inspired and is compatible with this project.

---

*Feel free to reach out with any questions or suggestions!*

