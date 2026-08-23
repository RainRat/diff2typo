# diff2typo Suite

The **diff2typo Suite** is a set of tools to help you find and fix typos in your code. It works with your Git history to learn from your past mistakes and helps you stop those typos from coming back.

## ✨ Key Features
- **Learn from history:** Automatically find typos you fixed in your Git logs.
- **Predict mistakes:** Create lists of likely typos based on your keyboard layout.
- **Clean your data:** Filter, merge, and organize typo lists.
- **Find patterns:** See which keys you hit by mistake most often.

## 📋 Prerequisites

- **Python 3.10 or newer:** The suite uses recent Python features.
- **Git:** Required to use `diff2typo.py` with your repository history.
- **Dependencies:** The following Python packages are required and will be installed in step 3:
  - `PyYAML`: Handles configuration files.
  - `pyahocorasick`: Performs fast string matching.
  - `tqdm`: Displays progress bars for long tasks.
  - `chardet`: Automatically detects file text encodings.
  - `toml`: Parses TOML configuration files.
  - `pytest`: Runs the project's test suite.
- **(Optional) The `typos` tool:** We recommend installing the [typos](https://github.com/crate-ci/typos) command-line tool. The suite uses it to automatically filter out known typos.

## 📦 Installation

Follow these steps to set up the project on your computer.

### 1. Download the code
Clone the repository and move into the project directory:
```bash
git clone https://github.com/yourusername/diff2typo.git
cd diff2typo
```

### 2. Create and activate a virtual environment (Recommended)
A virtual environment keeps your project dependencies separate from your global Python installation.

*   **On macOS and Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
*   **On Windows (Command Prompt):**
    ```cmd
    python -m venv venv
    venv\Scripts\activate.bat
    ```
*   **On Windows (PowerShell):**
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

### 3. Install the dependencies
Install all required Python packages:
```bash
pip install -r requirements.txt
```

## 🛠️ Tools Overview

| Tool | What it does | Documentation |
| :--- | :--- | :--- |
| **diff2typo** | Finds typos you fixed in your Git history. | [Read Docs](docs/diff2typo.md) |
| **gentypos** | Creates lists of likely typos based on common typing errors. | [Read Docs](docs/gentypos.md) |
| **multitool** | A multipurpose tool for cleaning, getting, and analyzing text, files, and paths. | [Read Docs](docs/multitool.md) |
| **cmdrunner** | Runs commands across many folders at once. | [Read Docs](docs/cmdrunner.md) |
| **typostats** | Analyzes your typos to find common typing errors. | [Read Docs](docs/typostats.md) |

## 🚀 Quick Start

Follow these steps to find typos you have fixed recently, see your common mistakes, and fix them in your project.

### 1. Create a Large Dictionary
The tools work best when they know which words are correct. Create a file named `words.csv` and add words you use often (like project names or technical terms), one per line. This is your "large dictionary." If you skip this, the tools will still work, but they might flag some correct words as typos.

### 2. Find Your Recent Typos
Run `diff2typo.py` to find typos you fixed in your recent Git history. If you run `diff2typo.py` inside a Git repository without arguments, it automatically scans your unstaged changes. You can also fetch diffs directly using the `-g` / `--git` option:
```bash
python diff2typo.py -g "HEAD~1" --output my_typos.txt --mode typos --format csv
```

### 3. See Your Patterns
Use `typostats.py` to see which keys you hit by mistake most often.
```bash
python typostats.py my_typos.txt --sort count
```

### 4. Fix Your Project
Use `multitool.py` to fix found typos in your project files using your saved typo mapping file. The `--diff` and `--dry-run` flags let you review the changes before they are applied:
```bash
python multitool.py scrub . -s my_typos.txt --diff --dry-run
```
You can also preview fixes for specific typos directly using `--add` and `--dry-run`:
```bash
python multitool.py scrub . --add teh:the --diff --dry-run
```

## 🧪 Running Tests

This project has a full suite of tests to make sure everything works correctly.

Always use `python -m pytest` instead of running `pytest` directly. This ensures that Python can find all project modules and avoids path errors.

To run all the tests in the repository:
```bash
python -m pytest
```

To run a specific test file, provide the path to that file:
```bash
python -m pytest tests/test_typostats.py
```

To see more details about each test while it runs, use the verbose flag:
```bash
python -m pytest -v
```

## 📄 License

This project is available under the MIT License and the Apache 2.0 License.
