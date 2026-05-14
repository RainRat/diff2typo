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
- **Dependencies:** The following Python packages are required and will be installed in step 2:
  - `PyYAML`: Handles configuration files.
  - `pyahocorasick`: Performs fast string matching.
  - `tqdm`: Displays progress bars for long tasks.
  - `chardet`: Automatically detects file text encodings.
  - `toml`: Parses TOML configuration files.
  - `pytest`: Runs the project's test suite.
- **(Optional) The `typos` tool:** We recommend installing the [typos](https://github.com/crate-ci/typos) command-line tool. The suite uses it to automatically filter out known typos.

## 📦 Installation

1. **Download the code:**
   ```bash
   git clone https://github.com/yourusername/diff2typo.git
   cd diff2typo
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ Tools Overview

| Tool | What it does | Documentation |
| :--- | :--- | :--- |
| **diff2typo** | Finds typos you fixed in your Git history. | [Read Docs](docs/diff2typo.md) |
| **gentypos** | Creates lists of likely typos based on common typing errors. | [Read Docs](docs/gentypos.md) |
| **multitool** | A multipurpose tool for cleaning, getting, and analyzing text files. | [Read Docs](docs/multitool.md) |
| **cmdrunner** | Runs commands across many folders at once. | [Read Docs](docs/cmdrunner.md) |
| **typostats** | Analyzes your typos to find common finger-slips. | [Read Docs](docs/typostats.md) |

## 🚀 Quick Start

Follow these steps to find typos you have fixed recently, see your common mistakes, and fix them in your project.

### 1. Create a Large Dictionary
The tools work best when they know which words are correct. Create a file named `words.csv` and add words you use often (like project names or technical terms), one per line. This is your "large dictionary." If you skip this, the tools will still work, but they might flag some correct words as typos.

### 2. Find Your Recent Typos
Run `diff2typo.py` to find typos you fixed in your recent Git history. For example, to check your last 5 changes and save them to a CSV file:
```bash
python diff2typo.py --git "HEAD~5" --output my_typos.txt --mode typos --format csv
```

### 3. See Your Patterns
Use `typostats.py` to see which keys you hit by mistake most often.
```bash
python typostats.py my_typos.txt --sort count
```

### 4. Fix Your Project
Use `multitool.py` to fix the found typos in your current project files. The `--diff` flag lets you review the changes before they are applied.
```bash
python multitool.py scrub . --mapping my_typos.txt --in-place --diff
```

## 📄 License

This project is available under the MIT License and the Apache 2.0 License.
