# diff2typo Suite

The **diff2typo Suite** is a set of tools to help you find and fix typos in your code. It works with your Git history to learn from your past mistakes and helps you stop those typos from coming back.

## ✨ Key Features
- **Learn from history:** Automatically find typos you've already fixed in your Git logs.
- **Predict mistakes:** Generate lists of likely typos based on how keyboards are laid out.
- **Clean your data:** Powerful text processing tools to filter, merge, and organize typo lists.
- **Find patterns:** See which keys you hit by mistake most often.

## 📋 Prerequisites

- **Python 3.10 or newer:** The suite uses modern Python features.
- **Git:** Required to use `diff2typo.py` with your repository history.
- **Dependencies:** The following Python packages are required and will be installed in step 2:
  - `PyYAML`: Handles configuration files.
  - `pyahocorasick`: Performs fast string matching.
  - `tqdm`: Displays progress bars for long tasks.
  - `chardet`: Automatically detects file text encodings.
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
| **gentypos** | Creates lists of "fake" typos based on common typing errors. | [Read Docs](docs/gentypos.md) |
| **multitool** | A multipurpose tool for cleaning, getting, and analyzing text files. | [Read Docs](docs/multitool.md) |
| **cmdrunner** | Runs commands across many folders at once. | [Read Docs](docs/cmdrunner.md) |
| **typostats** | Analyzes your typos to find common finger-slips. | [Read Docs](docs/typostats.md) |

## 🚀 Quick Start

Follow these steps to find typos you have fixed recently and identify your common mistakes.

### 1. Create a Large Dictionary
The tools work best when they know which words are correct. Create a file named `words.csv` and add words you use often (like project names or technical terms), one per line. This is your "large dictionary." If you skip this, the tools will still work, but they might flag some correct words as typos.

### 2. Get Your Recent Changes
Save your recent Git changes to a file. For example, to see your last 5 changes:
```bash
git diff HEAD~5 HEAD > changes.diff
```

### 3. Find the Typos
Run `diff2typo.py` on your changes. It will find words you corrected and save them to a list.
```bash
python diff2typo.py changes.diff --output my_typos.txt --mode typos --format csv
```

### 4. See Your Patterns
Use `typostats.py` to see which keys you hit by mistake most often.
```bash
python typostats.py my_typos.txt --sort count
```

## 📄 License

This project is available under the MIT License and the Apache 2.0 License.
