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
- **(Optional) The `typos` tool:** We recommend installing the [typos](https://github.com/crate-ci/typos) command-line utility. `diff2typo.py` uses it to automatically filter out typos that are already known, keeping your lists clean.

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

3. **(Optional) Better file support:**
   Install `chardet` to help the tools automatically handle files with different text encodings (like UTF-8 or Latin-1).
   ```bash
   pip install chardet
   ```

## 🛠️ Tools Overview

| Tool | What it does | Documentation |
| :--- | :--- | :--- |
| **diff2typo** | Finds typos you fixed in your Git history. | [Read Docs](docs/diff2typo.md) |
| **gentypos** | Creates lists of "fake" typos based on common typing errors. | [Read Docs](docs/gentypos.md) |
| **multitool** | A multipurpose tool for cleaning and processing text files. | [Read Docs](docs/multitool.md) |
| **cmdrunner** | Runs commands across many folders at once. | [Read Docs](docs/cmdrunner.md) |
| **typostats** | Analyzes your typos to find common finger-slips. | [Read Docs](docs/typostats.md) |

## 🚀 Quick Start

Follow these steps to find typos you have fixed recently and identify your common mistakes.

### 1. Prepare a Dictionary
The tools work best when they know which words are "valid." Create a file named `words.csv` and add words you use often (like project-specific terms) one per line. If you skip this, the tools will still work but may report more false positives.

### 2. Get Your Recent Changes
Create a file containing your last few Git commits.
```bash
git diff HEAD~5 HEAD > changes.diff
```

### 3. Extract the Typos
Run `diff2typo.py` on your changes. It will find words you corrected and save them to a file.
```bash
python diff2typo.py changes.diff --output my_typos.txt --mode typos --format csv
```

### 4. Analyze the Patterns
Use `typostats.py` to see which keys you hit by mistake most often.
```bash
python typostats.py my_typos.txt --sort count
```

## 📄 License

This project is available under the MIT License and the Apache 2.0 License.
