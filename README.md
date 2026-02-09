# diff2typo Suite

The **diff2typo Suite** is a set of tools to help you find and fix typos in your code. It works with your Git history to learn from your past mistakes and helps you stop those typos from coming back.

## ✨ Key Features
- **Learn from history:** Automatically find typos you've already fixed in your Git logs.
- **Predict mistakes:** Generate lists of likely typos based on how keyboards are laid out.
- **Clean your data:** Powerful text processing tools to filter, merge, and organize typo lists.
- **Find patterns:** See which keys you hit by mistake most often.

## 📦 Installation

**Requirement:** Python 3.10 or newer.

1. **Download the code:**
   ```bash
   git clone https://github.com/yourusername/diff2typo.git
   cd diff2typo
   ```

2. **Install the basics:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Better file support:**
   Install `chardet` to help the tools automatically handle files that use different text encodings (like UTF-8, Latin-1, etc.).
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

Follow these steps to find typos you've fixed recently and see your common mistakes.

**1. Get your recent changes:**
Create a file containing your last few changes.
```bash
git diff HEAD~5 HEAD > changes.diff
```

**2. Extract the typos:**
Run `diff2typo` on your changes to see what you fixed.
```bash
python diff2typo.py changes.diff --output my_typos.txt --mode typos --format csv
```

**3. See the patterns:**
Use `typostats` to find your most frequent mistakes.
```bash
python typostats.py my_typos.txt --sort count
```

## 📄 License

This project is available under the MIT License and the Apache 2.0 License.
