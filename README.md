# diff2typo Suite

The **diff2typo Suite** is a collection of standalone command-line tools designed to help developers find, fix, and manage typos in codebases. It bridges the gap between Git history, synthetic typo generation, and linguistic analysis.

## 📦 Installation

**Prerequisites:** Python 3.10+

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/diff2typo.git
   cd diff2typo
   ```

2. **Install dependencies:**

   You can install all required packages at once:
   ```bash
   pip install -r requirements.txt
   ```

   Or install them individually:
   ```bash
   pip install tqdm pyyaml pyahocorasick
   # Optional: for encoding detection in typostats and multitool
   pip install chardet
   ```

## 🛠️ Tools Overview

| Tool | Purpose | Documentation |
| :--- | :--- | :--- |
| **diff2typo** | Extracts fixed typos from Git diffs to prevent regression. | [Read Docs](docs/diff2typo.md) |
| **gentypos** | Generates synthetic typos based on keyboard adjacency and heuristics. | [Read Docs](docs/gentypos.md) |
| **multitool** | A Swiss Army knife for text processing (extraction, filtering, set operations). | [Read Docs](docs/multitool.md) |
| **cmdrunner** | Runs shell commands across multiple subdirectories based on a config. | [Read Docs](docs/cmdrunner.md) |
| **typostats** | Analyzes correction lists to find common replacement patterns (e.g., `teh` -> `the`). | [Read Docs](docs/typostats.md) |

## 🚀 Quick Start Workflow

**1. Generate a Diff:**
Create a diff of recent changes in your project.

```bash
git diff HEAD~5 HEAD > recent_changes.diff
```

**2. Extract Typos:**
Use `diff2typo` to find spelling corrections you've already made.

```bash
python diff2typo.py --input_file recent_changes.diff --output_file my_typos.txt --mode typos --output_format csv
```

**3. Analyze Patterns:**
Use `typostats` to see your most common finger-slips.

```bash
python typostats.py my_typos.txt --sort count
```

## 📄 License

This project is available under the MIT License and the Apache 2.0 License.
