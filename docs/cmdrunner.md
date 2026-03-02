# cmdrunner.py

**cmdrunner.py** runs a specific command in every folder within a main directory. This is useful for running tools like `git diff` or `npm install` across many different projects at once.

## Prerequisites

- **Python 3.10 or newer**
- **PyYAML:** This package is required to read your configuration file. You can install it using:
  ```bash
  pip install PyYAML
  ```

## Usage

To use the tool, provide the path to your configuration file:

```bash
python cmdrunner.py config.yaml
```

## Configuration

The tool uses a YAML file to know where to look and what to do.

### Example Configuration (`config.yaml`)

```yaml
# The main folder containing your projects
base_directory: "/home/user/projects"

# The command you want to run in each folder
command_to_run: "git diff >> ../daily_diff.txt"

# Folders you want the tool to skip
excluded_folders:
  - "node_modules"
  - ".git"
  - "venv"
```

## Options

- `--dry-run`: Shows which folders would be processed and which command would run without actually doing it. Use this to test your setup safely.
- `--quiet`: Hides status messages and progress bars.

## How it Works

1. **Find Folders:** The tool looks inside your `base_directory` and finds every sub-folder.
2. **Filter:** It removes any folders you listed in `excluded_folders`.
3. **Execute:** It enters each remaining folder and runs your `command_to_run`.
4. **Report:** It shows you the results of each command or any errors that occurred.

## Examples

**Test your configuration without running commands:**
```bash
python cmdrunner.py my_setup.yaml --dry-run
```

**Run a command across your projects quietly:**
```bash
python cmdrunner.py config.yaml --quiet
```
