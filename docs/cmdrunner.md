# cmdrunner.py

**cmdrunner.py** runs a specific command in every folder within a main folder. This is useful for running tools like `git diff` or `npm install` across many different projects at once.

## Prerequisites

- **Python 3.10 or newer**
- **PyYAML:** This package is required to read your configuration file. You can install it using:
  ```bash
  pip install PyYAML
  ```

## Usage

You can run the tool either by providing a YAML configuration file or by specifying the parameters directly on the command line.

**Using a configuration file:**
```bash
python cmdrunner.py config.yaml
```

**Direct execution using command-line arguments:**
```bash
python cmdrunner.py -m /home/user/projects -c "git status"
```

**Automatic Fallback:**

If you run the tool without arguments or a configuration file, it automatically loads `cmdrunner.yaml` from the current working directory if present.

## Configuration

The tool uses a YAML file to know where to look and what to do. Both `main_folder` and the legacy key `base_directory` are fully supported to specify the folder to scan.

### Example Configuration (`config.yaml`)

```yaml
# The main folder containing your projects
main_folder: "/home/user/projects"

# The command you want to run in each folder
command_to_run: "git diff >> ../daily_diff.txt"

# Folders you want the tool to skip
excluded_folders:
  - "node_modules"
  - ".git"
  - "venv"

# (Optional) Specific folders you want to run the command on
included_folders:
  - "my-app-1"
  - "my-app-2"

# (Optional) Stop execution immediately if any command fails or times out
stop_on_first_error: false
fail_fast: false

# (Optional) The maximum execution time in seconds for the command in each folder
timeout: 10.5

# (Optional) Only run the command in folders that contain this specific file or path
if_exists: "package.json"

# (Optional) Only run the command in folders that do NOT contain this specific file or path
if_not_exists: "initialized.log"

# (Optional) Run commands concurrently using this many jobs
jobs: 4
```

## Options

- `CONFIG_PATH`, `-C`, `--config`: (Optional) The path to your YAML configuration file. If you do not specify this, the tool automatically loads `cmdrunner.yaml` from your current directory if it exists.
- `-m`, `--main-folder`: The main folder containing your projects. This overrides the configuration file.
- `-b`, `--base-directory`: Legacy name for the main folder. This overrides the configuration file.
- `-c`, `--command-to-run`: The command you want to run in each folder. This overrides the configuration file.
- `-e`, `--excluded-folders`: A list of folders you want the tool to skip. This overrides the configuration file.
- `-i`, `--included-folders`: A list of folders you want to run the command on. This overrides the configuration file.
- `-n`, `--dry-run`: Show which folders the tool will check without running any commands. Use this to test your setup safely.
- `-q`, `--quiet`: Hide status messages and progress bars.
- `-s`, `--stop-on-first-error`, `--fail-fast`: Stop running commands immediately if any command fails. This overrides the configuration file.
- `-t`, `--timeout`: Set the maximum time in seconds for the command to run in each folder. This overrides the configuration file.
- `-x`, `--if-exists`: Only run the command in folders that contain this file or path (for example, `package.json`). This overrides the configuration file.
- `-X`, `--if-not-exists`: Only run the command in folders that do not contain this file or path (for example, `initialized.log`). This overrides the configuration file.
- `-j`, `--jobs`: Run commands concurrently using this many jobs. This overrides the configuration file.
- `-o`, `--output`: Save the execution report to this file. If you do not specify this, the tool will not save a report.
- `-f`, `--format`: Choose the format for the output report (`json`, `csv`, `txt`, `markdown`, `md`, `yaml`, or `yml`). If you do not specify this, the tool detects the format from the output file's extension.

## Dynamic Commands

You can use `{}` as a placeholder in your `command_to_run`. The tool will replace this with the name of the folder currently being processed.

This is useful for creating unique output files for each project:

```yaml
main_folder: "/home/user/projects"
command_to_run: "git diff > ../{}-changes.diff"
```

In this example, if the tool processes a folder named `my-web-app`, it will run the command `git diff > ../my-web-app-changes.diff`.

## How it Works

1. **Find Folders:** The tool looks inside your `main_folder` and finds every sub-folder.
2. **Filter:** It removes any folders you listed in `excluded_folders`, limits to those folders specified in `included_folders` (if provided), and filters remaining folders to only those containing the specified `--if-exists` file or path (if provided) and not containing the specified `--if-not-exists` file or path (if provided).
3. **Execute:** It enters each remaining folder and runs your `command_to_run`.
4. **Report:** It shows you the results of each command or any errors that occurred.

## Examples

**Run a command across your projects directly without a config file:**
```bash
python cmdrunner.py -m /home/user/projects -c "git status"
```

**Override configuration file settings from the command line:**
```bash
python cmdrunner.py config.yaml -c "git pull"
```

**Test your configuration without running commands:**
```bash
python cmdrunner.py my_setup.yaml --dry-run
```

**Run a command across your projects quietly:**
```bash
python cmdrunner.py config.yaml --quiet
```

**Only run commands in projects containing a `package.json` file:**
```bash
python cmdrunner.py --main-folder /home/user/projects --command-to-run "npm run build" --if-exists package.json
```

**Only run commands in projects that do NOT contain a `setup.log` file:**
```bash
python cmdrunner.py --main-folder /home/user/projects --command-to-run "bash setup.sh && touch setup.log" --if-not-exists setup.log
```

**Only run commands in specific folders:**
```bash
python cmdrunner.py --main-folder /home/user/projects --command-to-run "npm run build" --included-folders proj1 proj2
```

**Run commands concurrently across multiple projects:**
```bash
python cmdrunner.py --main-folder /home/user/projects --command-to-run "npm test" -j 4
```

**Save an execution report in Markdown format:**
```bash
python cmdrunner.py --main-folder /home/user/projects --command-to-run "git status" -o report.md
```

**Save an execution report in YAML format:**
```bash
python cmdrunner.py --main-folder /home/user/projects --command-to-run "git status" -o report.yaml
```
