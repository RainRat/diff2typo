# cmdrunner.py

**Purpose:** Executes a specific shell command across every subdirectory within a base folder. This is ideal for running `git diff`, `npm install`, or static analysis tools across a monorepo or a folder of multiple projects.

## Usage

```bash
python cmdrunner.py config.yaml
```

## Configuration

The tool requires a YAML config file:

```yaml
base_directory: "/home/user/projects"
command_to_run: "git diff >> ../daily_diff.txt"
excluded_folders:
  - "node_modules"
  - ".git"
  - "venv"
```

## Options

- `--dry-run`: Prints which directories would be processed and what command would be run, without actually executing it.
- `--quiet`: Suppresses informational logs.
