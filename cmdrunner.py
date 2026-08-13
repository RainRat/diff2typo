import os
import subprocess
import shlex
import csv
import json
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _YAML_AVAILABLE = False
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter([])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def update(self, n=1): pass
        def close(self): pass
        def set_description(self, desc=None, refresh=True): pass
        def set_postfix(self, *args, **kwargs): pass


VERSION = "1.1.0"


# ANSI Color Codes
BLUE = "\033[1;34m"
GREEN = "\033[1;32m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Disable colors if not running in a terminal or if NO_COLOR is set
if not sys.stdout.isatty() or os.environ.get('NO_COLOR'):
    BLUE = GREEN = RED = YELLOW = RESET = BOLD = ""


class MinimalFormatter(logging.Formatter):
    """A logging formatter that removes prefixes for INFO level messages."""

    LEVEL_COLORS = {
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            return record.getMessage()

        levelname = record.levelname
        # Colorize the level name if stderr is a terminal and color is available
        if sys.stderr.isatty() and levelname:
            color = self.LEVEL_COLORS.get(record.levelno)
            if color:
                levelname = f"{color}{levelname}{RESET}"

        return f"{levelname}: {record.getMessage()}"


class ConfigError(Exception):
    """Raised when a configuration file is invalid."""

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the YAML configuration file.
    """
    if not _YAML_AVAILABLE:
        logging.error("PyYAML is not installed. Install via 'pip install PyYAML' to use cmdrunner.")
        sys.exit(1)

    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Error parsing YAML file '{config_path}': {exc}")

    if not isinstance(config, dict):
        raise ConfigError(f"Configuration file '{config_path}' is empty or malformed.")

    errors = []
    # Support both 'main_folder' and the legacy 'base_directory'
    main_folder = config.get("main_folder") or config.get("base_directory")
    if not main_folder:
        errors.append("Missing required configuration field: 'main_folder'.")

    if not config.get("command_to_run"):
        errors.append("Missing required configuration field: 'command_to_run'.")

    for field in ["main_folder", "base_directory", "command_to_run"]:
        if field in config and not isinstance(config[field], str):
            errors.append(f"'{field}' must be a string.")

    if "excluded_folders" in config and not isinstance(config["excluded_folders"], list):
        errors.append("'excluded_folders' must be a list if provided.")

    if "fail_fast" in config and not isinstance(config["fail_fast"], bool):
        errors.append("'fail_fast' must be a boolean.")

    if "timeout" in config and (isinstance(config["timeout"], bool) or not isinstance(config["timeout"], (int, float))):
        errors.append("'timeout' must be a number.")

    if "if_exists" in config and not isinstance(config["if_exists"], str):
        errors.append("'if_exists' must be a string.")

    if "jobs" in config and (isinstance(config["jobs"], bool) or not isinstance(config["jobs"], int) or config["jobs"] < 1):
        errors.append("'jobs' must be an integer of 1 or more.")

    if "if_not_exists" in config and not isinstance(config["if_not_exists"], str):
        errors.append("'if_not_exists' must be a string.")

    if errors:
        raise ConfigError(" ".join(errors))

    return config

def run_command_in_folders(
    main_folder: str,
    command: str,
    excluded_folders: Optional[List[str]] = None,
    dry_run: bool = False,
    quiet: bool = False,
    fail_fast: bool = False,
    timeout: Optional[float] = None,
    output_file: Optional[str] = None,
    output_format: Optional[str] = None,
    if_exists: Optional[str] = None,
    if_not_exists: Optional[str] = None,
    jobs: int = 1,
) -> None:
    """
    Run a specified command in each folder within the main folder,
    excluding specified folders.
    """
    excluded_folders = excluded_folders or []

    if not os.path.isdir(main_folder):
        logging.error(f"The main folder '{main_folder}' does not exist or is not a folder.")
        sys.exit(1)

    directories = sorted([
        item for item in os.listdir(main_folder)
        if os.path.isdir(os.path.join(main_folder, item)) and item not in excluded_folders
    ])

    if if_exists:
        directories = [
            item for item in directories
            if os.path.exists(os.path.join(main_folder, item, if_exists))
        ]

    if if_not_exists:
        directories = [
            item for item in directories
            if not os.path.exists(os.path.join(main_folder, item, if_not_exists))
        ]

    report_data = []

    if jobs > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        futures = {}
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            for item in directories:
                item_path = os.path.join(main_folder, item)
                current_command = command.replace("{}", shlex.quote(item))

                if dry_run:
                    def run_dry(it, cmd):
                        return {
                            "folder": it,
                            "command": cmd,
                            "status": "dry-run",
                            "return_code": 0,
                            "stdout": "",
                            "stderr": "",
                        }
                    futures[executor.submit(run_dry, item, current_command)] = item
                else:
                    def run_cmd(it, it_path, cmd):
                        try:
                            result = subprocess.run(
                                cmd,
                                cwd=it_path,
                                shell=True,
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                timeout=timeout,
                            )
                            return {
                                "folder": it,
                                "command": cmd,
                                "status": "success",
                                "return_code": 0,
                                "stdout": result.stdout,
                                "stderr": result.stderr,
                            }
                        except subprocess.TimeoutExpired as e:
                            return {
                                "folder": it,
                                "command": cmd,
                                "status": "timeout",
                                "return_code": -1,
                                "stdout": e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or ""),
                                "stderr": e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or ""),
                            }
                        except subprocess.CalledProcessError as e:
                            return {
                                "folder": it,
                                "command": cmd,
                                "status": "failed",
                                "return_code": e.returncode,
                                "stdout": e.stdout or "",
                                "stderr": e.stderr or "",
                            }

                    futures[executor.submit(run_cmd, item, item_path, current_command)] = item

            iterator = as_completed(futures)
            if not (dry_run or quiet):
                iterator = tqdm(iterator, total=len(futures), desc="Processing folders", unit="folder")

            for future in iterator:
                item = futures[future]
                try:
                    result_item = future.result()
                    if result_item["status"] == "dry-run":
                        logging.warning(f"Dry run: would run command '{result_item['command']}' in '{result_item['folder']}'")
                    elif result_item["status"] == "success":
                        logging.info(f"Running command in: {item}")
                        logging.info(f"Command output for '{item}':\n{result_item['stdout']}")
                    elif result_item["status"] == "timeout":
                        logging.info(f"Running command in: {item}")
                        logging.error(f"The command in '{item}' timed out after {timeout} seconds.")
                        if fail_fast:
                            for f in futures:
                                f.cancel()
                            sys.exit(1)
                    elif result_item["status"] == "failed":
                        logging.info(f"Running command in: {item}")
                        logging.error(f"The command failed in '{item}':\n{result_item['stderr']}")
                        if fail_fast:
                            for f in futures:
                                f.cancel()
                            sys.exit(1)

                    report_data.append(result_item)
                except Exception as e:
                    logging.error(f"An unexpected error occurred in task '{item}': {e}")
                    if fail_fast:
                        sys.exit(1)
    else:
        iterator = tqdm(directories, desc="Processing folders", unit="folder", disable=dry_run or quiet)

        # Iterate through each item in the main folder
        for item in iterator:
            item_path = os.path.join(main_folder, item)
            current_command = command.replace("{}", shlex.quote(item))

            if dry_run:
                logging.warning(f"Dry run: would run command '{current_command}' in '{item}'")
                report_data.append({
                    "folder": item,
                    "command": current_command,
                    "status": "dry-run",
                    "return_code": 0,
                    "stdout": "",
                    "stderr": "",
                })
                continue

            logging.info(f"Running command in: {item}")

            # Run the command in the directory
            try:
                result = subprocess.run(
                    current_command,
                    cwd=item_path,
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )
                logging.info(f"Command output for '{item}':\n{result.stdout}")
                report_data.append({
                    "folder": item,
                    "command": current_command,
                    "status": "success",
                    "return_code": 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                })
            except subprocess.TimeoutExpired as e:
                logging.error(f"The command in '{item}' timed out after {timeout} seconds.")
                report_data.append({
                    "folder": item,
                    "command": current_command,
                    "status": "timeout",
                    "return_code": -1,
                    "stdout": e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or ""),
                    "stderr": e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or ""),
                })
                if fail_fast:
                    sys.exit(1)
            except subprocess.CalledProcessError as e:
                logging.error(f"The command failed in '{item}':\n{e.stderr}")
                report_data.append({
                    "folder": item,
                    "command": current_command,
                    "status": "failed",
                    "return_code": e.returncode,
                    "stdout": e.stdout or "",
                    "stderr": e.stderr or "",
                })
                if fail_fast:
                    sys.exit(1)

    # Sort report_data by folder name to ensure deterministic order (especially important for parallel mode)
    report_data.sort(key=lambda x: x["folder"])

    if output_file:
        # Determine the format
        fmt = output_format
        if not fmt:
            ext = os.path.splitext(output_file)[1].lower().lstrip('.')
            if ext in ['json', 'csv', 'txt']:
                fmt = ext
            else:
                fmt = 'txt'

        try:
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                if fmt == 'json':
                    json.dump(report_data, f, indent=2)
                elif fmt == 'csv':
                    writer = csv.DictWriter(f, fieldnames=["folder", "command", "status", "return_code", "stdout", "stderr"])
                    writer.writeheader()
                    for row in report_data:
                        writer.writerow(row)
                else:  # txt
                    for row in report_data:
                        f.write(f"Folder: {row['folder']}\n")
                        f.write(f"Command: {row['command']}\n")
                        f.write(f"Status: {row['status']}\n")
                        f.write(f"Return Code: {row['return_code']}\n")
                        if row['stdout'].strip():
                            f.write("Stdout:\n")
                            f.write(row['stdout'])
                            if not row['stdout'].endswith('\n'):
                                f.write('\n')
                        if row['stderr'].strip():
                            f.write("Stderr:\n")
                            f.write(row['stderr'])
                            if not row['stderr'].endswith('\n'):
                                f.write('\n')
                        f.write("=" * 40 + "\n")
            logging.info(f"Execution report saved to '{output_file}' in {fmt} format.")
        except Exception as e:
            logging.error(f"Failed to write report to '{output_file}': {e}")
            sys.exit(1)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments to get the path to the YAML configuration file.
    """
    parser = argparse.ArgumentParser(
        description=f"{BOLD}Run a command in every folder within a main folder, skipping specific folders.{RESET}",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"""{BLUE}Dynamic Commands:{RESET}
  You can use {BOLD}{{}}{RESET} as a placeholder in your command. It will be replaced
  with the name of the folder currently being processed.

{BLUE}Examples:{RESET}
  {GREEN}python cmdrunner.py config.yaml{RESET}
  {GREEN}python cmdrunner.py my_setup.yaml --dry-run{RESET}
""",
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {VERSION}'
    )

    # Configuration Group
    config_group = parser.add_argument_group(f"{BLUE}CONFIGURATION{RESET}")
    config_group.add_argument(
        'config',
        metavar='CONFIG_PATH',
        type=str,
        nargs='?',
        help='The path to your YAML configuration file. If omitted, the tool automatically falls back to loading "cmdrunner.yaml" from the current working directory if it exists.'
    )

    # Direct Execution / Overrides Group
    direct_group = parser.add_argument_group(f"{BLUE}CLI OVERRIDES / DIRECT OPTIONS{RESET}")
    direct_group.add_argument(
        '-m', '--main-folder',
        type=str,
        help='The main folder containing your projects. Overrides config file if provided.'
    )
    direct_group.add_argument(
        '-b', '--base-directory',
        type=str,
        help='Legacy alias for main folder. Overrides config file if provided.'
    )
    direct_group.add_argument(
        '-c', '--command-to-run',
        dest='command_to_run',
        type=str,
        help='The command you want to run in each folder. Overrides config file if provided.'
    )
    direct_group.add_argument(
        '-e', '--excluded-folders',
        dest='excluded_folders',
        nargs='+',
        help='Folders you want the tool to skip. Overrides config file if provided.'
    )
    direct_group.add_argument(
        '--if-exists',
        type=str,
        help='Only run the command in folders that contain this specific file or path (e.g., package.json).'
    )
    direct_group.add_argument(
        '--if-not-exists',
        type=str,
        help='Only run the command in folders that do NOT contain this specific file or path (e.g., initialized.log).'
    )

    # Execution Options Group
    options_group = parser.add_argument_group(f"{BLUE}EXECUTION OPTIONS{RESET}")
    options_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Show which folders would be checked without executing the command.'
    )
    options_group.add_argument(
        '--quiet',
        action='store_true',
        help='Hide progress bars and status messages.'
    )
    options_group.add_argument(
        '--fail-fast',
        action='store_true',
        default=None,
        help='Stop execution immediately if any command fails.'
    )
    options_group.add_argument(
        '--timeout',
        type=float,
        help='The maximum execution time in seconds for the command in each folder.'
    )
    options_group.add_argument(
        '-j', '--jobs',
        type=int,
        help='The number of concurrent jobs/threads to run in parallel. Defaults to 1 (sequential).'
    )

    # Output Options Group
    output_group = parser.add_argument_group(f"{BLUE}OUTPUT OPTIONS{RESET}")
    output_group.add_argument(
        '-o', '--output',
        type=str,
        help='Where to save the execution report. If not provided, no report is saved.'
    )
    output_group.add_argument(
        '-f', '--format',
        choices=['json', 'csv', 'txt'],
        help='Choose the format for the output report (default: txt).'
    )

    return parser.parse_args()

def main() -> None:
    # Parse command-line arguments
    args = parse_arguments()
    config_file = args.config

    log_level = logging.WARNING if args.quiet else logging.INFO
    # Use a custom handler and formatter to keep output clean
    handler = logging.StreamHandler()
    handler.setFormatter(MinimalFormatter('%(levelname)s: %(message)s'))
    logging.basicConfig(level=log_level, handlers=[handler])

    config = {}
    if not config_file:
        # Friction reduction: if no configuration file path is specified via the command line,
        # and direct run parameters (both main folder and command to run) are not fully provided as CLI overrides,
        # automatically fall back to loading 'cmdrunner.yaml' from the current working directory if it exists.
        has_direct = bool(args.main_folder or args.base_directory) and bool(args.command_to_run)
        if not has_direct:
            default_config_path = "cmdrunner.yaml"
            if os.path.isfile(default_config_path):
                config_file = default_config_path
                logging.info(f"No configuration file specified and direct options incomplete. Falling back to loading '{default_config_path}'...")

    if config_file:
        # Load configuration
        try:
            config = load_config(config_file)
        except FileNotFoundError:
            logging.error(f"Configuration file '{config_file}' not found.")
            sys.exit(1)
        except ConfigError as exc:
            logging.error(str(exc))
            sys.exit(1)

    # Extract configuration parameters with defaults
    # Support both 'main_folder' and the legacy 'base_directory', allowing CLI overrides
    main_folder = args.main_folder or args.base_directory or config.get('main_folder') or config.get('base_directory', '')
    command_to_run = args.command_to_run or config.get('command_to_run', '')
    excluded = args.excluded_folders if args.excluded_folders is not None else config.get('excluded_folders', [])

    # Prioritize CLI values over config file values
    fail_fast = args.fail_fast if args.fail_fast is not None else config.get('fail_fast', False)
    timeout = args.timeout if args.timeout is not None else config.get('timeout', None)
    jobs = args.jobs if args.jobs is not None else config.get('jobs', 1)
    if_exists = args.if_exists or config.get('if_exists', None)
    if_not_exists = args.if_not_exists or config.get('if_not_exists', None)

    # Validate that required options are present
    errors = []
    if not main_folder:
        errors.append("main_folder")
    if not command_to_run:
        errors.append("command_to_run")

    if errors:
        logging.error(f"Missing required option(s): {', '.join(errors)}.")
        sys.exit(1)

    # Run the command in the specified folders
    run_command_in_folders(
        main_folder,
        command_to_run,
        excluded,
        dry_run=args.dry_run,
        quiet=args.quiet,
        fail_fast=fail_fast,
        timeout=timeout,
        output_file=args.output,
        output_format=args.format,
        if_exists=if_exists,
        if_not_exists=if_not_exists,
        jobs=jobs,
    )

if __name__ == "__main__":
    main()
