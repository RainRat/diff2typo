import os
import subprocess
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _YAML_AVAILABLE = False
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional

from tqdm import tqdm


# ANSI Color Codes
BLUE = "\033[1;34m"
GREEN = "\033[1;32m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Disable colors if not running in a terminal
if not sys.stdout.isatty():
    BLUE = GREEN = RED = YELLOW = RESET = BOLD = ""


class MinimalFormatter(logging.Formatter):
    """A logging formatter that removes prefixes for INFO level messages."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._info_formatter = logging.Formatter('%(message)s')

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            return self._info_formatter.format(record)
        return super().format(record)


class ConfigError(Exception):
    """Raised when a configuration file is invalid."""

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the YAML configuration file.
    """
    if not _YAML_AVAILABLE:
        logging.error("PyYAML is not installed. Install via 'pip install PyYAML' to use cmdrunner.")
        sys.exit(1)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ConfigError(f"Configuration file '{config_path}' is empty or malformed.")

    errors = []
    missing_fields = [field for field in ("base_directory", "command_to_run") if not config.get(field)]
    if missing_fields:
        errors.append(f"Missing required configuration field(s): {', '.join(missing_fields)}.")

    if "base_directory" in config and not isinstance(config.get("base_directory"), str):
        errors.append("'base_directory' must be a string.")

    if "command_to_run" in config and not isinstance(config.get("command_to_run"), str):
        errors.append("'command_to_run' must be a string.")

    if "excluded_folders" in config and not isinstance(config.get("excluded_folders"), list):
        errors.append("'excluded_folders' must be a list if provided.")

    if errors:
        raise ConfigError(" ".join(errors))

    return config

def run_command_in_folders(
    base_dir: str,
    command: str,
    excluded_folders: Optional[List[str]] = None,
    dry_run: bool = False,
    quiet: bool = False,
) -> None:
    """
    Run a specified command in each subdirectory of the base directory,
    excluding specified folders.
    """
    excluded_folders = excluded_folders or []

    if not os.path.isdir(base_dir):
        logging.error(f"The base directory '{base_dir}' does not exist or is not a directory.")
        sys.exit(1)

    directories = [
        item for item in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, item)) and item not in excluded_folders
    ]

    iterator = tqdm(directories, desc="Processing directories", unit="dir", disable=dry_run or quiet)

    # Iterate through each item in the base directory
    for item in iterator:
        item_path = os.path.join(base_dir, item)

        if dry_run:
            logging.warning(f"Dry run: would run command '{command}' in '{item_path}'")
            continue

        logging.info(f"Running command in: {item_path}")

        # Run the command in the directory
        try:
            result = subprocess.run(
                command,
                cwd=item_path,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True  # Automatically decode to string
            )
            logging.info(f"Command output for '{item_path}':\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed in '{item_path}' with error:\n{e.stderr}")

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments to get the path to the YAML configuration file.
    """
    parser = argparse.ArgumentParser(
        description=f"{BOLD}Run a specified command in each subdirectory of a base directory, excluding certain folders.{RESET}",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"""{BLUE}Examples:{RESET}
  {GREEN}python cmdrunner.py config.yaml{RESET}
  {GREEN}python cmdrunner.py my_setup.yaml --dry-run{RESET}
""",
    )

    # Configuration Group
    config_group = parser.add_argument_group(f"{BLUE}CONFIGURATION{RESET}")
    config_group.add_argument(
        'config',
        metavar='CONFIG_PATH',
        type=str,
        help='The path to your YAML configuration file.'
    )

    # Execution Options Group
    options_group = parser.add_argument_group(f"{BLUE}EXECUTION OPTIONS{RESET}")
    options_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Show which directories would be processed without executing the command.'
    )
    options_group.add_argument(
        '--quiet',
        action='store_true',
        help='Hide progress bars and show fewer log messages.'
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

    # Load configuration
    try:
        config = load_config(config_file)
    except FileNotFoundError:
        logging.error(f"Configuration file '{config_file}' not found.")
        sys.exit(1)
    except yaml.YAMLError as exc:
        logging.error(f"Error parsing YAML file '{config_file}': {exc}")
        sys.exit(1)
    except ConfigError as exc:
        logging.error(str(exc))
        sys.exit(1)

    # Extract configuration parameters with defaults
    base_directory = config.get('base_directory', '')
    command_to_run = config.get('command_to_run', '')
    excluded = config.get('excluded_folders', [])

    # Run the command in the specified folders
    run_command_in_folders(
        base_directory,
        command_to_run,
        excluded,
        dry_run=args.dry_run,
        quiet=args.quiet,
    )

if __name__ == "__main__":
    main()
