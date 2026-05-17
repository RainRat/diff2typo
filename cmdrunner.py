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

    if "main_folder" in config and not isinstance(config.get("main_folder"), str):
        errors.append("'main_folder' must be a string.")

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
    main_folder: str,
    command: str,
    excluded_folders: Optional[List[str]] = None,
    dry_run: bool = False,
    quiet: bool = False,
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

    iterator = tqdm(directories, desc="Processing folders", unit="folder", disable=dry_run or quiet)

    # Iterate through each item in the main folder
    for item in iterator:
        item_path = os.path.join(main_folder, item)
        current_command = command.replace("{}", item)

        if dry_run:
            logging.warning(f"Dry run: would run command '{current_command}' in '{item}'")
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
                text=True  # Automatically decode to string
            )
            logging.info(f"Command output for '{item}':\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"The command failed in '{item}':\n{e.stderr}")

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
        help='The path to your YAML configuration file.'
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
    except ConfigError as exc:
        logging.error(str(exc))
        sys.exit(1)

    # Extract configuration parameters with defaults
    # Support both 'main_folder' and the legacy 'base_directory'
    main_folder = config.get('main_folder') or config.get('base_directory', '')
    command_to_run = config.get('command_to_run', '')
    excluded = config.get('excluded_folders', [])

    # Run the command in the specified folders
    run_command_in_folders(
        main_folder,
        command_to_run,
        excluded,
        dry_run=args.dry_run,
        quiet=args.quiet,
    )

if __name__ == "__main__":
    main()
