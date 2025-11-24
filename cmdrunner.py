import os
import subprocess
import yaml
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional


class ConfigError(Exception):
    """Raised when a configuration file is invalid."""

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the YAML configuration file.
    """
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

def run_command_in_folders(base_dir: str, command: str, excluded_folders: Optional[List[str]] = None, dry_run: bool = False) -> None:
    """
    Run a specified command in each subdirectory of the base directory,
    excluding specified folders.
    """
    if excluded_folders is None:
        excluded_folders = []

    if not os.path.isdir(base_dir):
        logging.error(f"The base directory '{base_dir}' does not exist or is not a directory.")
        sys.exit(1)

    # Iterate through each item in the base directory
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        # Check if the item is a directory and not in the excluded list
        if os.path.isdir(item_path) and item not in excluded_folders:
            if dry_run:
                logging.info(f"Dry run: would run command '{command}' in '{item_path}'")
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
        description="Run a specified command in each subdirectory of a base directory, excluding certain folders."
    )
    parser.add_argument(
        'config',
        metavar='CONFIG_PATH',
        type=str,
        help='Path to the YAML configuration file.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show which directories would be processed without executing the command.'
    )
    return parser.parse_args()

def main() -> None:
    # Parse command-line arguments
    args = parse_arguments()
    config_file = args.config

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    # Validate required configuration parameters
    if not base_directory:
        logging.error("'base_directory' is not specified in the configuration.")
        sys.exit(1)

    if not command_to_run:
        logging.error("'command_to_run' is not specified in the configuration.")
        sys.exit(1)

    # Run the command in the specified folders
    run_command_in_folders(base_directory, command_to_run, excluded, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
