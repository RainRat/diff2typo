import os
import subprocess
import yaml
import sys
import argparse
import logging

def load_config(config_path):
    """
    Load the YAML configuration file.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        logging.error("Configuration file '%s' not found.", config_path)
        sys.exit(1)
    except yaml.YAMLError as exc:
        logging.error("Error parsing YAML file '%s': %s", config_path, exc)
        sys.exit(1)

def run_command_in_folders(base_dir, command, excluded_folders=None, dry_run=False):
    """
    Run a specified command in each subdirectory of the base directory,
    excluding specified folders.
    """
    if excluded_folders is None:
        excluded_folders = []

    if not os.path.isdir(base_dir):
        logging.error("The base directory '%s' does not exist or is not a directory.", base_dir)
        sys.exit(1)

    # Iterate through each item in the base directory
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        # Check if the item is a directory and not in the excluded list
        if os.path.isdir(item_path) and item not in excluded_folders:
            if dry_run:
                logging.info("Dry run: would run command '%s' in '%s'", command, item_path)
                continue

            logging.info("Running command in: %s", item_path)

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
                logging.info("Command output for '%s':\n%s", item_path, result.stdout)
            except subprocess.CalledProcessError as e:
                logging.error("Command failed in '%s' with error:\n%s", item_path, e.stderr)

def parse_arguments():
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

def main():
    # Parse command-line arguments
    args = parse_arguments()
    config_file = args.config

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Load configuration
    config = load_config(config_file)

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
