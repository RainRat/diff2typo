import os
import subprocess
import yaml
import sys
import argparse

def load_config(config_path):
    """
    Load the YAML configuration file.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file '{config_path}': {exc}")
        sys.exit(1)

def run_command_in_folders(base_dir, command, excluded_folders=None):
    """
    Run a specified command in each subdirectory of the base directory,
    excluding specified folders.
    """
    if excluded_folders is None:
        excluded_folders = []

    if not os.path.isdir(base_dir):
        print(f"Error: The base directory '{base_dir}' does not exist or is not a directory.")
        sys.exit(1)

    # Iterate through each item in the base directory
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        # Check if the item is a directory and not in the excluded list
        if os.path.isdir(item_path) and item not in excluded_folders:
            print(f"\nRunning command in: {item_path}")
            
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
                print(f"Command output for '{item_path}':\n{result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"Command failed in '{item_path}' with error:\n{e.stderr}")

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
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    config_file = args.config

    # Load configuration
    config = load_config(config_file)

    # Extract configuration parameters with defaults
    base_directory = config.get('base_directory', '')
    command_to_run = config.get('command_to_run', '')
    excluded = config.get('excluded_folders', [])

    # Validate required configuration parameters
    if not base_directory:
        print("Error: 'base_directory' is not specified in the configuration.")
        sys.exit(1)
    
    if not command_to_run:
        print("Error: 'command_to_run' is not specified in the configuration.")
        sys.exit(1)

    # Run the command in the specified folders
    run_command_in_folders(base_directory, command_to_run, excluded)

if __name__ == "__main__":
    main()
