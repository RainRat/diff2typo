import os
import subprocess
import shlex
import csv
import json
import html
import time
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
CYAN = "\033[1;36m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Disable colors if not running in a terminal or if NO_COLOR is set
if not sys.stdout.isatty() or os.environ.get('NO_COLOR'):
    BLUE = GREEN = RED = YELLOW = CYAN = RESET = BOLD = ""


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
        if _should_enable_color(sys.stderr) and levelname:
            color = self.LEVEL_COLORS.get(record.levelno)
            if color:
                levelname = f"{color}{levelname}{RESET}"

        return f"{levelname}: {record.getMessage()}"


class ConfigError(Exception):
    """Raised when a configuration file is invalid."""


def _should_enable_color(stream: Any) -> bool:
    """Check if color should be enabled for a given stream."""
    if os.environ.get('NO_COLOR'):
        return False
    if os.environ.get('FORCE_COLOR'):
        return True
    return hasattr(stream, 'isatty') and stream.isatty()


def _render_visual_bar(percentage: float, max_bar: int = 20) -> str:
    """
    Creates a high-resolution visual bar using Unicode block characters.
    """
    total_blocks = (percentage * max_bar) / 100
    full_blocks = int(total_blocks)
    fraction = total_blocks - full_blocks
    blocks = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
    frac_idx = int(fraction * 8)

    bar = "█" * full_blocks
    if full_blocks < max_bar:
        bar += blocks[frac_idx]
        bar += " " * (max_bar - full_blocks - 1)
    return bar


def _format_execution_summary(
    total_found: int,
    skipped: int,
    processed: int,
    success: int,
    failed: int,
    timeout_count: int,
    dry_run_count: int,
    elapsed_time: float,
    use_color: bool = False,
) -> List[str]:
    """
    Standardizes the "EXECUTION SUMMARY" block with consistent colors and a visual success rate bar.
    """
    c_bold = BOLD if use_color else ""
    c_blue = BLUE if use_color else ""
    c_green = GREEN if use_color else ""
    c_yellow = YELLOW if use_color else ""
    c_red = RED if use_color else ""
    c_cyan = CYAN if use_color else ""
    c_reset = RESET if use_color else ""

    padding = "  "
    label_width = 35
    report = []

    report.append(f"\n{padding}{c_bold}{c_blue}EXECUTION SUMMARY{c_reset}")
    report.append(f"{padding}{c_bold}{c_blue}───────────────────────────────────────────────────────{c_reset}")

    report.append(
        f"  {c_bold}{c_blue}{'Total folders found:':<{label_width}}{c_reset} {c_yellow}{total_found}{c_reset}"
    )
    if skipped > 0:
        report.append(
            f"  {c_bold}{c_blue}{'Folders skipped:':<{label_width}}{c_reset} {c_yellow}{skipped}{c_reset}"
        )
    report.append(
        f"  {c_bold}{c_blue}{'Folders processed:':<{label_width}}{c_reset} {c_yellow}{processed}{c_reset}"
    )

    if processed > 0:
        if dry_run_count > 0:
            report.append(
                f"  {c_bold}{c_blue}{'Dry-runs executed:':<{label_width}}{c_reset} {c_yellow}{dry_run_count}{c_reset}"
            )
        else:
            report.append(
                f"  {c_bold}{c_blue}{'Successful runs:':<{label_width}}{c_reset} {c_green}{success}{c_reset}"
            )
            if failed > 0:
                report.append(
                    f"  {c_bold}{c_blue}{'Failed runs:':<{label_width}}{c_reset} {c_red}{failed}{c_reset}"
                )
            if timeout_count > 0:
                report.append(
                    f"  {c_bold}{c_blue}{'Timed out runs:':<{label_width}}{c_reset} {c_red}{timeout_count}{c_reset}"
                )

            success_rate = (success / processed) * 100
            max_bar = 20
            bar = _render_visual_bar(success_rate, max_bar)
            report.append(
                f"  {c_bold}{c_blue}{'Success rate:':<{label_width}}{c_reset} {c_green}{success_rate:>5.1f}%{c_reset} {c_cyan}{bar}{c_reset}"
            )

    report.append(
        f"  {c_bold}{c_blue}{'Total execution time:':<{label_width}}{c_reset} {c_green}{elapsed_time:.3f}s{c_reset}"
    )
    return report


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
        errors.append("The configuration file is missing the required field 'main_folder'.")

    if not config.get("command_to_run"):
        errors.append("The configuration file is missing the required field 'command_to_run'.")

    for field in ["main_folder", "base_directory", "command_to_run"]:
        if field in config and not isinstance(config[field], str):
            errors.append(f"The field '{field}' must be a string.")

    if "excluded_folders" in config and not isinstance(config["excluded_folders"], list):
        errors.append("The field 'excluded_folders' must be a list if you provide it.")

    if "included_folders" in config and not isinstance(config["included_folders"], list):
        errors.append("'included_folders' must be a list if provided.")

    if "fail_fast" in config and not isinstance(config["fail_fast"], bool):
        errors.append("The field 'fail_fast' must be a boolean.")

    if "stop_on_first_error" in config and not isinstance(config["stop_on_first_error"], bool):
        errors.append("The field 'stop_on_first_error' must be a boolean.")

    if "timeout" in config and (isinstance(config["timeout"], bool) or not isinstance(config["timeout"], (int, float))):
        errors.append("The field 'timeout' must be a number.")

    if "if_exists" in config and not isinstance(config["if_exists"], str):
        errors.append("The field 'if_exists' must be a string.")

    if "if_not_exists" in config and not isinstance(config["if_not_exists"], str):
        errors.append("The field 'if_not_exists' must be a string.")

    if "jobs" in config and (isinstance(config["jobs"], bool) or not isinstance(config["jobs"], int) or config["jobs"] < 1):
        errors.append("'jobs' must be an integer of 1 or more.")

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
    included_folders: Optional[List[str]] = None,
    jobs: int = 1,
) -> None:
    """
    Run a specified command in each folder within the main folder,
    excluding specified folders.
    """
    start_time = time.time()
    excluded_folders = excluded_folders or []

    if not os.path.isdir(main_folder):
        logging.error(f"Could not find main folder '{main_folder}'. Please check that the folder path is correct.")
        sys.exit(1)

    directories = sorted([
        item for item in os.listdir(main_folder)
        if os.path.isdir(os.path.join(main_folder, item)) and item not in excluded_folders
    ])

    if included_folders:
        directories = [item for item in directories if item in included_folders]

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

    def run_single_folder(item: str) -> Dict[str, Any]:
        item_path = os.path.join(main_folder, item)
        current_command = command.replace("{}", shlex.quote(item))

        if dry_run:
            logging.warning(f"Dry run: would run command '{current_command}' in '{item}'")
            return {
                "folder": item,
                "command": current_command,
                "status": "dry-run",
                "return_code": 0,
                "stdout": "",
                "stderr": "",
            }

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
            if result.stdout.strip():
                logging.info(f"Command output for '{item}':\n{result.stdout}")
            else:
                logging.info(f"Command executed successfully in '{item}' with no output.")
            return {
                "folder": item,
                "command": current_command,
                "status": "success",
                "return_code": 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired as e:
            stdout_str = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr_str = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            err_msg = f"The command in '{item}' timed out after {timeout} seconds."
            if stderr_str.strip():
                err_msg += f"\nStderr:\n{stderr_str}"
            elif stdout_str.strip():
                err_msg += f"\nStdout:\n{stdout_str}"
            logging.error(err_msg)
            return {
                "folder": item,
                "command": current_command,
                "status": "timeout",
                "return_code": -1,
                "stdout": stdout_str,
                "stderr": stderr_str,
            }
        except subprocess.CalledProcessError as e:
            err_msg = f"The command failed in '{item}' with exit code {e.returncode}."
            stderr_str = e.stderr or ""
            stdout_str = e.stdout or ""
            if stderr_str.strip():
                err_msg += f"\nStderr:\n{stderr_str}"
            elif stdout_str.strip():
                err_msg += f"\nStdout:\n{stdout_str}"
            logging.error(err_msg)
            return {
                "folder": item,
                "command": current_command,
                "status": "failed",
                "return_code": e.returncode,
                "stdout": stdout_str,
                "stderr": stderr_str,
            }

    if jobs > 1 and not dry_run:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            future_to_item = {
                executor.submit(run_single_folder, item): item
                for item in directories
            }
            # Progress bar tracking
            for future in tqdm(
                concurrent.futures.as_completed(future_to_item),
                total=len(future_to_item),
                desc="Processing folders",
                unit="folder",
                disable=quiet,
            ):
                item = future_to_item[future]
                try:
                    result = future.result()
                    report_data.append(result)
                    if fail_fast and result["status"] in ("failed", "timeout"):
                        # Shutdown executor immediately without waiting for other jobs
                        executor.shutdown(wait=False, cancel_futures=True)
                        sys.exit(1)
                except Exception as exc:
                    logging.error(f"An unexpected error occurred in folder '{item}': {exc}")
                    if fail_fast:
                        executor.shutdown(wait=False, cancel_futures=True)
                        sys.exit(1)
    else:
        iterator = tqdm(directories, desc="Processing folders", unit="folder", disable=dry_run or quiet)
        for item in iterator:
            res = run_single_folder(item)
            report_data.append(res)
            if fail_fast and res["status"] in ("failed", "timeout"):
                sys.exit(1)

    # Sort report_data by folder name to ensure consistent deterministic output
    report_data.sort(key=lambda x: x["folder"])

    if output_file:
        # Determine the format
        fmt = output_format
        if not fmt:
            ext = os.path.splitext(output_file)[1].lower().lstrip('.')
            if ext in ['json', 'csv', 'txt']:
                fmt = ext
            elif ext in ['yaml', 'yml']:
                fmt = 'yaml'
            elif ext in ['md', 'markdown']:
                fmt = 'markdown'
            elif ext in ['html', 'htm']:
                fmt = 'html'
            else:
                fmt = 'txt'

        try:
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                if fmt == 'json':
                    json.dump(report_data, f, indent=2)
                elif fmt in ['yaml', 'yml']:
                    if _YAML_AVAILABLE:
                        yaml.safe_dump(report_data, f, default_flow_style=False)
                    else:
                        logging.warning("PyYAML is not installed. Falling back to JSON for YAML report format.")
                        json.dump(report_data, f, indent=2)
                elif fmt == 'csv':
                    writer = csv.DictWriter(f, fieldnames=["folder", "command", "status", "return_code", "stdout", "stderr"])
                    writer.writeheader()
                    for row in report_data:
                        writer.writerow(row)
                elif fmt in ['markdown', 'md']:
                    f.write("# Execution Report\n\n")
                    f.write("| Folder | Command | Status | Return Code |\n")
                    f.write("| :--- | :--- | :--- | :--- |\n")
                    for row in report_data:
                        cmd_escaped = row['command'].replace('|', '\\|')
                        f.write(f"| `{row['folder']}` | `{cmd_escaped}` | `{row['status']}` | `{row['return_code']}` |\n")
                    f.write("\n## Details\n\n")
                    for row in report_data:
                        f.write(f"### `{row['folder']}`\n\n")
                        f.write(f"- **Command:** `{row['command']}`\n")
                        f.write(f"- **Status:** `{row['status']}`\n")
                        f.write(f"- **Return Code:** `{row['return_code']}`\n\n")
                        if row['stdout'].strip():
                            f.write("#### Stdout\n```\n")
                            f.write(row['stdout'])
                            if not row['stdout'].endswith('\n'):
                                f.write('\n')
                            f.write("```\n\n")
                        if row['stderr'].strip():
                            f.write("#### Stderr\n```\n")
                            f.write(row['stderr'])
                            if not row['stderr'].endswith('\n'):
                                f.write('\n')
                            f.write("```\n\n")
                elif fmt in ['html', 'htm']:
                    f.write("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n")
                    f.write("<meta charset=\"UTF-8\">\n")
                    f.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n")
                    f.write("<title>Cmdrunner Execution Report</title>\n")
                    f.write("<style>\n")
                    f.write("body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 2rem; background-color: #f8f9fa; color: #212529; }\n")
                    f.write("h1, h2, h3 { color: #343a40; }\n")
                    f.write("table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }\n")
                    f.write("th, td { text-align: left; padding: 12px 15px; border-bottom: 1px solid #dee2e6; }\n")
                    f.write("th { background-color: #e9ecef; }\n")
                    f.write(".badge { display: inline-block; padding: 0.25em 0.5em; font-size: 0.85em; font-weight: 600; border-radius: 4px; border: 1px solid transparent; }\n")
                    f.write(".badge-success { background-color: #d4edda; color: #155724; border-color: #c3e6cb; }\n")
                    f.write(".badge-failed { background-color: #f8d7da; color: #721c24; border-color: #f5c6cb; }\n")
                    f.write(".badge-timeout { background-color: #fff3cd; color: #856404; border-color: #ffeeba; }\n")
                    f.write(".badge-dry-run { background-color: #e2e3e5; color: #383d41; border-color: #d6d8db; }\n")
                    f.write("pre { background: #212529; color: #f8f9fa; padding: 1rem; border-radius: 4px; overflow-x: auto; font-family: SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 0.9em; }\n")
                    f.write(".detail-card { background: #fff; padding: 1.5rem; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1.5rem; }\n")
                    f.write("</style>\n</head>\n<body>\n")
                    f.write("<h1>Cmdrunner Execution Report</h1>\n")
                    f.write("<table>\n<thead>\n<tr><th>Folder</th><th>Command</th><th>Status</th><th>Return Code</th></tr>\n</thead>\n<tbody>\n")
                    for row in report_data:
                        folder_esc = html.escape(row['folder'])
                        cmd_esc = html.escape(row['command'])
                        status_esc = html.escape(row['status'])
                        badge_cls = f"badge-{status_esc}" if status_esc in ('success', 'failed', 'timeout', 'dry-run') else "badge"
                        f.write(f"<tr><td><code>{folder_esc}</code></td><td><code>{cmd_esc}</code></td><td><span class=\"badge {badge_cls}\">{status_esc}</span></td><td><code>{row['return_code']}</code></td></tr>\n")
                    f.write("</tbody>\n</table>\n")
                    f.write("<h2>Execution Details</h2>\n")
                    for row in report_data:
                        folder_esc = html.escape(row['folder'])
                        cmd_esc = html.escape(row['command'])
                        status_esc = html.escape(row['status'])
                        badge_cls = f"badge-{status_esc}" if status_esc in ('success', 'failed', 'timeout', 'dry-run') else "badge"
                        f.write("<div class=\"detail-card\">\n")
                        f.write(f"<h3>Folder: <code>{folder_esc}</code></h3>\n")
                        f.write(f"<p><strong>Command:</strong> <code>{cmd_esc}</code></p>\n")
                        f.write(f"<p><strong>Status:</strong> <span class=\"badge {badge_cls}\">{status_esc}</span></p>\n")
                        f.write(f"<p><strong>Return Code:</strong> <code>{row['return_code']}</code></p>\n")
                        if row['stdout'].strip():
                            f.write("<h4>Stdout</h4>\n<pre>")
                            f.write(html.escape(row['stdout']))
                            f.write("</pre>\n")
                        if row['stderr'].strip():
                            f.write("<h4>Stderr</h4>\n<pre>")
                            f.write(html.escape(row['stderr']))
                            f.write("</pre>\n")
                        f.write("</div>\n")
                    f.write("</body>\n</html>\n")
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
            logging.error(f"Could not save execution report to '{output_file}'. Please verify that the directory exists and that you have write permissions.")
            sys.exit(1)

    if not quiet:
        elapsed_time = time.time() - start_time
        all_items = os.listdir(main_folder) if os.path.isdir(main_folder) else []
        total_found = len([item for item in all_items if os.path.isdir(os.path.join(main_folder, item))])
        skipped_count = len([item for item in all_items if os.path.isdir(os.path.join(main_folder, item)) and item in excluded_folders])
        processed_count = len(directories)

        if processed_count == 0:
            if total_found == 0:
                logging.warning(f"No subdirectories found in main folder '{main_folder}'.")
            else:
                logging.warning("No subdirectories matched the specified inclusion, exclusion, or existence criteria.")

        success_count = len([r for r in report_data if r["status"] == "success"])
        failed_count = len([r for r in report_data if r["status"] == "failed"])
        timeout_count = len([r for r in report_data if r["status"] == "timeout"])
        dry_run_count = len([r for r in report_data if r["status"] == "dry-run"])

        use_color = _should_enable_color(sys.stderr)
        summary = _format_execution_summary(
            total_found=total_found,
            skipped=skipped_count,
            processed=processed_count,
            success=success_count,
            failed=failed_count,
            timeout_count=timeout_count,
            dry_run_count=dry_run_count,
            elapsed_time=elapsed_time,
            use_color=use_color,
        )
        sys.stderr.write("\n".join(summary) + "\n")


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
        help='The path to your YAML configuration file. If you do not specify this, the tool automatically loads "cmdrunner.yaml" from your current directory if it exists.'
    )
    config_group.add_argument(
        '-C', '--config',
        dest='config_flag',
        type=str,
        help='The path to your YAML configuration file. Overrides positional argument.'
    )

    # Direct Execution / Overrides Group
    direct_group = parser.add_argument_group(f"{BLUE}CLI OVERRIDES / DIRECT OPTIONS{RESET}")
    direct_group.add_argument(
        '-m', '--main-folder',
        type=str,
        help='The main folder containing your projects. This overrides the configuration file.'
    )
    direct_group.add_argument(
        '-b', '--base-directory',
        type=str,
        help='Legacy name for the main folder. This overrides the configuration file.'
    )
    direct_group.add_argument(
        '-c', '--command', '--command-to-run',
        dest='command_to_run',
        type=str,
        help='The command you want to run in each folder. This overrides the configuration file.'
    )
    direct_group.add_argument(
        '-e', '--excluded-folders',
        dest='excluded_folders',
        nargs='+',
        help='Folders you want the tool to skip. This overrides the configuration file.'
    )
    direct_group.add_argument(
        '-i', '--included-folders',
        dest='included_folders',
        nargs='+',
        help='Specific folders you want to run the command on. Overrides config file if provided.'
    )
    direct_group.add_argument(
        '-x', '--if-exists',
        type=str,
        help='Only run the command in folders that contain this file or path (for example, "package.json").'
    )
    direct_group.add_argument(
        '-X', '--if-not-exists',
        type=str,
        help='Only run the command in folders that do not contain this file or path (for example, "initialized.log").'
    )

    # Execution Options Group
    options_group = parser.add_argument_group(f"{BLUE}EXECUTION OPTIONS{RESET}")
    options_group.add_argument(
        '-n', '--dry-run',
        action='store_true',
        help='Show which folders the tool will check without running any command.'
    )
    options_group.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Hide status messages and progress bars.'
    )
    options_group.add_argument(
        '-s', '--stop-on-first-error', '--fail-fast',
        dest='fail_fast',
        action='store_true',
        default=None,
        help='Stop running commands immediately if any command fails.'
    )
    options_group.add_argument(
        '-t', '--timeout',
        type=float,
        help='Set the maximum time in seconds for the command to run in each folder.'
    )
    options_group.add_argument(
        '-j', '--jobs',
        type=int,
        help='Run commands concurrently using this many jobs.'
    )

    # Output Options Group
    output_group = parser.add_argument_group(f"{BLUE}OUTPUT OPTIONS{RESET}")
    output_group.add_argument(
        '-o', '--output',
        type=str,
        help='Save the execution report to this file. If you do not specify this, the tool will not save a report.'
    )
    output_group.add_argument(
        '-f', '--format',
        choices=['json', 'csv', 'txt', 'markdown', 'md', 'yaml', 'yml', 'html', 'htm'],
        help='Choose the format for the output report (default: txt).'
    )

    return parser.parse_args()

def main() -> None:
    # Parse command-line arguments
    args = parse_arguments()
    config_file = args.config_flag or args.config

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
            logging.error(f"Could not find configuration file '{config_file}'. Please check the file path and try again.")
            sys.exit(1)
        except ConfigError as exc:
            logging.error(str(exc))
            sys.exit(1)

    # Extract configuration parameters with defaults
    # Support both 'main_folder' and the legacy 'base_directory', allowing CLI overrides
    main_folder = args.main_folder or args.base_directory or config.get('main_folder') or config.get('base_directory', '')
    command_to_run = args.command_to_run or config.get('command_to_run', '')
    excluded = args.excluded_folders if args.excluded_folders is not None else config.get('excluded_folders', [])
    included = args.included_folders if args.included_folders is not None else config.get('included_folders', None)

    # Prioritize CLI values over config file values
    config_fail_fast = config.get('stop_on_first_error', config.get('fail_fast', False))
    fail_fast = args.fail_fast if args.fail_fast is not None else config_fail_fast
    timeout = args.timeout if args.timeout is not None else config.get('timeout', None)
    if_exists = args.if_exists or config.get('if_exists', None)
    if_not_exists = args.if_not_exists or config.get('if_not_exists', None)
    jobs = args.jobs if args.jobs is not None else config.get('jobs', 1)

    # Validate that required options are present
    errors = []
    if not main_folder:
        errors.append("main_folder")
    if not command_to_run:
        errors.append("command_to_run")

    if errors:
        logging.error(f"Missing required option(s): {', '.join(errors)}. Please provide a main folder (-m/--main-folder) and a command to run (-c/--command-to-run).")
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
        included_folders=included,
        jobs=jobs,
    )

if __name__ == "__main__":
    main()
