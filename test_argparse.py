import argparse
import sys

BLUE = "\033[1;34m"
RESET = "\033[0m"

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
io_group = parser.add_argument_group(
    title=f"{BLUE}INPUT/OUTPUT OPTIONS{RESET}",
    description=f"{BLUE}{'─' * 55}{RESET}"
)
io_group.add_argument('--foo', help='foo help')
parser.print_help()
