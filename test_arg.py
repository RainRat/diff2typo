import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--allow-two-char', dest='allow_two_char', action='store_true')
print(sys.argv)
args = parser.parse_args()
print(args.allow_two_char)
