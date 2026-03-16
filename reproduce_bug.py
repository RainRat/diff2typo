import typostats
import logging
import sys

# Setup logging to see the output
logging.basicConfig(level=logging.INFO)

try:
    typostats.print_processing_stats(10, 5, "item")
    print("Success")
except NameError as e:
    print(f"Caught expected NameError: {e}")
except Exception as e:
    print(f"Caught unexpected exception: {type(e).__name__}: {e}")
