# gentypos.py

**Purpose:** Generates synthetic typos from a list of words. It simulates human error using keyboard adjacency maps, character deletion, transposition, and duplication.

## Usage

```bash
python gentypos.py --config gentypos.yaml
```

## Configuration (`gentypos.yaml`)

This tool relies on a YAML configuration file.

### Structure

```yaml
input_file: "words_small.txt"      # Source words to mess up
dictionary_file: "words_large.txt" # Valid words (to ensure we don't generate real words)
output_file: "typos_mega.toml"     # Output destination
output_format: "table"             # arrow, csv, table, or list

typo_types:
  deletion: true       # e.g., "word" -> "wrd"
  transposition: true  # e.g., "word" -> "wrod"
  replacement: true    # e.g., "word" -> "wprd" (adjacent key)
  duplication: true    # e.g., "word" -> "woord"

word_length:
  min_length: 8
  max_length: null
```

## Logic

1. **Generation:** Applies configured error types to every word in the input list.
2. **Filtering:** Checks generated typos against the `dictionary_file`. If a generated typo is actually a valid word (e.g., typing "form" instead of "from"), it is discarded to avoid flagging correct code as incorrect.
