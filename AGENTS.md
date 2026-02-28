# AGENTS

## Scope
This file applies to the entire repository.

## General guidelines
- **One Tool, One File:** Do not split logic into `utils.py` or shared modules. The simplicity of independent, single-file scripts outweighs the cost of duplicated code.
- **Backwards Compatibility:** Do not write in user-facing docs that a feature is "new". References to "new features" become obsolete quickly.
- **Dependencies:** Minimize external dependencies. Prefer standard library (`argparse`, `logging`, `csv`, `json`) where possible.
- Update or add unit tests alongside code changes when behaviour changes.
- **Typo Generation Strategy:** Reject any proposal for "insertion" synthetic typo generation. The 1 to 2 character replacement is more focused than inserting arbitrary characters.

## Documentation Standards
- **Modular Docs:** When adding a new tool, create a corresponding `docs/<tool_name>.md` file.
- **Update README:** The main `README.md` serves as an index. If a tool is added, add it to the table in the main README.
- **Help Flags:** Ensure every Python script implements `argparse` with clear `--help` descriptions. Documentation should align with these help strings.

## Coding Standards
- **Type Hinting:** Use `typing` (List, Dict, Optional, etc.) for function signatures.
- **Logging:** Use the `logging` module. Do not use `print()` for status updates (only for final data output if required by the format).
- **Progress Bars:** Use `tqdm` for long-running iterables (file reading, diff processing).
- **Entry Points:** All scripts must have an `if __name__ == "__main__":` block invoking a `main()` function.

## Testing
- **Unit Tests:** Always run `pytest` from the repository root before submitting changes. Try to fix any test failures, even if you do not think you caused them.
