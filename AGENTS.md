# AGENTS

## Scope
This file applies to the entire repository.

## General guidelines
- Follow the existing Python coding style in the repository. Prefer clear variable names and short helper functions when logic becomes complex.
- Keep modules importable; do not introduce side effects at import time.
- Update or add unit tests alongside code changes when behaviour changes.

## Testing
- Run `pytest` from the repository root before submitting changes.

## Pull request message
- Begin the summary with a concise bullet list of the key changes.
- Include a "Testing" section listing the commands you executed, even if no tests were run (state "not run" when applicable).
