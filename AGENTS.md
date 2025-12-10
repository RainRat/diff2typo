# AGENTS

## Scope
This file applies to the entire repository.

## General guidelines
- Update or add unit tests alongside code changes when behaviour changes.
- Do not write in user-facing docs that a feature is "new", or write in code comments that a method uses a "new way". Such references become obsolete. Remove them if seen.
- Do not split out logic into say, a new `utils.py`. The simplicity of a one tool, one file approach currently outweighs the downside of duplicated code.

## Testing
- Always run `pytest` from the repository root before submitting changes, unless it is a documentation-only change. Try to fix any test failures, even if you don't think you caused them.