"""Model adapters are intentionally imported lazily by the CLI.

This keeps lightweight commands such as `inspect` usable before the heavy LLM/VLM
runtime dependencies are installed.
"""
