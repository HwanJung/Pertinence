"""Compatibility entry point for :mod:`pertinence.pipeline.cli`."""

from .pipeline.cli import add_commands, main, run_command

__all__ = ["add_commands", "main", "run_command"]


if __name__ == "__main__":
    raise SystemExit(main())
