"""Dependency-light entry point for the CPU-only serving container."""

from __future__ import annotations

import argparse

from .serving_cli import add_commands, run_command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_commands(subparsers)
    args = parser.parse_args(argv)
    try:
        return run_command(args)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"{args.command} failed: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
