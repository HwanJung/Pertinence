#!/usr/bin/env python3
"""Validate approved local inputs and create a benchmark evidence session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pertinence.serving.session_gate import (
    REPOSITORY_ROOT,
    create_session,
    parse_assignments,
    read_env,
    snapshot_worktree,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path)
    parser.add_argument(
        "--print-worktree-snapshot",
        action="store_true",
        help="print the current repository snapshot for independent approval, then exit",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="supply an input directly; cannot duplicate a value from --env",
    )
    args = parser.parse_args()
    if args.print_worktree_snapshot:
        if args.env is not None or args.set:
            parser.error("--print-worktree-snapshot cannot be combined with session inputs")
        print(json.dumps(snapshot_worktree(REPOSITORY_ROOT), sort_keys=True, indent=2))
        return 0
    if args.env is None and not args.set:
        parser.error("at least one of --env or --set is required")
    try:
        values = read_env(args.env) if args.env is not None else {}
        direct = parse_assignments(args.set)
        duplicates = sorted(values.keys() & direct.keys())
        if duplicates:
            raise ValueError(f"inputs supplied twice: {', '.join(duplicates)}")
        receipt = create_session({**values, **direct})
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
