"""Command routing for serving, trace, promotion, and benchmark operations."""

from __future__ import annotations

import argparse

from . import benchmark, promotion, runtime, trace


def add_commands(subparsers: argparse._SubParsersAction) -> None:
    promotion.add_commands(subparsers)
    trace.add_commands(subparsers)
    runtime.add_commands(subparsers)
    benchmark.add_commands(subparsers)


def run_command(args: argparse.Namespace) -> int:
    if args.serving_action.startswith("promotion-"):
        return promotion.run_command(args)
    if args.serving_action.startswith("trace-"):
        return trace.run_command(args)
    if args.serving_action.startswith("serve-"):
        return runtime.run_command(args)
    return benchmark.run_command(args)


__all__ = ["add_commands", "run_command"]
