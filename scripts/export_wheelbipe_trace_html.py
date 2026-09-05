#!/usr/bin/env -S uv run --script
# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Adapted from wheeled-legged_RL/scripts/utils/export_velocity_trace_html.py at
# the pinned revision recorded in THIRD_PARTY_NOTICES.md.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================
"""Export a WheelBipe velocity/reward trace CSV to interactive offline HTML."""

from __future__ import annotations

import argparse
from pathlib import Path

from unilab.visualization.wheelbipe_trace import export_wheelbipe_trace_html


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("-o", "--html-path", type=Path, default=None)
    parser.add_argument(
        "--reward-signs-json",
        type=Path,
        default=None,
        help="JSON object containing either reward_* signs or owner reward scales",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    destination = export_wheelbipe_trace_html(
        args.csv_path,
        html_path=args.html_path,
        reward_signs_path=args.reward_signs_json,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
