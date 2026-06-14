#!/usr/bin/env python3
"""Backward-compatible wrapper for scripts/knowledge.py refresh."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent / "knowledge.py"
    argv = sys.argv[1:]
    source = "all"
    if "--website" in argv:
        source = "website"
        argv = [arg for arg in argv if arg != "--website"]
    elif "--pdfs" in argv:
        source = "pdfs"
        argv = [arg for arg in argv if arg != "--pdfs"]
    elif "--all" in argv:
        argv = [arg for arg in argv if arg != "--all"]

    command = [sys.executable, str(script), "refresh", source, *argv]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
