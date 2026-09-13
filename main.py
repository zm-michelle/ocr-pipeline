#!/usr/bin/env python3
"""Thin wrapper so `python main.py --mode ...` keeps working. Logic lives in ocr/cli.py."""

from ocr.cli import main

if __name__ == "__main__":
    main()
