#!/usr/bin/env python3
"""Standalone entry point for IFS Cloud MCP Server.

This script ensures the server runs in a clean environment without asyncio conflicts.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the IFS Cloud MCP Server as a subprocess to avoid asyncio conflicts."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent

    # Build the command to run the main server
    main_module = script_dir / "main.py"

    # Pass through all command line arguments
    cmd = [sys.executable, str(main_module)] + sys.argv[1:]

    try:
        # Run as subprocess to ensure clean asyncio environment
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
