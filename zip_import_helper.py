#!/usr/bin/env python3
"""
IFS Cloud ZIP Import Helper Script

Quick utility for importing and managing IFS Cloud ZIP files.
Provides an easy-to-use interface for the underlying import functionality.
"""

import argparse
import sys
import subprocess
from pathlib import Path
import json


def run_import_command(
    zip_path: str, version: str, index_path: str = None, log_level: str = "INFO"
):
    """Run the import command with proper arguments."""

    cmd = [
        sys.executable,
        "-m",
        "src.ifs_cloud_mcp_server.main",
        "import",
        zip_path,
        "--version",
        version,
        "--log-level",
        log_level,
    ]

    if index_path:
        cmd.extend(["--index-path", index_path])

    print(f"🚀 Running: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Import failed with exit code: {e.returncode}")
        return False


def list_versions():
    """List all available versions."""
    cmd = [sys.executable, "-m", "src.ifs_cloud_mcp_server.main", "list"]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to list versions: {e}")


def start_server(version: str = None, index_path: str = None):
    """Start the server with specified version or index path."""
    cmd = [sys.executable, "-m", "src.ifs_cloud_mcp_server.main", "server"]

    if version:
        cmd.extend(["--version", version])
    elif index_path:
        cmd.extend(["--index-path", index_path])

    print(f"🚀 Starting server: {' '.join(cmd)}")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n✅ Server stopped")


def main():
    """Main entry point for the helper script."""
    parser = argparse.ArgumentParser(
        description="IFS Cloud ZIP Import Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import a ZIP file
  python zip_import_helper.py import my_ifs_cloud.zip --version "24.2.1"
  
  # Import with debug logging
  python zip_import_helper.py import my_ifs_cloud.zip --version "24.2.1" --debug
  
  # List available versions
  python zip_import_helper.py list
  
  # Start server with specific version
  python zip_import_helper.py server --version "24.2.1"
  
  # Quick import and start workflow
  python zip_import_helper.py quick my_ifs_cloud.zip "24.2.1"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import IFS Cloud ZIP file")
    import_parser.add_argument("zip_file", help="Path to IFS Cloud ZIP file")
    import_parser.add_argument("--version", required=True, help="Version identifier")
    import_parser.add_argument("--index-path", help="Custom index path")
    import_parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List available versions")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start MCP server")
    server_group = server_parser.add_mutually_exclusive_group()
    server_group.add_argument("--version", help="Version to use")
    server_group.add_argument("--index-path", help="Index path to use")

    # Quick command (import + start)
    quick_parser = subparsers.add_parser("quick", help="Quick import and start server")
    quick_parser.add_argument("zip_file", help="Path to IFS Cloud ZIP file")
    quick_parser.add_argument("version", help="Version identifier")
    quick_parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "import":
        log_level = "DEBUG" if args.debug else "INFO"
        success = run_import_command(
            args.zip_file, args.version, args.index_path, log_level
        )
        return 0 if success else 1

    elif args.command == "list":
        list_versions()
        return 0

    elif args.command == "server":
        start_server(args.version, args.index_path)
        return 0

    elif args.command == "quick":
        print("🔥 Quick Import & Start Workflow")
        print("=" * 40)

        # Step 1: Import
        print("📦 Step 1: Importing ZIP file...")
        log_level = "DEBUG" if args.debug else "INFO"
        success = run_import_command(args.zip_file, args.version, None, log_level)

        if not success:
            print("❌ Import failed, cannot start server")
            return 1

        print("\n✅ Import successful!")
        print("\n🚀 Step 2: Starting server...")

        # Step 2: Start server
        start_server(args.version)
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
