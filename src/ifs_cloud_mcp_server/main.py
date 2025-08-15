"""Main entry point for IFS Cloud MCP Server."""

import asyncio
import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Set

from .config import ConfigManager
from .server_fastmcp import IFSCloudMCPServer
from .indexer import IFSCloudTantivyIndexer


def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
        ],
    )


def get_data_directory() -> Path:
    """Get the platform-appropriate data directory for IFS Cloud files."""
    app_name = "ifs_cloud_mcp_server"

    if sys.platform == "win32":
        # Windows: %APPDATA%/ifs_cloud_mcp_server
        base_path = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        # macOS: ~/Library/Application Support/ifs_cloud_mcp_server
        base_path = Path.home() / "Library" / "Application Support"
    else:
        # Linux/Unix: ~/.local/share/ifs_cloud_mcp_server
        base_path = Path(
            os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
        )

    return base_path / app_name


def get_supported_extensions() -> Set[str]:
    """Get the set of file extensions that IFS Cloud MCP Server supports."""
    return {
        ".entity",
        ".plsql",
        ".views",
        ".storage",
        ".fragment",
        ".client",
        ".projection",
        ".plsvc",
    }


def resolve_version_to_index_path(version: str) -> Path:
    """Resolve a version name to its corresponding index path.

    Args:
        version: Version identifier

    Returns:
        Path to the index directory for this version

    Raises:
        ValueError: If version doesn't exist or has no index
    """
    data_dir = get_data_directory()
    safe_version = "".join(c for c in version if c.isalnum() or c in "._-")

    extract_path = data_dir / "extracts" / safe_version
    index_path = data_dir / "indexes" / safe_version

    if not extract_path.exists():
        raise ValueError(
            f"Version '{version}' not found. Available versions can be listed with: python -m src.ifs_cloud_mcp_server.main list"
        )

    if not index_path.exists():
        raise ValueError(
            f"Version '{version}' found but not indexed. Please re-import with: python -m src.ifs_cloud_mcp_server.main import <zip_file> --version {version}"
        )

    return index_path


def extract_ifs_cloud_zip(zip_path: Path, version: str) -> Path:
    """Extract IFS Cloud ZIP file to versioned directory with only supported files.

    Args:
        zip_path: Path to the ZIP file
        version: Version identifier for this extract

    Returns:
        Path to the extracted directory

    Raises:
        FileNotFoundError: If ZIP file doesn't exist
        zipfile.BadZipFile: If ZIP file is corrupted
        ValueError: If version contains invalid characters
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    # Sanitize version name for filesystem
    safe_version = "".join(c for c in version if c.isalnum() or c in "._-")
    if not safe_version:
        raise ValueError("Version must contain at least one alphanumeric character")

    # Get extraction directory
    data_dir = get_data_directory()
    extract_dir = data_dir / "extracts" / safe_version

    # Remove existing extraction if it exists
    if extract_dir.exists():
        import shutil

        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)

    supported_extensions = get_supported_extensions()
    extracted_count = 0

    logging.info(f"Extracting IFS Cloud files from {zip_path} to {extract_dir}")
    logging.info(f"Supported file types: {', '.join(sorted(supported_extensions))}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            # Skip directories
            if file_info.is_dir():
                continue

            file_path = Path(file_info.filename)

            # Check if file has supported extension
            if file_path.suffix.lower() in supported_extensions:
                try:
                    # Extract file maintaining directory structure
                    zip_ref.extract(file_info, extract_dir)
                    extracted_count += 1

                    if extracted_count % 100 == 0:
                        logging.info(f"Extracted {extracted_count} files...")

                except Exception as e:
                    logging.warning(f"Failed to extract {file_info.filename}: {e}")
                    continue

    logging.info(
        f"Successfully extracted {extracted_count} supported files to {extract_dir}"
    )
    return extract_dir


async def build_index_for_extract(extract_path: Path, index_path: Path) -> bool:
    """Build search index for extracted IFS Cloud files.

    Args:
        extract_path: Path to extracted files
        index_path: Path where index should be stored

    Returns:
        True if indexing was successful
    """
    try:
        logging.info(
            f"Building search index at {index_path} for files in {extract_path}"
        )

        # Create indexer
        indexer = IFSCloudTantivyIndexer(index_path=index_path)

        # Build index
        stats = await indexer.index_directory(str(extract_path))

        logging.info(f"Index built successfully:")
        logging.info(f"  Files indexed: {stats.get('indexed', 0)}")
        logging.info(f"  Files cached: {stats.get('cached', 0)}")
        logging.info(f"  Files skipped: {stats.get('skipped', 0)}")
        logging.info(f"  Errors: {stats.get('errors', 0)}")

        return True

    except Exception as e:
        logging.error(f"Failed to build index: {e}")
        return False


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="IFS Cloud MCP Server with Tantivy search"
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server command (default)
    server_parser = subparsers.add_parser("server", help="Start MCP server")

    # Create mutually exclusive group for index-path vs version
    index_group = server_parser.add_mutually_exclusive_group()
    index_group.add_argument(
        "--index-path",
        type=str,
        help="Path to store the Tantivy index (default: ./index if no version specified)",
    )
    index_group.add_argument(
        "--version",
        type=str,
        help="IFS Cloud version to use (automatically resolves index path)",
    )

    server_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    server_parser.add_argument(
        "--name",
        type=str,
        default="ifs-cloud-mcp-server",
        help="Server name (default: ifs-cloud-mcp-server)",
    )
    server_parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio"],
        help="Transport type (default: stdio)",
    )

    # Import command
    import_parser = subparsers.add_parser("import", help="Import IFS Cloud ZIP file")
    import_parser.add_argument("zip_file", type=str, help="Path to IFS Cloud ZIP file")
    import_parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version identifier for this IFS Cloud release (e.g., '24.2.1', 'latest')",
    )
    import_parser.add_argument(
        "--index-path",
        type=str,
        help="Custom path for index (default: auto-generated based on version)",
    )
    import_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    # List command
    list_parser = subparsers.add_parser(
        "list", help="List available IFS Cloud versions"
    )
    list_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format for programmatic use"
    )

    # If no command specified, default to server mode for backward compatibility
    if len(sys.argv) == 1 or not any(
        sys.argv[1] == cmd for cmd in ["server", "import", "list"]
    ):
        # Add server args directly to main parser for backward compatibility
        index_group = parser.add_mutually_exclusive_group()
        index_group.add_argument(
            "--index-path",
            type=str,
            help="Path to store the Tantivy index (default: ./index if no version specified)",
        )
        index_group.add_argument(
            "--version",
            type=str,
            help="IFS Cloud version to use (automatically resolves index path)",
        )

        parser.add_argument(
            "--log-level",
            type=str,
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Log level (default: INFO)",
        )
        parser.add_argument(
            "--name",
            type=str,
            default="ifs-cloud-mcp-server",
            help="Server name (default: ifs-cloud-mcp-server)",
        )
        parser.add_argument(
            "--transport",
            type=str,
            default="stdio",
            choices=["stdio"],
            help="Transport type (default: stdio)",
        )

    args = parser.parse_args()

    # Set up logging for non-list commands
    if getattr(args, "command", None) != "list":
        setup_logging(args.log_level)

    # Handle commands
    if getattr(args, "command", None) == "import":
        return await handle_import_command(args)
    elif getattr(args, "command", None) == "list":
        return handle_list_command(args)

    # Handle server command (default)
    return await handle_server_command(args)


async def handle_import_command(args) -> int:
    """Handle the import command."""
    try:
        zip_path = Path(args.zip_file)
        version = args.version

        # Extract ZIP file
        extract_path = extract_ifs_cloud_zip(zip_path, version)

        # Determine index path
        if args.index_path:
            index_path = Path(args.index_path)
        else:
            data_dir = get_data_directory()
            safe_version = "".join(c for c in version if c.isalnum() or c in "._-")
            index_path = data_dir / "indexes" / safe_version

        # Build index
        index_path.mkdir(parents=True, exist_ok=True)
        success = await build_index_for_extract(extract_path, index_path)

        if success:
            logging.info(f"✅ Import completed successfully!")
            logging.info(f"📁 Extracted files: {extract_path}")
            logging.info(f"🔍 Search index: {index_path}")
            logging.info(f"🏷️  Version: {version}")
            logging.info("")
            logging.info("To use this version with the MCP server:")
            logging.info(
                f'  python -m src.ifs_cloud_mcp_server.main server --version "{version}"'
            )
            return 0
        else:
            logging.error("❌ Import failed during indexing")
            return 1

    except Exception as e:
        logging.error(f"❌ Import failed: {e}")
        return 1


def handle_list_command(args) -> int:
    """Handle the list command."""
    import json
    from datetime import datetime

    try:
        data_dir = get_data_directory()
        extracts_dir = data_dir / "extracts"
        indexes_dir = data_dir / "indexes"

        versions = []

        # Scan for available versions
        if extracts_dir.exists():
            for version_dir in extracts_dir.iterdir():
                if version_dir.is_dir():
                    index_path = indexes_dir / version_dir.name

                    # Get file count
                    file_count = 0
                    if version_dir.exists():
                        for ext in get_supported_extensions():
                            file_count += len(list(version_dir.rglob(f"*{ext}")))

                    # Get creation time
                    try:
                        created = datetime.fromtimestamp(version_dir.stat().st_ctime)
                        created_str = created.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        created_str = "Unknown"

                    version_info = {
                        "version": version_dir.name,
                        "extract_path": str(version_dir),
                        "index_path": str(index_path),
                        "has_index": index_path.exists(),
                        "file_count": file_count,
                        "created": created_str,
                    }
                    versions.append(version_info)

        # Sort by creation time (newest first)
        versions.sort(key=lambda x: x["created"], reverse=True)

        if args.json:
            # Output JSON for programmatic use
            print(json.dumps(versions, indent=2))
        else:
            # Human-readable output
            if not versions:
                print("No IFS Cloud versions found.")
                print("")
                print("To import a version:")
                print(
                    "  python -m src.ifs_cloud_mcp_server.main import <zip_file> --version <version_name>"
                )
            else:
                print("Available IFS Cloud versions:")
                print("")
                for v in versions:
                    status = "✅ Indexed" if v["has_index"] else "⚠️  Not indexed"
                    print(f"📦 {v['version']}")
                    print(f"   Status: {status}")
                    print(f"   Files: {v['file_count']:,}")
                    print(f"   Created: {v['created']}")
                    if v["has_index"]:
                        print(
                            f"   Command: python -m src.ifs_cloud_mcp_server.main server --version \"{v['version']}\""
                        )
                    print("")

        return 0

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"❌ Failed to list versions: {e}")
        return 1


async def handle_server_command(args) -> int:
    """Handle the server command."""
    try:
        # Determine index path
        if hasattr(args, "version") and args.version:
            # Resolve version to index path
            index_path = resolve_version_to_index_path(args.version)
            logging.info(f"Using IFS Cloud version: {args.version}")
            logging.info(f"Index path: {index_path}")
        elif hasattr(args, "index_path") and args.index_path:
            # Use provided index path
            index_path = Path(args.index_path)
        else:
            # Default to ./index
            index_path = Path("./index")

        # Create index directory if it doesn't exist
        index_path.mkdir(parents=True, exist_ok=True)

        # Create and run server
        server = IFSCloudMCPServer(
            index_path=index_path, name=getattr(args, "name", "ifs-cloud-mcp-server")
        )

        server.run(transport_type=getattr(args, "transport", "stdio"))

    except ValueError as e:
        logging.error(f"❌ Version error: {e}")
        return 1
    except KeyboardInterrupt:
        logging.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logging.error(f"Server error: {e}")
        return 1
    finally:
        if "server" in locals():
            await server.cleanup()

    return 0


def main_sync():
    """Synchronous main entry point for console scripts."""
    # Use asyncio.run for the main async function
    return asyncio.run(main())


if __name__ == "__main__":
    sys.exit(main_sync())
