"""Main entry point for IFS Cloud MCP Server."""

import asyncio
import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Set, Optional, Dict, Any
from datetime import datetime

from .config import ConfigManager
from .server_fastmcp import IFSCloudMCPServer
from .indexer import IFSCloudIndexer


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
        indexer = IFSCloudIndexer(index_path=index_path)

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


def build_connection_string_from_args(args) -> str:
    """Build Oracle connection string from command line arguments."""

    # If direct connection string provided, use it
    if hasattr(args, "connection") and args.connection:
        return args.connection

    # Build from individual components
    driver = getattr(args, "driver", "oracle+oracledb")
    host = args.host
    port = getattr(args, "port", 1521)
    username = args.username

    # Get password from argument or environment
    password = getattr(args, "password", None) or os.getenv("IFS_DB_PASSWORD")
    if not password:
        raise ValueError(
            "Password must be provided via --password or IFS_DB_PASSWORD environment variable"
        )

    # Service name or SID
    if hasattr(args, "service") and args.service:
        return f"{driver}://{username}:{password}@{host}:{port}/?service_name={args.service}"
    elif hasattr(args, "sid") and args.sid:
        return f"{driver}://{username}:{password}@{host}:{port}/{args.sid}"
    else:
        raise ValueError("Must specify either --service or --sid")


async def handle_extract_command(args) -> int:
    """Handle the database extraction command."""
    try:
        # Validate dependencies first
        try:
            from .metadata_extractor import DatabaseMetadataExtractor, MetadataManager
            from sqlalchemy import create_engine, text
            import oracledb
        except ImportError as e:
            logging.error(f"✗ Missing database dependencies: {e}")
            logging.error("Install with: uv add sqlalchemy oracledb")
            return 1

        # Build connection string
        try:
            connection_string = build_connection_string_from_args(args)
            # Mask password in logs
            log_connection = connection_string.split("@")[0].split(":")
            if len(log_connection) >= 3:
                log_connection[2] = "***"
            logging.debug(f"Connection string: {':'.join(log_connection)}@***")
        except Exception as e:
            logging.error(f"✗ Failed to build connection string: {e}")
            return 1

        # Test database connection
        logging.info(
            f"🔌 Connecting to IFS Cloud database ({args.host or 'provided connection'})..."
        )
        start_time = datetime.now()

        try:
            engine = create_engine(connection_string, echo=(args.log_level == "DEBUG"))

            with engine.connect() as conn:
                result = conn.execute(text("SELECT 'Connected' as status FROM dual"))
                status = result.fetchone()[0]
                logging.info(f"✓ Database connection successful: {status}")

        except Exception as e:
            logging.error(f"✗ Database connection failed: {e}")
            return 1

        # Initialize metadata extractor
        version = args.version
        logging.info(f"🚀 Starting metadata extraction for IFS Cloud {version}...")

        try:
            extractor = DatabaseMetadataExtractor(db_connection=engine)
            metadata_extract = extractor.extract_from_database(version)

        except Exception as e:
            logging.error(f"✗ Metadata extraction failed: {e}")
            return 1

        # Setup output directory - use same structure as other commands
        if hasattr(args, "output") and args.output:
            output_dir = Path(args.output)
        else:
            # Use same data directory structure as import/server commands
            data_dir = get_data_directory()
            output_dir = data_dir / "metadata"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        try:
            manager = MetadataManager(output_dir)
            saved_path = manager.save_metadata(metadata_extract)

        except Exception as e:
            logging.error(f"✗ Failed to save metadata: {e}")
            return 1

        # Calculate extraction time
        end_time = datetime.now()
        extraction_time = (end_time - start_time).total_seconds()

        # Prepare results
        results = {
            "success": True,
            "ifs_version": metadata_extract.ifs_version,
            "extraction_date": metadata_extract.extraction_date,
            "extraction_time_seconds": extraction_time,
            "output_path": str(saved_path),
            "checksum": metadata_extract.checksum,
            "statistics": {
                "logical_units": len(metadata_extract.logical_units),
                "modules": len(metadata_extract.modules),
                "domain_mappings": len(metadata_extract.domain_mappings),
                "views": len(metadata_extract.views),
                "navigator_entries": len(metadata_extract.navigator_entries),
            },
        }

        # Output results
        if hasattr(args, "json") and args.json:
            import json

            print(json.dumps(results, indent=2, default=str))
        elif hasattr(args, "quiet") and args.quiet:
            pass  # Quiet mode - minimal output
        else:
            # Display user-friendly results
            logging.info("=" * 80)
            logging.info("🎉 METADATA EXTRACTION COMPLETE")
            logging.info("=" * 80)
            logging.info(f"📦 IFS Version: {results['ifs_version']}")
            logging.info(f"📅 Extraction Date: {results['extraction_date']}")
            logging.info(
                f"⏱️  Extraction Time: {results['extraction_time_seconds']:.1f} seconds"
            )
            logging.info(f"💾 Saved to: {results['output_path']}")
            logging.info(f"🔍 Checksum: {results['checksum']}")
            logging.info("")
            logging.info("📊 Statistics:")
            stats = results["statistics"]
            logging.info(f"   • Logical Units: {stats['logical_units']:,}")
            logging.info(f"   • Modules: {stats['modules']:,}")
            logging.info(f"   • Domain Mappings: {stats['domain_mappings']:,}")
            logging.info(f"   • Views: {stats['views']:,}")
            logging.info(f"   • Navigator Entries: {stats['navigator_entries']:,}")

            # Show top modules by LU count
            if metadata_extract.modules:
                logging.info("")
                logging.info("🏆 Top 10 modules by Logical Unit count:")
                for i, module in enumerate(metadata_extract.modules[:10], 1):
                    logging.info(
                        f"   {i:2d}. {module.name:<20} - {module.lu_count:>4} LUs"
                    )

            # Show sample navigator entries
            if metadata_extract.navigator_entries:
                logging.info("")
                logging.info("🧭 Sample Navigator Entries (GUI to Backend mapping):")
                for i, nav in enumerate(metadata_extract.navigator_entries[:5], 1):
                    logging.info(
                        f"   {i:2d}. '{nav.label}' → {nav.entity_name} ({nav.projection})"
                    )

            logging.info("=" * 80)
            logging.info(
                "✅ Ready for enhanced search! Use this metadata with your IFS Cloud MCP server."
            )

        return 0

    except KeyboardInterrupt:
        logging.info("❌ Extraction cancelled by user")
        return 130
    except Exception as e:
        logging.error(f"✗ Unexpected error during extraction: {e}")
        if hasattr(args, "log_level") and args.log_level == "DEBUG":
            import traceback

            logging.error(traceback.format_exc())
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


def handle_server_command(args) -> int:
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

        # Create server
        server = IFSCloudMCPServer(
            index_path=index_path, name=getattr(args, "name", "ifs-cloud-mcp-server")
        )

        # Try to run the server, handling asyncio context issues
        server.run(transport_type=getattr(args, "transport", "stdio"))

    except ValueError as e:
        logging.error(f"❌ Version error: {e}")
        return 1
    except KeyboardInterrupt:
        logging.info("Received interrupt signal, shutting down...")
    except RuntimeError as e:
        if "Already running asyncio in this thread" in str(e):
            logging.error("❌ AsyncIO conflict detected")
            logging.error("This server must be run as a standalone process")
            logging.error("Please ensure no other asyncio event loop is running")
            return 1
        else:
            logging.error(f"Runtime error: {e}")
            return 1
    except Exception as e:
        logging.error(f"Server error: {e}")
        return 1
    finally:
        if "server" in locals() and server is not None:
            server.cleanup()

    return 0


def main_sync():
    """Synchronous main entry point for console scripts."""
    import argparse

    # Check if we're being called in an asyncio context
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        # If we get here, we're in an asyncio context
        print(
            "❌ Error: IFS Cloud MCP Server cannot be run from within an asyncio context.",
            file=sys.stderr,
        )
        print("", file=sys.stderr)
        print("This typically happens when:", file=sys.stderr)
        print("  1. Running from a Jupyter notebook or IPython", file=sys.stderr)
        print("  2. Being called from within an async function", file=sys.stderr)
        print("  3. Called by an MCP client that uses asyncio", file=sys.stderr)
        print("", file=sys.stderr)
        print("Solutions:", file=sys.stderr)
        print("  1. Run from a regular command line terminal", file=sys.stderr)
        print("  2. Use the standalone wrapper script", file=sys.stderr)
        print(
            "  3. Ensure your MCP client runs the server as a subprocess",
            file=sys.stderr,
        )
        return 1
    except RuntimeError:
        # No event loop, we're good to proceed
        pass

    # Parse arguments first to determine which command to run
    parser = argparse.ArgumentParser(
        description="IFS Cloud MCP Server with Tantivy search and production database metadata extraction"
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Import command (requires async)
    import_parser = subparsers.add_parser(
        "import", help="Import IFS Cloud ZIP file and create search index"
    )
    import_parser.add_argument("zip_file", help="Path to IFS Cloud ZIP file")
    import_parser.add_argument("version", help="IFS Cloud version (e.g., 25.1.0)")
    import_parser.add_argument(
        "--index-path", help="Custom path for search index (optional)"
    )
    import_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    # Extract command (requires async) - Extract metadata from database
    extract_parser = subparsers.add_parser(
        "extract", help="Extract metadata from IFS Cloud database for enhanced search"
    )

    # Connection options
    extract_conn_group = extract_parser.add_argument_group("Connection Options")
    extract_conn_group.add_argument(
        "--connection",
        "-c",
        help="Complete Oracle connection string (oracle://user:pass@host:port/service)",
    )
    extract_conn_group.add_argument("--host", help="Database host")
    extract_conn_group.add_argument(
        "--port", type=int, default=1521, help="Database port (default: 1521)"
    )
    extract_conn_group.add_argument("--username", "-u", help="Database username")
    extract_conn_group.add_argument(
        "--password", "-p", help="Database password (or use IFS_DB_PASSWORD env var)"
    )
    extract_conn_group.add_argument("--service", help="Oracle service name")
    extract_conn_group.add_argument(
        "--sid", help="Oracle SID (alternative to service name)"
    )
    extract_conn_group.add_argument(
        "--driver",
        default="oracle+oracledb",
        help="SQLAlchemy driver (default: oracle+oracledb)",
    )

    # Extraction options
    extract_parser.add_argument(
        "version", help="IFS Cloud version (e.g., '25.1.0', '24.2.1')"
    )
    extract_parser.add_argument(
        "--output",
        "-o",
        help="Output directory for metadata files (default: platform data directory)",
    )

    # Output options
    extract_parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress non-error output"
    )
    extract_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )
    extract_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    # List command (synchronous)
    list_parser = subparsers.add_parser(
        "list", help="List available IFS Cloud versions and their index status"
    )
    list_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )

    # Server command (synchronous - default)
    server_parser = subparsers.add_parser(
        "server", help="Start the MCP server (default command)"
    )
    server_parser.add_argument(
        "--version", help="IFS Cloud version to use (e.g., 25.1.0)"
    )
    server_parser.add_argument("--index-path", help="Path to search index")
    server_parser.add_argument(
        "--name", default="ifs-cloud-mcp-server", help="Server name"
    )
    server_parser.add_argument(
        "--transport", default="stdio", help="Transport type (stdio, sse)"
    )
    server_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()

    # Route to appropriate handler based on command
    if getattr(args, "command", None) == "import":
        # Import command requires async
        setup_logging(args.log_level)
        return asyncio.run(handle_import_command(args))
    elif getattr(args, "command", None) == "extract":
        # Extract command requires async
        setup_logging(args.log_level)
        return asyncio.run(handle_extract_command(args))
    elif getattr(args, "command", None) == "list":
        # List command is synchronous
        return handle_list_command(args)
    else:
        # Server command (default) is synchronous and manages its own event loop
        setup_logging(getattr(args, "log_level", "INFO"))
        return handle_server_command(args)


if __name__ == "__main__":
    sys.exit(main_sync())
