#!/usr/bin/env python3
"""
Startup script for IFS Cloud MCP Server - supports both MCP server and Web UI modes.
"""

import argparse
import asyncio
import sys
from pathlib import Path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="IFS Cloud MCP Server - Intelligent search for IFS Cloud codebases"
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")

    # MCP Server mode
    mcp_parser = subparsers.add_parser(
        "mcp", help="Run as MCP server (for Claude/Copilot)"
    )
    mcp_parser.add_argument(
        "--index-path", default="./index", help="Path to store the Tantivy index"
    )
    mcp_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level",
    )
    mcp_parser.add_argument(
        "--name", default="ifs-cloud-mcp-server", help="Server name"
    )
    mcp_parser.add_argument(
        "--transport", choices=["stdio"], default="stdio", help="Transport type"
    )

    # Web UI mode
    web_parser = subparsers.add_parser("web", help="Run web UI server")
    web_parser.add_argument(
        "--index-path", default="./index", help="Path to store the Tantivy index"
    )
    web_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    web_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    web_parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    # Index management
    index_parser = subparsers.add_parser("index", help="Index management")
    index_parser.add_argument(
        "action", choices=["build", "rebuild", "stats"], help="Index action"
    )
    index_parser.add_argument(
        "--directory",
        required=False,
        help="Directory to index (required for build/rebuild)",
    )
    index_parser.add_argument(
        "--index-path", default="./index", help="Path to store the Tantivy index"
    )

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return

    if args.mode == "mcp":
        run_mcp_server(args)
    elif args.mode == "web":
        run_web_ui(args)
    elif args.mode == "index":
        asyncio.run(manage_index(args))


def run_mcp_server(args):
    """Run the MCP server."""
    from src.ifs_cloud_mcp_server.server_fastmcp import main as mcp_main

    print("🔌 Starting IFS Cloud MCP Server...")
    print(f"📊 Index path: {args.index_path}")
    print(f"📝 Log level: {args.log_level}")
    print(
        "💡 This server provides intelligent IFS Cloud search capabilities to MCP clients"
    )

    # Set up arguments for MCP server
    sys.argv = [
        "server_fastmcp.py",
        "--index-path",
        args.index_path,
        "--log-level",
        args.log_level,
        "--name",
        args.name,
        "--transport",
        args.transport,
    ]

    mcp_main()


def run_web_ui(args):
    """Run the web UI server."""
    import uvicorn
    from src.ifs_cloud_mcp_server.web_ui import IFSCloudWebUI, create_web_ui_files

    print("🌐 Starting IFS Cloud Web UI...")
    print(f"📊 Index path: {args.index_path}")
    print(f"🔗 Interface: http://{args.host}:{args.port}")
    print("🔍 Features:")
    print("  • Type-ahead search with intelligent suggestions")
    print("  • Frontend element discovery (pages, iconsets, trees, navigators)")
    print("  • Module-aware search and filtering")
    print("  • Real-time search with highlighting")
    print("  • Responsive design with modern UI")

    # Create web UI files
    create_web_ui_files()

    # Create the web UI application
    web_ui = IFSCloudWebUI(args.index_path)

    # Start the server
    uvicorn.run(web_ui.app, host=args.host, port=args.port, reload=args.reload)


async def manage_index(args):
    """Manage the search index."""
    from src.ifs_cloud_mcp_server.indexer import IFSCloudTantivyIndexer

    indexer = IFSCloudTantivyIndexer(args.index_path)

    if args.action == "stats":
        print("📊 Index Statistics:")
        stats = indexer.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    elif args.action in ["build", "rebuild"]:
        if not args.directory:
            print("❌ Error: --directory is required for build/rebuild actions")
            return

        directory = Path(args.directory)
        if not directory.exists():
            print(f"❌ Error: Directory {directory} does not exist")
            return

        print(f"🔄 {'Rebuilding' if args.action == 'rebuild' else 'Building'} index...")
        print(f"📁 Source directory: {directory}")
        print(f"📊 Index path: {args.index_path}")

        await indexer.index_directory(directory)

        print("✅ Index operation completed!")

        # Show stats
        stats = indexer.get_stats()
        print(
            f"📊 Indexed {stats.get('total_files', 0)} files with {stats.get('total_entities', 0)} entities"
        )


if __name__ == "__main__":
    main()
