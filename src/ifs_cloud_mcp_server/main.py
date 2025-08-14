"""Main entry point for IFS Cloud MCP Server."""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

from .server import IFSCloudMCPServer


def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
        ]
    )


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="IFS Cloud MCP Server with Tantivy search")
    parser.add_argument(
        "--index-path",
        type=str,
        default="./index",
        help="Path to store the Tantivy index (default: ./index)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="ifs-cloud-mcp-server",
        help="Server name (default: ifs-cloud-mcp-server)"
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio"],
        help="Transport type (default: stdio)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Create index directory if it doesn't exist
    index_path = Path(args.index_path)
    index_path.mkdir(parents=True, exist_ok=True)
    
    # Create and run server
    server = IFSCloudMCPServer(index_path=index_path, name=args.name)
    
    try:
        await server.run(transport_type=args.transport)
    except KeyboardInterrupt:
        logging.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logging.error(f"Server error: {e}")
        return 1
    finally:
        await server.cleanup()
    
    return 0


def main_sync():
    """Synchronous main entry point for console scripts."""
    return asyncio.run(main())


if __name__ == "__main__":
    sys.exit(main_sync())