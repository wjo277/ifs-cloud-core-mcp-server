"""IFS Cloud MCP Server with Tantivy search engine integration.

A high-performance Model Context Protocol server for IFS Cloud codebases,
providing enterprise-grade search capabilities.
"""

__version__ = "0.1.0"
__author__ = "IFS Cloud Team"

from .server import IFSCloudMCPServer
from .indexer import IFSCloudTantivyIndexer

__all__ = ["IFSCloudMCPServer", "IFSCloudTantivyIndexer"]