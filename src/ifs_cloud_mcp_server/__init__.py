"""IFS Cloud MCP Server with Tantivy search engine integration.

A high-performance Model Context Protocol server for IFS Cloud codebases,
providing enterprise-grade search capabilities.
"""

__version__ = "0.1.0"
__author__ = "IFS Cloud Team"

# Optional imports - only import if dependencies are available
__all__ = []

try:
    from .server import IFSCloudMCPServer

    __all__.append("IFSCloudMCPServer")
except ImportError:
    pass

try:
    from .indexer import IFSCloudIndexer

    __all__.append("IFSCloudIndexer")
except ImportError:
    pass

# Always make parsers available as they have no external dependencies
from .parsers import IFSFileParser, ParsedFile

__all__.extend(["IFSFileParser", "ParsedFile"])

try:
    from .config import ConfigManager

    __all__.append("ConfigManager")
except ImportError:
    pass
