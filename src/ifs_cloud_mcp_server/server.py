"""IFS Cloud MCP Server implementation."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mcp.server import Server
from mcp.types import Tool, TextContent, Resource, ResourceTemplate
from pydantic import BaseModel

from .indexer import IFSCloudTantivyIndexer, SearchResult


logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    """Search request model."""
    query: str
    limit: int = 10
    file_type: Optional[str] = None
    min_complexity: Optional[float] = None
    max_complexity: Optional[float] = None


class IndexRequest(BaseModel):
    """Index request model."""
    path: str
    recursive: bool = True


class IFSCloudMCPServer:
    """MCP Server for IFS Cloud with Tantivy search integration."""
    
    def __init__(self, index_path: Union[str, Path], name: str = "ifs-cloud-mcp-server"):
        """Initialize the MCP server.
        
        Args:
            index_path: Path to store the Tantivy index
            name: Server name
        """
        self.name = name
        self.server = Server(name)
        self.indexer = IFSCloudTantivyIndexer(index_path)
        
        # Register MCP tools
        self._register_tools()
        self._register_resources()
        
        logger.info(f"Initialized IFS Cloud MCP Server: {name}")
    
    def _register_tools(self):
        """Register MCP tools for search and indexing operations."""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="search_content",
                    description="Search IFS Cloud files by content with advanced filtering",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (supports full-text search)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10)",
                                "default": 10
                            },
                            "file_type": {
                                "type": "string",
                                "description": "Filter by file type (.entity, .plsql, .views, etc.)",
                                "enum": [".entity", ".plsql", ".views", ".storage", ".fragment", ".client", ".projection"]
                            },
                            "min_complexity": {
                                "type": "number",
                                "description": "Minimum complexity score (0.0-1.0)"
                            },
                            "max_complexity": {
                                "type": "number",
                                "description": "Maximum complexity score (0.0-1.0)"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="search_entities",
                    description="Search for files containing specific IFS entities",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {
                                "type": "string",
                                "description": "Entity name to search for"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10)",
                                "default": 10
                            }
                        },
                        "required": ["entity"]
                    }
                ),
                Tool(
                    name="find_similar_files",
                    description="Find files similar to a given file based on entities and content",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the reference file"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of similar files (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="search_by_complexity",
                    description="Search files by complexity score range",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "min_complexity": {
                                "type": "number",
                                "description": "Minimum complexity score (0.0-1.0)"
                            },
                            "max_complexity": {
                                "type": "number",
                                "description": "Maximum complexity score (0.0-1.0)"
                            },
                            "file_type": {
                                "type": "string",
                                "description": "Filter by file type",
                                "enum": [".entity", ".plsql", ".views", ".storage", ".fragment", ".client", ".projection"]
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10)",
                                "default": 10
                            }
                        }
                    }
                ),
                Tool(
                    name="index_directory",
                    description="Index all IFS Cloud files in a directory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path to index"
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "Index subdirectories recursively (default: true)",
                                "default": True
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="index_file",
                    description="Index a single IFS Cloud file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to index"
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="get_index_statistics",
                    description="Get statistics about the search index",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="fuzzy_search",
                    description="Perform fuzzy search to handle typos and partial matches",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (fuzzy matching enabled)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10)",
                                "default": 10
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "search_content":
                    return await self._handle_search_content(arguments)
                elif name == "search_entities":
                    return await self._handle_search_entities(arguments)
                elif name == "find_similar_files":
                    return await self._handle_find_similar_files(arguments)
                elif name == "search_by_complexity":
                    return await self._handle_search_by_complexity(arguments)
                elif name == "index_directory":
                    return await self._handle_index_directory(arguments)
                elif name == "index_file":
                    return await self._handle_index_file(arguments)
                elif name == "get_index_statistics":
                    return await self._handle_get_index_statistics(arguments)
                elif name == "fuzzy_search":
                    return await self._handle_fuzzy_search(arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
                    
            except Exception as e:
                logger.error(f"Error in tool '{name}': {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    def _register_resources(self):
        """Register MCP resources."""
        
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources."""
            return [
                Resource(
                    uri="ifs-cloud://index/statistics",
                    name="Index Statistics",
                    description="Current search index statistics",
                    mimeType="application/json"
                ),
                Resource(
                    uri="ifs-cloud://supported/file-types",
                    name="Supported File Types",
                    description="List of supported IFS Cloud file types",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read resource content."""
            if uri == "ifs-cloud://index/statistics":
                stats = self.indexer.get_statistics()
                return json.dumps(stats, indent=2)
            elif uri == "ifs-cloud://supported/file-types":
                return json.dumps({
                    "supported_extensions": list(self.indexer.SUPPORTED_EXTENSIONS),
                    "descriptions": {
                        ".entity": "Entity definitions",
                        ".plsql": "PL/SQL code",
                        ".views": "Database views",
                        ".storage": "Storage configurations",
                        ".fragment": "Code fragments",
                        ".client": "Client-side code",
                        ".projection": "Data projections"
                    }
                }, indent=2)
            else:
                raise ValueError(f"Unknown resource URI: {uri}")
    
    async def _handle_search_content(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle content search tool call."""
        query = arguments["query"]
        limit = arguments.get("limit", 10)
        file_type = arguments.get("file_type")
        min_complexity = arguments.get("min_complexity")
        max_complexity = arguments.get("max_complexity")
        
        results = self.indexer.search(
            query=query,
            limit=limit,
            file_type=file_type,
            min_complexity=min_complexity,
            max_complexity=max_complexity
        )
        
        if not results:
            return [TextContent(type="text", text=f"No results found for query: '{query}'")]
        
        response = self._format_search_results(results, f"Content search results for '{query}'")
        return [TextContent(type="text", text=response)]
    
    async def _handle_search_entities(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle entity search tool call."""
        entity = arguments["entity"]
        limit = arguments.get("limit", 10)
        
        # Search in entities field specifically
        query = f"entities:{entity}"
        results = self.indexer.search(query=query, limit=limit)
        
        if not results:
            return [TextContent(type="text", text=f"No files found containing entity: '{entity}'")]
        
        response = self._format_search_results(results, f"Files containing entity '{entity}'")
        return [TextContent(type="text", text=response)]
    
    async def _handle_find_similar_files(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle find similar files tool call."""
        file_path = arguments["file_path"]
        limit = arguments.get("limit", 5)
        
        results = self.indexer.find_similar_files(file_path, limit)
        
        if not results:
            return [TextContent(type="text", text=f"No similar files found for: {file_path}")]
        
        # Filter out the original file if it appears in results
        filtered_results = [r for r in results if r.path != str(file_path)][:limit]
        
        response = self._format_search_results(
            filtered_results, 
            f"Files similar to '{file_path}'"
        )
        return [TextContent(type="text", text=response)]
    
    async def _handle_search_by_complexity(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle complexity search tool call."""
        min_complexity = arguments.get("min_complexity")
        max_complexity = arguments.get("max_complexity")
        file_type = arguments.get("file_type")
        limit = arguments.get("limit", 10)
        
        # Use a broad query and filter by complexity
        query = "*"
        results = self.indexer.search(
            query=query,
            limit=limit * 2,  # Get more results to filter
            file_type=file_type,
            min_complexity=min_complexity,
            max_complexity=max_complexity
        )
        
        # Additional client-side filtering if needed
        if min_complexity is not None or max_complexity is not None:
            filtered_results = []
            for result in results:
                if min_complexity is not None and result.complexity_score < min_complexity:
                    continue
                if max_complexity is not None and result.complexity_score > max_complexity:
                    continue
                filtered_results.append(result)
            results = filtered_results[:limit]
        
        if not results:
            complexity_range = ""
            if min_complexity is not None and max_complexity is not None:
                complexity_range = f" (complexity: {min_complexity}-{max_complexity})"
            elif min_complexity is not None:
                complexity_range = f" (complexity: >={min_complexity})"
            elif max_complexity is not None:
                complexity_range = f" (complexity: <={max_complexity})"
            
            return [TextContent(type="text", text=f"No files found{complexity_range}")]
        
        response = self._format_search_results(results, "Files by complexity")
        return [TextContent(type="text", text=response)]
    
    async def _handle_index_directory(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle index directory tool call."""
        path = arguments["path"]
        recursive = arguments.get("recursive", True)
        
        stats = await self.indexer.index_directory(path, recursive)
        
        response = f"""Directory indexing completed for: {path}

Statistics:
- Files indexed: {stats['indexed']}
- Files skipped: {stats['skipped']}
- Errors: {stats['errors']}
- Recursive: {recursive}

Index now contains {self.indexer.get_statistics()['total_documents']} total documents."""
        
        return [TextContent(type="text", text=response)]
    
    async def _handle_index_file(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle index file tool call."""
        file_path = arguments["file_path"]
        
        success = await self.indexer.index_file(file_path)
        
        if success:
            response = f"Successfully indexed file: {file_path}"
        else:
            response = f"Failed to index file: {file_path} (unsupported type or error)"
        
        return [TextContent(type="text", text=response)]
    
    async def _handle_get_index_statistics(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle get index statistics tool call."""
        stats = self.indexer.get_statistics()
        
        response = f"""Search Index Statistics:

Total Documents: {stats['total_documents']}
Index Size: {stats['index_size']:,} bytes ({stats['index_size'] / 1024 / 1024:.1f} MB)
Index Path: {stats['index_path']}

Supported File Types:
{chr(10).join(f"  - {ext}" for ext in sorted(stats['supported_extensions']))}"""
        
        return [TextContent(type="text", text=response)]
    
    async def _handle_fuzzy_search(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle fuzzy search tool call."""
        query = arguments["query"]
        limit = arguments.get("limit", 10)
        
        # Add fuzzy search operators to the query
        fuzzy_query = f"{query}~"  # Tantivy fuzzy search syntax
        
        results = self.indexer.search(query=fuzzy_query, limit=limit)
        
        if not results:
            return [TextContent(type="text", text=f"No results found for fuzzy query: '{query}'")]
        
        response = self._format_search_results(results, f"Fuzzy search results for '{query}'")
        return [TextContent(type="text", text=response)]
    
    def _format_search_results(self, results: List[SearchResult], title: str) -> str:
        """Format search results for display."""
        if not results:
            return "No results found."
        
        lines = [f"{title}\n{'=' * len(title)}\n"]
        
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result.name} ({result.type})")
            lines.append(f"   Path: {result.path}")
            lines.append(f"   Score: {result.score:.3f} | Complexity: {result.complexity_score:.2f} | Lines: {result.line_count}")
            
            if result.entities:
                entities_str = ", ".join(result.entities[:5])
                if len(result.entities) > 5:
                    entities_str += f" (and {len(result.entities) - 5} more)"
                lines.append(f"   Entities: {entities_str}")
            
            if result.content_preview:
                preview = result.content_preview.replace('\n', ' ')[:100]
                if len(result.content_preview) > 100:
                    preview += "..."
                lines.append(f"   Preview: {preview}")
            
            lines.append("")  # Empty line between results
        
        return "\n".join(lines)
    
    async def run(self, transport_type: str = "stdio", **kwargs):
        """Run the MCP server.
        
        Args:
            transport_type: Transport type ("stdio", "sse", etc.)
            **kwargs: Additional transport arguments
        """
        logger.info(f"Starting IFS Cloud MCP Server with {transport_type} transport")
        
        if transport_type == "stdio":
            from mcp.server.stdio import stdio_server
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(read_stream, write_stream, **kwargs)
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")
    
    async def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'indexer'):
            self.indexer.close()
        logger.info("IFS Cloud MCP Server cleanup completed")