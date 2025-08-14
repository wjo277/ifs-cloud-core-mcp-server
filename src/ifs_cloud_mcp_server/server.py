"""IFS Cloud MCP Server implementation."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from mcp.server import Server
from mcp.types import Tool, TextContent, Resource, ResourceTemplate
from pydantic import BaseModel

from .indexer import IFSCloudTantivyIndexer, SearchResult
from .config import ConfigManager


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

    def __init__(
        self, index_path: Union[str, Path], name: str = "ifs-cloud-mcp-server"
    ):
        """Initialize the MCP server.

        Args:
            index_path: Path to store the Tantivy index
            name: Server name
        """
        self.name = name
        self.server = Server(name)
        self.indexer = IFSCloudTantivyIndexer(index_path)
        self.config_manager = ConfigManager()

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
                                "description": "Search query (supports full-text search)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10)",
                                "default": 10,
                            },
                            "file_type": {
                                "type": "string",
                                "description": "Filter by file type (.entity, .plsql, .views, etc.)",
                                "enum": [
                                    ".entity",
                                    ".plsql",
                                    ".views",
                                    ".storage",
                                    ".fragment",
                                    ".client",
                                    ".projection",
                                    ".plsvc",
                                ],
                            },
                            "min_complexity": {
                                "type": "number",
                                "description": "Minimum complexity score (0.0-1.0)",
                            },
                            "max_complexity": {
                                "type": "number",
                                "description": "Maximum complexity score (0.0-1.0)",
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="search_entities",
                    description="Search for files containing specific IFS entities",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {
                                "type": "string",
                                "description": "Entity name to search for",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10)",
                                "default": 10,
                            },
                        },
                        "required": ["entity"],
                    },
                ),
                Tool(
                    name="find_similar_files",
                    description="Find files similar to a given file based on entities and content",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the reference file",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of similar files (default: 5)",
                                "default": 5,
                            },
                        },
                        "required": ["file_path"],
                    },
                ),
                Tool(
                    name="search_by_complexity",
                    description="Search files by complexity score range",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "min_complexity": {
                                "type": "number",
                                "description": "Minimum complexity score (0.0-1.0)",
                            },
                            "max_complexity": {
                                "type": "number",
                                "description": "Maximum complexity score (0.0-1.0)",
                            },
                            "file_type": {
                                "type": "string",
                                "description": "Filter by file type",
                                "enum": [
                                    ".entity",
                                    ".plsql",
                                    ".views",
                                    ".storage",
                                    ".fragment",
                                    ".client",
                                    ".projection",
                                    ".plsvc",
                                ],
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10)",
                                "default": 10,
                            },
                        },
                    },
                ),
                Tool(
                    name="index_directory",
                    description="Index all IFS Cloud files in a directory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path to index",
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "Index subdirectories recursively (default: true)",
                                "default": True,
                            },
                        },
                        "required": ["path"],
                    },
                ),
                Tool(
                    name="index_file",
                    description="Index a single IFS Cloud file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to index",
                            }
                        },
                        "required": ["file_path"],
                    },
                ),
                Tool(
                    name="get_index_statistics",
                    description="Get statistics about the search index",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="fuzzy_search",
                    description="Perform fuzzy search to handle typos and partial matches",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (fuzzy matching enabled)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10)",
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="set_core_codes_path",
                    description="Set the path to IFS Cloud Core Codes directory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the IFS Cloud Core Codes directory",
                            }
                        },
                        "required": ["path"],
                    },
                ),
                Tool(
                    name="get_core_codes_path",
                    description="Get the currently configured IFS Cloud Core Codes path",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="index_core_codes",
                    description="Index the configured IFS Cloud Core Codes directory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "recursive": {
                                "type": "boolean",
                                "description": "Index subdirectories recursively (default: true)",
                                "default": True,
                            }
                        },
                    },
                ),
                Tool(
                    name="analyze_entity_dependencies",
                    description="Analyze dependencies for a specific entity across all files",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity_name": {
                                "type": "string",
                                "description": "Name of the entity to analyze",
                            }
                        },
                        "required": ["entity_name"],
                    },
                ),
                Tool(
                    name="find_overrides_and_overtakes",
                    description="Find all @Override and @Overtake annotations in the codebase",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity_name": {
                                "type": "string",
                                "description": "Optional: filter by specific entity name",
                            }
                        },
                    },
                ),
                Tool(
                    name="force_reindex_directory",
                    description="Force re-index all files in a directory, ignoring cache",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path to force re-index",
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "Re-index subdirectories recursively (default: true)",
                                "default": True,
                            },
                        },
                        "required": ["path"],
                    },
                ),
                Tool(
                    name="cleanup_cache",
                    description="Remove stale cache entries for files that no longer exist",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="get_cache_statistics",
                    description="Get detailed cache and index statistics",
                    inputSchema={"type": "object", "properties": {}},
                ),
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
                elif name == "set_core_codes_path":
                    return await self._handle_set_core_codes_path(arguments)
                elif name == "get_core_codes_path":
                    return await self._handle_get_core_codes_path(arguments)
                elif name == "index_core_codes":
                    return await self._handle_index_core_codes(arguments)
                elif name == "analyze_entity_dependencies":
                    return await self._handle_analyze_entity_dependencies(arguments)
                elif name == "find_overrides_and_overtakes":
                    return await self._handle_find_overrides_and_overtakes(arguments)
                elif name == "force_reindex_directory":
                    return await self._handle_force_reindex_directory(arguments)
                elif name == "cleanup_cache":
                    return await self._handle_cleanup_cache(arguments)
                elif name == "get_cache_statistics":
                    return await self._handle_get_cache_statistics(arguments)
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
                    mimeType="application/json",
                ),
                Resource(
                    uri="ifs-cloud://supported/file-types",
                    name="Supported File Types",
                    description="List of supported IFS Cloud file types",
                    mimeType="application/json",
                ),
            ]

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read resource content."""
            if uri == "ifs-cloud://index/statistics":
                stats = self.indexer.get_statistics()
                return json.dumps(stats, indent=2)
            elif uri == "ifs-cloud://supported/file-types":
                return json.dumps(
                    {
                        "supported_extensions": list(self.indexer.SUPPORTED_EXTENSIONS),
                        "descriptions": {
                            ".entity": "Entity definitions",
                            ".plsql": "PL/SQL code",
                            ".views": "Database views",
                            ".storage": "Storage configurations",
                            ".fragment": "Code fragments",
                            ".client": "Client-side code",
                            ".projection": "Data projections",
                            ".plsvc": "PL/SQL service layer",
                        },
                    },
                    indent=2,
                )
            else:
                raise ValueError(f"Unknown resource URI: {uri}")

    async def _handle_search_content(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
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
            max_complexity=max_complexity,
        )

        if not results:
            return [
                TextContent(type="text", text=f"No results found for query: '{query}'")
            ]

        response = self._format_search_results(
            results, f"Content search results for '{query}'"
        )
        return [TextContent(type="text", text=response)]

    async def _handle_search_entities(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle entity search tool call."""
        entity = arguments["entity"]
        limit = arguments.get("limit", 10)

        # Search in entities field specifically
        query = f"entities:{entity}"
        results = self.indexer.search(query=query, limit=limit)

        if not results:
            return [
                TextContent(
                    type="text", text=f"No files found containing entity: '{entity}'"
                )
            ]

        response = self._format_search_results(
            results, f"Files containing entity '{entity}'"
        )
        return [TextContent(type="text", text=response)]

    async def _handle_find_similar_files(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle find similar files tool call."""
        file_path = arguments["file_path"]
        limit = arguments.get("limit", 5)

        results = self.indexer.find_similar_files(file_path, limit)

        if not results:
            return [
                TextContent(
                    type="text", text=f"No similar files found for: {file_path}"
                )
            ]

        # Filter out the original file if it appears in results
        filtered_results = [r for r in results if r.path != str(file_path)][:limit]

        response = self._format_search_results(
            filtered_results, f"Files similar to '{file_path}'"
        )
        return [TextContent(type="text", text=response)]

    async def _handle_search_by_complexity(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
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
            max_complexity=max_complexity,
        )

        # Additional client-side filtering if needed
        if min_complexity is not None or max_complexity is not None:
            filtered_results = []
            for result in results:
                if (
                    min_complexity is not None
                    and result.complexity_score < min_complexity
                ):
                    continue
                if (
                    max_complexity is not None
                    and result.complexity_score > max_complexity
                ):
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

    async def _handle_index_directory(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle index directory tool call with caching support."""
        path = arguments["path"]
        recursive = arguments.get("recursive", True)

        stats = await self.indexer.index_directory(path, recursive)

        response_lines = [
            f"Directory indexing completed for: {path}",
            "",
            "Statistics:",
            f"  • Files indexed: {stats['indexed']}",
            f"  • Files cached: {stats['cached']}",
            f"  • Files skipped: {stats['skipped']}",
            f"  • Errors: {stats['errors']}",
            f"  • Recursive: {recursive}",
            "",
            f"Index now contains {self.indexer.get_statistics()['total_documents']} total documents.",
        ]

        if stats["cached"] > 0:
            response_lines.extend(
                [
                    "",
                    f"⚡ Performance: {stats['cached']} files were already cached and up-to-date!",
                    "   This significantly improved indexing speed.",
                ]
            )

        return [TextContent(type="text", text="\n".join(response_lines))]

    async def _handle_index_file(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle index file tool call."""
        file_path = arguments["file_path"]

        success = await self.indexer.index_file(file_path)

        if success:
            response = f"Successfully indexed file: {file_path}"
        else:
            response = f"Failed to index file: {file_path} (unsupported type or error)"

        return [TextContent(type="text", text=response)]

    async def _handle_get_index_statistics(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle get index statistics tool call."""
        stats = self.indexer.get_statistics()

        response = f"""Search Index Statistics:

Total Documents: {stats['total_documents']}
Index Size: {stats['index_size']:,} bytes ({stats['index_size'] / 1024 / 1024:.1f} MB)
Index Path: {stats['index_path']}

Supported File Types:
{chr(10).join(f"  - {ext}" for ext in sorted(stats['supported_extensions']))}"""

        return [TextContent(type="text", text=response)]

    async def _handle_fuzzy_search(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle fuzzy search tool call."""
        query = arguments["query"]
        limit = arguments.get("limit", 10)

        # Add fuzzy search operators to the query
        fuzzy_query = f"{query}~"  # Tantivy fuzzy search syntax

        results = self.indexer.search(query=fuzzy_query, limit=limit)

        if not results:
            return [
                TextContent(
                    type="text", text=f"No results found for fuzzy query: '{query}'"
                )
            ]

        response = self._format_search_results(
            results, f"Fuzzy search results for '{query}'"
        )
        return [TextContent(type="text", text=response)]

    def _format_search_results(self, results: List[SearchResult], title: str) -> str:
        """Format search results for display."""
        if not results:
            return "No results found."

        lines = [f"{title}\n{'=' * len(title)}\n"]

        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result.name} ({result.type})")
            lines.append(f"   Path: {result.path}")
            lines.append(
                f"   Score: {result.score:.3f} | Complexity: {result.complexity_score:.2f} | Lines: {result.line_count}"
            )

            if result.entities:
                entities_str = ", ".join(result.entities[:5])
                if len(result.entities) > 5:
                    entities_str += f" (and {len(result.entities) - 5} more)"
                lines.append(f"   Entities: {entities_str}")

            if result.content_preview:
                preview = result.content_preview.replace("\n", " ")[:100]
                if len(result.content_preview) > 100:
                    preview += "..."
                lines.append(f"   Preview: {preview}")

            lines.append("")  # Empty line between results

        return "\n".join(lines)

    async def _handle_set_core_codes_path(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle setting the IFS Cloud Core Codes path."""
        path = arguments["path"]

        if self.config_manager.set_core_codes_path(path):
            return [
                TextContent(
                    type="text",
                    text=f"Successfully set IFS Cloud Core Codes path to: {path}",
                )
            ]
        else:
            return [
                TextContent(
                    type="text",
                    text=f"Failed to set path. Please ensure the directory exists: {path}",
                )
            ]

    async def _handle_get_core_codes_path(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle getting the IFS Cloud Core Codes path."""
        path = self.config_manager.get_core_codes_path()

        if path:
            return [TextContent(type="text", text=f"IFS Cloud Core Codes path: {path}")]
        else:
            return [
                TextContent(
                    type="text",
                    text="No IFS Cloud Core Codes path configured. Use 'set_core_codes_path' tool to configure it.",
                )
            ]

    async def _handle_index_core_codes(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle indexing the configured core codes directory."""
        recursive = arguments.get("recursive", True)

        core_path = self.config_manager.get_core_codes_path()
        if not core_path:
            return [
                TextContent(
                    type="text",
                    text="No IFS Cloud Core Codes path configured. Use 'set_core_codes_path' tool first.",
                )
            ]

        try:
            indexed_count = await self.indexer.index_directory(
                core_path, recursive=recursive
            )
            self.indexer.commit()

            # Update last indexed timestamp
            self.config_manager.set_last_indexed(datetime.now().isoformat())

            return [
                TextContent(
                    type="text",
                    text=f"Successfully indexed {indexed_count} files from: {core_path}",
                )
            ]
        except Exception as e:
            logger.error(f"Error indexing core codes: {e}")
            return [
                TextContent(type="text", text=f"Error indexing core codes: {str(e)}")
            ]

    async def _handle_analyze_entity_dependencies(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle analyzing entity dependencies."""
        entity_name = arguments["entity_name"]

        # Search for files that contain this entity
        entity_results = self.indexer.search(query=f"entities:{entity_name}", limit=50)

        # Search for files that depend on this entity
        dependency_results = self.indexer.search(
            query=f"dependencies:{entity_name}", limit=50
        )

        response_lines = [
            f"Dependency Analysis for Entity: {entity_name}",
            "=" * 50,
            "",
        ]

        if entity_results:
            response_lines.extend(
                [
                    f"Files defining/using this entity ({len(entity_results)} found):",
                    "-" * 40,
                ]
            )
            for result in entity_results[:10]:  # Limit to first 10 for readability
                response_lines.append(
                    f"  • {result.path} ({result.type}) - Score: {result.score:.2f}"
                )

            if len(entity_results) > 10:
                response_lines.append(
                    f"  ... and {len(entity_results) - 10} more files"
                )
        else:
            response_lines.append(
                f"No files found defining/using entity: {entity_name}"
            )

        response_lines.append("")

        if dependency_results:
            response_lines.extend(
                [
                    f"Files depending on this entity ({len(dependency_results)} found):",
                    "-" * 40,
                ]
            )
            for result in dependency_results[:10]:
                response_lines.append(
                    f"  • {result.path} ({result.type}) - Score: {result.score:.2f}"
                )

            if len(dependency_results) > 10:
                response_lines.append(
                    f"  ... and {len(dependency_results) - 10} more files"
                )
        else:
            response_lines.append(f"No files found depending on entity: {entity_name}")

        return [TextContent(type="text", text="\n".join(response_lines))]

    async def _handle_find_overrides_and_overtakes(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle finding Override and Overtake annotations."""
        entity_name = arguments.get("entity_name")

        # Search for @Override and @Overtake patterns
        query = "@Override OR @Overtake"
        if entity_name:
            query += f" AND entities:{entity_name}"

        results = self.indexer.search(query=query, limit=50)

        response_lines = ["Override and Overtake Analysis", "=" * 40, ""]

        if entity_name:
            response_lines[0] += f" for Entity: {entity_name}"

        if not results:
            search_scope = f" for entity '{entity_name}'" if entity_name else ""
            response_lines.append(
                f"No @Override or @Overtake annotations found{search_scope}"
            )
        else:
            response_lines.extend(
                [
                    f"Found {len(results)} files with override/overtake annotations:",
                    "-" * 50,
                ]
            )

            for result in results:
                response_lines.extend(
                    [
                        f"File: {result.path} ({result.type})",
                        f"  Entities: {', '.join(result.entities) if result.entities else 'None'}",
                        f"  Preview: {result.content_preview[:100]}...",
                        "",
                    ]
                )

        return [TextContent(type="text", text="\n".join(response_lines))]

    async def _handle_force_reindex_directory(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle force re-indexing a directory."""
        path = arguments.get("path")
        recursive = arguments.get("recursive", True)

        if not path:
            return [TextContent(type="text", text="Error: path parameter is required")]

        try:
            directory_path = Path(path)
            if not directory_path.exists():
                return [
                    TextContent(type="text", text=f"Error: Directory not found: {path}")
                ]

            stats = await self.indexer.index_directory(
                directory_path, recursive=recursive, force_reindex=True
            )

            response_lines = [
                "Force Re-indexing Complete",
                "=" * 30,
                "",
                f"Directory: {path}",
                f"Recursive: {recursive}",
                "",
                "Results:",
                f"  • Files indexed: {stats['indexed']}",
                f"  • Files cached: {stats['cached']}",
                f"  • Files skipped: {stats['skipped']}",
                f"  • Errors: {stats['errors']}",
                "",
                f"Total processed: {stats['indexed'] + stats['cached'] + stats['skipped']}",
            ]

            return [TextContent(type="text", text="\n".join(response_lines))]

        except Exception as e:
            return [
                TextContent(
                    type="text", text=f"Error during force re-indexing: {str(e)}"
                )
            ]

    async def _handle_cleanup_cache(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle cleaning up stale cache entries."""
        try:
            removed_count = self.indexer.cleanup_cache()

            response_lines = [
                "Cache Cleanup Complete",
                "=" * 25,
                "",
                f"Stale entries removed: {removed_count}",
            ]

            if removed_count > 0:
                response_lines.extend(
                    [
                        "",
                        "Cache has been cleaned of entries for files that no longer exist.",
                        "This helps improve performance and accuracy.",
                    ]
                )
            else:
                response_lines.extend(["", "No stale entries found. Cache is clean!"])

            return [TextContent(type="text", text="\n".join(response_lines))]

        except Exception as e:
            return [
                TextContent(type="text", text=f"Error during cache cleanup: {str(e)}")
            ]

    async def _handle_get_cache_statistics(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle getting detailed cache and index statistics."""
        try:
            stats = self.indexer.get_statistics()

            # Format file sizes
            def format_size(size_bytes):
                for unit in ["B", "KB", "MB", "GB"]:
                    if size_bytes < 1024:
                        return f"{size_bytes:.1f} {unit}"
                    size_bytes /= 1024
                return f"{size_bytes:.1f} TB"

            response_lines = [
                "Index & Cache Statistics",
                "=" * 30,
                "",
                "INDEX STATUS:",
                f"  • Total documents: {stats['total_documents']:,}",
                f"  • Index size: {format_size(stats['index_size'])}",
                f"  • Index path: {stats['index_path']}",
                "",
                "CACHE STATUS:",
                f"  • Cached files: {stats['cached_files']:,}",
                f"  • Cache size: {format_size(stats['cache_size'])}",
                f"  • Cache metadata: {stats['cache_metadata_path']}",
                "",
                "SUPPORTED FILE TYPES:",
                f"  • {', '.join(stats['supported_extensions'])}",
                "",
                "PERFORMANCE BENEFITS:",
                f"  • Cache hit ratio improves indexing speed significantly",
                f"  • Only changed files are re-processed",
                f"  • Metadata tracking prevents unnecessary work",
            ]

            return [TextContent(type="text", text="\n".join(response_lines))]

        except Exception as e:
            return [
                TextContent(
                    type="text", text=f"Error getting cache statistics: {str(e)}"
                )
            ]

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
        if hasattr(self, "indexer"):
            self.indexer.close()
        logger.info("IFS Cloud MCP Server cleanup completed")
