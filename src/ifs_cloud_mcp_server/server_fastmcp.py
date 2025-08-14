"""IFS Cloud MCP Server implementation using FastMCP."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from fastmcp import FastMCP

from .indexer import IFSCloudTantivyIndexer, SearchResult
from .config import ConfigManager


logger = logging.getLogger(__name__)


class IFSCloudMCPServer:
    """MCP Server for IFS Cloud with Tantivy search integration using FastMCP."""

    def __init__(
        self, index_path: Union[str, Path], name: str = "ifs-cloud-mcp-server"
    ):
        """Initialize the MCP server.

        Args:
            index_path: Path to store the Tantivy index
            name: Server name
        """
        self.name = name
        self.mcp = FastMCP(name)
        self.indexer = IFSCloudTantivyIndexer(index_path)
        self.config_manager = ConfigManager()

        # Register tools
        self._register_tools()

        logger.info(f"Initialized IFS Cloud MCP Server: {name}")

    def _register_tools(self):
        """Register MCP tools for search and indexing operations."""

        @self.mcp.tool()
        async def search_content(
            query: str,
            limit: int = 10,
            file_type: Optional[str] = None,
            min_complexity: Optional[float] = None,
            max_complexity: Optional[float] = None,
        ) -> str:
            """Search IFS Cloud files by content with advanced filtering.

            Args:
                query: Search query (supports full-text search)
                limit: Maximum number of results (default: 10)
                file_type: Filter by file type (.entity, .plsql, .views, etc.)
                min_complexity: Minimum complexity score (0.0-1.0)
                max_complexity: Maximum complexity score (0.0-1.0)
            """
            results = self.indexer.search(
                query=query,
                limit=limit,
                file_type=file_type,
                min_complexity=min_complexity,
                max_complexity=max_complexity,
            )

            if not results:
                return f"No results found for query: '{query}'"

            return self._format_search_results(
                results, f"Content search results for '{query}'"
            )

        @self.mcp.tool()
        async def search_entities(entity: str, limit: int = 10) -> str:
            """Search for files containing specific IFS entities.

            Args:
                entity: Entity name to search for
                limit: Maximum number of results (default: 10)
            """
            # Search in entities field specifically
            query = f"entities:{entity}"
            results = self.indexer.search(query=query, limit=limit)

            if not results:
                return f"No files found containing entity: '{entity}'"

            return self._format_search_results(
                results, f"Files containing entity '{entity}'"
            )

        @self.mcp.tool()
        async def search_by_module(
            query: str,
            module: Optional[str] = None,
            logical_unit: Optional[str] = None,
            limit: int = 10,
        ) -> str:
            """Search IFS Cloud files with module and logical unit filtering.

            Args:
                query: Search query (supports full-text search)
                module: Filter by IFS module (e.g., "proj", "career", "finance")
                logical_unit: Filter by logical unit (entity name from filename)
                limit: Maximum number of results (default: 10)
            """
            # Enhance the query with module and logical unit filters
            enhanced_query = query
            if module:
                enhanced_query += f" module:{module}"
            if logical_unit:
                enhanced_query += f" logical_unit:{logical_unit}"

            results = self.indexer.search(query=enhanced_query, limit=limit)

            if not results:
                filter_desc = ""
                if module and logical_unit:
                    filter_desc = (
                        f" in module '{module}' with logical unit '{logical_unit}'"
                    )
                elif module:
                    filter_desc = f" in module '{module}'"
                elif logical_unit:
                    filter_desc = f" with logical unit '{logical_unit}'"
                return f"No results found for '{query}'{filter_desc}"

            # Sort results to prioritize exact module and logical unit matches
            def sort_key(result):
                score = result.score
                # Boost score for exact module match
                if module and result.module == module:
                    score += 1.0
                # Boost score for exact logical unit match
                if logical_unit and result.logical_unit == logical_unit:
                    score += 1.0
                return -score  # Negative for descending order

            sorted_results = sorted(results, key=sort_key)

            title_parts = [f"Module-aware search results for '{query}'"]
            if module:
                title_parts.append(f"Module: {module}")
            if logical_unit:
                title_parts.append(f"Logical Unit: {logical_unit}")

            return self._format_search_results(sorted_results, " | ".join(title_parts))

        @self.mcp.tool()
        async def find_similar_files(file_path: str, limit: int = 5) -> str:
            """Find files similar to a given file based on entities and content.

            Args:
                file_path: Path to the reference file
                limit: Maximum number of similar files (default: 5)
            """
            results = self.indexer.find_similar_files(file_path, limit)

            if not results:
                return f"No similar files found for: {file_path}"

            # Filter out the original file if it appears in results
            filtered_results = [r for r in results if r.path != str(file_path)][:limit]

            return self._format_search_results(
                filtered_results, f"Files similar to '{file_path}'"
            )

        @self.mcp.tool()
        async def search_by_complexity(
            min_complexity: Optional[float] = None,
            max_complexity: Optional[float] = None,
            file_type: Optional[str] = None,
            limit: int = 10,
        ) -> str:
            """Search files by complexity score range.

            Args:
                min_complexity: Minimum complexity score (0.0-1.0)
                max_complexity: Maximum complexity score (0.0-1.0)
                file_type: Filter by file type
                limit: Maximum number of results (default: 10)
            """
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
                    complexity_range = (
                        f" (complexity: {min_complexity}-{max_complexity})"
                    )
                elif min_complexity is not None:
                    complexity_range = f" (complexity: >={min_complexity})"
                elif max_complexity is not None:
                    complexity_range = f" (complexity: <={max_complexity})"

                return f"No files found{complexity_range}"

            return self._format_search_results(results, "Files by complexity")

        @self.mcp.tool()
        async def index_directory(path: str, recursive: bool = True) -> str:
            """Index all IFS Cloud files in a directory.

            Args:
                path: Directory path to index
                recursive: Index subdirectories recursively (default: true)
            """
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

            return "\n".join(response_lines)

        @self.mcp.tool()
        async def index_file(file_path: str) -> str:
            """Index a single IFS Cloud file.

            Args:
                file_path: Path to the file to index
            """
            success = await self.indexer.index_file(file_path)

            if success:
                return f"Successfully indexed file: {file_path}"
            else:
                return f"Failed to index file: {file_path} (unsupported type or error)"

        @self.mcp.tool()
        async def get_index_statistics() -> str:
            """Get statistics about the search index."""
            stats = self.indexer.get_statistics()

            return f"""Search Index Statistics:

Total Documents: {stats['total_documents']}
Index Size: {stats['index_size']:,} bytes ({stats['index_size'] / 1024 / 1024:.1f} MB)
Index Path: {stats['index_path']}

Supported File Types:
{chr(10).join(f"  - {ext}" for ext in sorted(stats['supported_extensions']))}"""

        @self.mcp.tool()
        async def fuzzy_search(query: str, limit: int = 10) -> str:
            """Perform fuzzy search to handle typos and partial matches.

            Args:
                query: Search query (fuzzy matching enabled)
                limit: Maximum number of results (default: 10)
            """
            # Add fuzzy search operators to the query
            fuzzy_query = f"{query}~"  # Tantivy fuzzy search syntax

            results = self.indexer.search(query=fuzzy_query, limit=limit)

            if not results:
                return f"No results found for fuzzy query: '{query}'"

            return self._format_search_results(
                results, f"Fuzzy search results for '{query}'"
            )

        @self.mcp.tool()
        async def set_core_codes_path(path: str) -> str:
            """Set the path to IFS Cloud Core Codes directory.

            Args:
                path: Path to the IFS Cloud Core Codes directory
            """
            if self.config_manager.set_core_codes_path(path):
                return f"Successfully set IFS Cloud Core Codes path to: {path}"
            else:
                return f"Failed to set path. Please ensure the directory exists: {path}"

        @self.mcp.tool()
        async def get_core_codes_path() -> str:
            """Get the currently configured IFS Cloud Core Codes path."""
            path = self.config_manager.get_core_codes_path()

            if path:
                return f"IFS Cloud Core Codes path: {path}"
            else:
                return "No IFS Cloud Core Codes path configured. Use 'set_core_codes_path' tool to configure it."

        @self.mcp.tool()
        async def index_core_codes(recursive: bool = True) -> str:
            """Index the configured IFS Cloud Core Codes directory.

            Args:
                recursive: Index subdirectories recursively (default: true)
            """
            core_path = self.config_manager.get_core_codes_path()
            if not core_path:
                return "No IFS Cloud Core Codes path configured. Use 'set_core_codes_path' tool first."

            try:
                stats = await self.indexer.index_directory(
                    core_path, recursive=recursive
                )

                # Update last indexed timestamp
                self.config_manager.set_last_indexed(datetime.now().isoformat())

                response_lines = [
                    f"Successfully indexed IFS Cloud Core Codes from: {core_path}",
                    "",
                    "Statistics:",
                    f"  • Files indexed: {stats['indexed']}",
                    f"  • Files cached: {stats['cached']}",
                    f"  • Files skipped: {stats['skipped']}",
                    f"  • Errors: {stats['errors']}",
                ]

                return "\n".join(response_lines)

            except Exception as e:
                logger.error(f"Error indexing core codes: {e}")
                return f"Error indexing core codes: {str(e)}"

        @self.mcp.tool()
        async def analyze_entity_dependencies(entity_name: str) -> str:
            """Analyze dependencies for a specific entity across all files.

            Args:
                entity_name: Name of the entity to analyze
            """
            # Search for files that contain this entity
            entity_results = self.indexer.search(
                query=f"entities:{entity_name}", limit=50
            )

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
                response_lines.append(
                    f"No files found depending on entity: {entity_name}"
                )

            return "\n".join(response_lines)

        @self.mcp.tool()
        async def find_overrides_and_overtakes(
            entity_name: Optional[str] = None,
        ) -> str:
            """Find all @Override and @Overtake annotations in the codebase.

            Args:
                entity_name: Optional: filter by specific entity name
            """
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

            return "\n".join(response_lines)

        @self.mcp.tool()
        async def force_reindex_directory(path: str, recursive: bool = True) -> str:
            """Force re-index all files in a directory, ignoring cache.

            Args:
                path: Directory path to force re-index
                recursive: Re-index subdirectories recursively (default: true)
            """
            try:
                directory_path = Path(path)
                if not directory_path.exists():
                    return f"Error: Directory not found: {path}"

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

                return "\n".join(response_lines)

            except Exception as e:
                return f"Error during force re-indexing: {str(e)}"

        @self.mcp.tool()
        async def cleanup_cache() -> str:
            """Remove stale cache entries for files that no longer exist."""
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
                    response_lines.extend(
                        ["", "No stale entries found. Cache is clean!"]
                    )

                return "\n".join(response_lines)

            except Exception as e:
                return f"Error during cache cleanup: {str(e)}"

        @self.mcp.tool()
        async def get_cache_statistics() -> str:
            """Get detailed cache and index statistics."""
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

                return "\n".join(response_lines)

            except Exception as e:
                return f"Error getting cache statistics: {str(e)}"

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

            # Add IFS Cloud structure information
            ifs_info_parts = []
            if result.module:
                ifs_info_parts.append(f"Module: {result.module}")
            if result.logical_unit:
                ifs_info_parts.append(f"Logical Unit: {result.logical_unit}")
            if result.entity_name:
                ifs_info_parts.append(f"Entity: {result.entity_name}")
            if result.component:
                ifs_info_parts.append(f"Component: {result.component}")

            if ifs_info_parts:
                lines.append(f"   IFS: {' | '.join(ifs_info_parts)}")

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

    def run(self, transport_type: str = "stdio", **kwargs):
        """Run the MCP server (synchronous version for FastMCP).

        Args:
            transport_type: Transport type ("stdio", "sse", etc.)
            **kwargs: Additional transport arguments
        """
        logger.info(f"Starting IFS Cloud MCP Server with {transport_type} transport")

        if transport_type == "stdio":
            self.mcp.run(transport="stdio")
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "indexer"):
            self.indexer.close()
        logger.info("IFS Cloud MCP Server cleanup completed")
