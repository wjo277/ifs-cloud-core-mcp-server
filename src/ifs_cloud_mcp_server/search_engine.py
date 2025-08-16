"""
IFS Cloud Search Engine

Main search orchestrator

This module provides the primary search functionality for IFS Cloud files,
coordinating between the indexer and various search enhancements.
"""

import logging
from typing import Dict, List, Optional, Set, Any, Union
from pathlib import Path
from collections import Counter

from .indexer import IFSCloudIndexer, SearchResult
from .enhanced_search import (
    MetadataEnhancedSearchEngine,
    SearchContext,
    SearchResult as EnhancedSearchResult,
)
from .metadata_extractor import MetadataManager

logger = logging.getLogger(__name__)


class IFSCloudSearchEngine:
    """Main search engine for IFS Cloud files with metadata enhancement capabilities."""

    def __init__(self, indexer: "IFSCloudIndexer", metadata_dir: Optional[Path] = None):
        """
        Initialize the search engine.

        Args:
            indexer: IFSCloudIndexer instance for accessing the file index
            metadata_dir: Optional directory for metadata storage (defaults to index_path/metadata)
        """
        self.indexer = indexer

        # Set up metadata directory
        if metadata_dir is None:
            metadata_dir = Path(indexer.index_path) / "metadata"

        self.metadata_manager = MetadataManager(metadata_dir)
        self.enhanced_search_engine: Optional[MetadataEnhancedSearchEngine] = None
        self.current_ifs_version: Optional[str] = None

    def set_ifs_version(self, version: str) -> bool:
        """
        Set the IFS version for metadata-enhanced search.

        Args:
            version: IFS version (e.g., "25.1.0")

        Returns:
            True if metadata was successfully loaded
        """
        try:
            self.current_ifs_version = version
            metadata = self.metadata_manager.get_metadata(version)

            if metadata:
                self.enhanced_search_engine = MetadataEnhancedSearchEngine(
                    base_search_engine=self, metadata_manager=self.metadata_manager
                )
                self.enhanced_search_engine.set_ifs_version(version)
                logger.info(f"Enhanced search enabled for IFS version {version}")
                return True
            else:
                logger.warning(f"No metadata found for IFS version {version}")
                self.enhanced_search_engine = None
                return False

        except Exception as e:
            logger.error(f"Failed to set IFS version {version}: {e}")
            self.enhanced_search_engine = None
            return False

    def search(
        self,
        query: str,
        limit: int = 10,
        file_type: Optional[str] = None,
        min_complexity: Optional[float] = None,
        max_complexity: Optional[float] = None,
        module: Optional[str] = None,
        logical_unit: Optional[str] = None,
        include_related: bool = False,
        max_related_per_lu: int = 5,
    ) -> List[SearchResult]:
        """
        Primary search method with optional related file inclusion.

        Args:
            query: Search query
            limit: Maximum number of results
            file_type: Filter by file type (optional)
            min_complexity: Minimum complexity score (optional)
            max_complexity: Maximum complexity score (optional)
            module: Filter by module (optional)
            logical_unit: Filter by logical unit (optional)
            include_related: Whether to include related files from same logical units
            max_related_per_lu: Maximum related files per logical unit

        Returns:
            List of search results
        """
        if include_related:
            return self.search_with_related_files(
                query=query,
                limit=limit,
                file_type=file_type,
                min_complexity=min_complexity,
                max_complexity=max_complexity,
                max_related_per_lu=max_related_per_lu,
                module=module,
                logical_unit=logical_unit,
            )
        else:
            # Dual-search approach: combine index and metadata results
            all_results = []
            seen_paths = set()

            # 1. Get index-based results (traditional search) - prioritize this
            index_results = self.indexer.search_deduplicated(
                query=query,
                limit=limit * 3,  # Get more results to allow for better merging
                file_type=None,  # Don't filter at indexer level, we'll filter later
                min_complexity=min_complexity,
                max_complexity=max_complexity,
            )

            # Add index results with proper scoring
            for result in index_results:
                if result.path not in seen_paths:
                    all_results.append(result)
                    seen_paths.add(result.path)

            # 2. Get metadata-enhanced results if available (but limit this)
            if self.enhanced_search_engine and len(all_results) < limit * 2:
                try:
                    from .enhanced_search import SearchContext

                    # Map file_type to content_types_filter for metadata search
                    content_types_filter = None
                    if file_type:
                        # Remove the dot and map to content type
                        content_type = file_type.lstrip(".")
                        content_types_filter = [content_type]

                    context = SearchContext(
                        query=query,
                        limit=max(
                            10, limit
                        ),  # Limit metadata search to prevent too many calls
                        modules_filter=[module] if module else None,
                        content_types_filter=content_types_filter,
                        logical_units_filter=[logical_unit] if logical_unit else None,
                        fuzzy_threshold=80.0,  # Higher threshold for more precision
                        include_related=False,
                    )

                    enhanced_results = self.enhanced_search_engine.enhanced_search(
                        context
                    )

                    # Convert enhanced results to SearchResult and merge (limit to prevent spam)
                    for enhanced_result in enhanced_results[:limit]:
                        if enhanced_result.file_path not in seen_paths:
                            # Convert enhanced result to SearchResult
                            search_result = self._convert_enhanced_to_search_result(
                                enhanced_result
                            )
                            # Boost score for metadata-found results
                            search_result.score = (
                                enhanced_result.confidence * 1.5
                            )  # Moderate boost for metadata matches
                            all_results.append(search_result)
                            seen_paths.add(enhanced_result.file_path)

                except Exception as e:
                    logger.debug(f"Metadata search failed: {e}")

            # 3. Re-rank combined results using business logic
            all_results = self._apply_business_ranking(all_results, query)

            # 4. Apply final filters that weren't handled by individual searches
            filtered_results = self._apply_post_search_filters(
                all_results,
                file_type,
                module,
                logical_unit,
                min_complexity,
                max_complexity,
            )

            # 5. Sort by final score and limit
            filtered_results.sort(key=lambda x: x.score, reverse=True)
            return filtered_results[:limit]

    def search_with_related_files(
        self,
        query: str,
        limit: int = 10,
        file_type: Optional[str] = None,
        min_complexity: Optional[float] = None,
        max_complexity: Optional[float] = None,
        max_related_per_lu: int = 5,
        module: Optional[str] = None,
        logical_unit: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search with related files from the same logical units.

        When a logical unit is found (e.g., "Activity"), includes related files
        like Activity.entity, Activity.plsql, Activity.views, etc.
        """
        # Get primary search results (use regular search method to apply filters)
        primary_results = self.search(
            query=query,
            limit=limit,
            file_type=file_type,
            min_complexity=min_complexity,
            max_complexity=max_complexity,
            module=module,
            logical_unit=logical_unit,
            include_related=False,  # Avoid recursion
        )

        if not primary_results:
            return primary_results

        # Collect unique logical units from primary results
        logical_units = set()
        primary_paths = set()

        for result in primary_results:
            primary_paths.add(result.path)
            if result.logical_unit:
                logical_units.add(result.logical_unit)

        # Find related files for each logical unit
        related_results = []

        for lu in logical_units:
            related_files = self.find_related_files(lu)

            # Add related files not already in primary results
            related_count = 0
            for related in related_files:
                if (
                    related.path not in primary_paths
                    and related_count < max_related_per_lu
                ):
                    # Apply file type filter to related files too
                    if file_type and related.type:
                        if not related.type.endswith(file_type):
                            continue
                    elif file_type and not related.type:
                        continue

                    # Reduce score slightly to indicate it's a related result
                    related.score *= 0.8
                    related_results.append(related)
                    related_count += 1

        # Combine and sort results
        all_results = primary_results + related_results
        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results[
            : limit * 2
        ]  # Allow more results when including related files

    def find_related_files(
        self, logical_unit: str, include_extensions: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Find all files related to a specific logical unit.

        Args:
            logical_unit: Name of the logical unit (e.g., "Activity", "CustomerOrder")
            include_extensions: File extensions to include (default: all supported)

        Returns:
            List of related files for the logical unit
        """
        if not logical_unit:
            return []

        include_extensions = include_extensions or list(
            self.indexer.SUPPORTED_EXTENSIONS
        )

        # Search for files matching the logical unit
        results = []

        # Primary patterns to match
        patterns = [
            logical_unit,  # Exact match
            f"{logical_unit}API",  # API files
            f"{logical_unit}Handling",  # Handling files
            f"{logical_unit}Client",  # Client files
        ]

        for pattern in patterns:
            search_results = self.indexer.search_deduplicated(
                query=pattern, limit=50  # Get more results for filtering
            )

            # Filter by file extension
            for result in search_results:
                if any(result.type.endswith(ext) for ext in include_extensions):
                    # Check if this is truly related (not just a coincidental match)
                    if (
                        result.logical_unit
                        and result.logical_unit.lower() == logical_unit.lower()
                    ) or (logical_unit.lower() in result.name.lower()):
                        results.append(result)

        # Deduplicate and sort by relevance
        seen_paths = set()
        unique_results = []

        for result in results:
            if result.path not in seen_paths:
                seen_paths.add(result.path)
                unique_results.append(result)

        # Sort by file type priority and score
        file_type_priority = {
            ".entity": 1,  # Core definition
            ".plsql": 2,  # Business logic
            ".views": 3,  # Data access
            ".projection": 4,  # API layer
            ".client": 5,  # UI layer
            ".fragment": 6,  # Reusable components
            ".storage": 7,  # Database schema
            ".plsvc": 8,  # Service layer
        }

        unique_results.sort(
            key=lambda r: (file_type_priority.get(r.type, 99), -r.score)
        )

        return unique_results

    def enhanced_search(
        self,
        query: str,
        limit: int = 10,
        modules_filter: Optional[List[str]] = None,
        content_types_filter: Optional[List[str]] = None,
        logical_units_filter: Optional[List[str]] = None,
        fuzzy_threshold: float = 80.0,
        include_related: bool = True,
    ) -> List[EnhancedSearchResult]:
        """
        Perform metadata-enhanced search with intelligent ranking and related files.

        Args:
            query: Search query
            limit: Maximum number of results
            modules_filter: Filter by IFS modules (e.g., ['ORDER', 'PERSON'])
            content_types_filter: Filter by file types (e.g., ['entity', 'plsql'])
            logical_units_filter: Filter by logical units
            fuzzy_threshold: Fuzzy matching threshold (0-100)
            include_related: Include related files for matched logical units

        Returns:
            List of enhanced search results with metadata context
        """
        context = SearchContext(
            query=query,
            limit=limit,
            modules_filter=modules_filter or [],
            content_types_filter=content_types_filter or [],
            logical_units_filter=logical_units_filter or [],
            fuzzy_threshold=fuzzy_threshold,
            include_related=include_related,
        )

        if self.enhanced_search_engine:
            # Use enhanced search with metadata
            return self.enhanced_search_engine.enhanced_search(context)
        else:
            # Fallback to basic search and convert results
            return self._fallback_enhanced_search(context)

    def _fallback_enhanced_search(
        self, context: SearchContext
    ) -> List[EnhancedSearchResult]:
        """Fallback enhanced search without metadata."""
        # Get basic search results
        basic_results = self.search(
            query=context.query,
            limit=context.limit,
            include_related=context.include_related,
        )

        # Convert to enhanced results
        enhanced_results = []
        for result in basic_results:
            enhanced_results.append(
                EnhancedSearchResult(
                    file_path=result.path,
                    content_type=result.type.lstrip("."),  # Remove leading dot
                    line_number=1,
                    snippet=result.content_preview,
                    confidence=min(100.0, result.score / 10.0),  # Normalize score
                    logical_unit=result.logical_unit,
                    module=result.module,
                    business_description=None,
                    related_entities=[],
                    search_context=["Basic search - no metadata available"],
                )
            )

        return enhanced_results

    def suggest_related_searches(self, query: str, limit: int = 5) -> List[str]:
        """
        Suggest related search terms based on metadata and index content.

        Args:
            query: Original search query
            limit: Maximum number of suggestions

        Returns:
            List of suggested search terms
        """
        if self.enhanced_search_engine:
            return self.enhanced_search_engine.suggest_related_searches(query, limit)
        else:
            # Basic suggestions from index content
            return self._basic_suggest_related(query, limit)

    def _basic_suggest_related(self, query: str, limit: int) -> List[str]:
        """Basic related search suggestions without metadata."""
        suggestions = []

        # Get some search results to extract related terms
        results = self.search(query, limit=10)

        # Extract entities from results as suggestions
        entity_counts = Counter()

        for result in results:
            # Add entities from the result
            for entity in result.entities:
                if entity.lower() != query.lower():  # Don't suggest the same term
                    entity_counts[entity] += 1

        # Return most common entities as suggestions
        suggestions = [entity for entity, _ in entity_counts.most_common(limit)]

        return suggestions

    # Metadata management methods
    def get_current_ifs_version(self) -> Optional[str]:
        """Get the current IFS version being used for enhanced search."""
        return self.current_ifs_version

    def has_metadata_enhancement(self) -> bool:
        """Check if metadata enhancement is available."""
        return self.enhanced_search_engine is not None

    def get_available_ifs_versions(self) -> List[str]:
        """Get list of available IFS versions with metadata."""
        return self.metadata_manager.get_available_versions()

    def get_module_statistics(self) -> Dict[str, Any]:
        """Get statistics about available modules from metadata."""
        if self.enhanced_search_engine:
            return self.enhanced_search_engine.get_module_statistics()
        else:
            return {"error": "No metadata available - enhanced search not enabled"}

    def extract_metadata_from_mcp_results(
        self, ifs_version: str, query_results: Dict[str, List[Dict[str, Any]]]
    ) -> bool:
        """
        Extract and save metadata from MCP SQLcl query results.

        Args:
            ifs_version: IFS version for the metadata
            query_results: Results from MCP SQLcl queries

        Returns:
            True if metadata was successfully extracted and saved
        """
        try:
            return self.metadata_manager.process_mcp_results(ifs_version, query_results)
        except Exception as e:
            logger.error(f"Failed to extract metadata from MCP results: {e}")
            return False

    def get_metadata_extract_queries(self) -> Dict[str, str]:
        """
        Get the SQL queries needed for metadata extraction via MCP SQLcl.

        Returns:
            Dictionary of query names to SQL statements
        """
        return {
            "logical_units": """
SELECT 
    lu_name,
    module,
    lu_prompt,
    lu_type,
    base_table,
    base_view
FROM dictionary_sys_lu_active 
WHERE lu_type IN ('Entity', 'BasicLU', 'MasterLU')
    AND module IS NOT NULL
ORDER BY module, lu_name
            """.strip(),
            "modules": """
SELECT DISTINCT 
    module,
    COUNT(*) as lu_count
FROM dictionary_sys_lu_active 
WHERE module IS NOT NULL 
    AND lu_type IN ('Entity', 'BasicLU', 'MasterLU')
GROUP BY module
ORDER BY module
            """.strip(),
            "domain_mappings": """
SELECT DISTINCT
    domain_id,
    client_value,
    db_value,
    description
FROM domain_sys_tab
WHERE client_value IS NOT NULL
    AND db_value IS NOT NULL
ORDER BY domain_id, client_value
            """.strip(),
            "views": """
SELECT 
    view_name,
    module,
    lu_name,
    view_comment
FROM dictionary_sys_view_active
WHERE module IS NOT NULL
    AND lu_name IS NOT NULL
ORDER BY module, lu_name, view_name
            """.strip(),
        }

    def _convert_enhanced_to_search_result(self, enhanced_result) -> "SearchResult":
        """Convert an enhanced search result to a standard SearchResult."""
        # Try to find the result in our index to get full metadata
        try:
            index_results = self.indexer.search_deduplicated(
                query=f'path:"{enhanced_result.file_path}"', limit=1
            )
            if index_results:
                # Use the index result but with enhanced score
                return index_results[0]
        except Exception:
            pass

        # Fallback: create a basic SearchResult from enhanced result
        from datetime import datetime
        from .indexer import SearchResult

        return SearchResult(
            path=enhanced_result.file_path,
            name=Path(enhanced_result.file_path).name,
            type=f".{enhanced_result.content_type}",
            content_preview=enhanced_result.snippet,
            score=enhanced_result.confidence,
            entities=[],
            functions=[],
            line_count=0,
            complexity_score=0.0,
            pagerank_score=0.0,
            modified_time=datetime.now(),
            hash="",
            module=enhanced_result.module,
            logical_unit=enhanced_result.logical_unit,
            entity_name=None,
            component=None,
            pages=[],
            lists=[],
            groups=[],
            entitysets=[],
            iconsets=[],
            trees=[],
            navigators=[],
            contexts=[],
            dependencies=[],
            imports=[],
            highlight="",
            tags=[],
        )

    def _apply_business_ranking(
        self, results: List["SearchResult"], query: str
    ) -> List["SearchResult"]:
        """Apply business logic ranking to search results."""
        query_lower = query.lower()
        business_terms = [
            "authorization",
            "approval",
            "workflow",
            "validation",
            "business",
            "rule",
        ]
        entity_terms = [
            "employee",
            "customer",
            "order",
            "activity",
            "person",
            "company",
        ]

        for result in results:
            original_score = result.score

            # For entity-like searches, prioritize files that contain business logic
            if any(term in query_lower for term in entity_terms):
                if result.type == ".plsql":
                    result.score *= 2.5  # Strong boost for business logic files
                elif result.type == ".views":
                    result.score *= 1.8  # Database views often contain useful logic
                elif result.type == ".projection":
                    result.score *= 1.6  # API projections show the interface
                elif result.type == ".client":
                    result.score *= 1.4  # UI definitions show user interaction
                elif result.type == ".entity":
                    result.score *= 0.8  # Reduce entity priority for practical searches

            # Business logic boosting for certain queries
            if any(term in query_lower for term in business_terms):
                if result.type == ".plsql":
                    result.score *= 3.0  # Very strong boost for business logic files
                elif result.type == ".entity":
                    result.score *= (
                        0.5  # Significantly reduce entity ranking for business queries
                    )

            # File type priority boosting for exact name matches
            if query_lower in result.name.lower():
                filename_match_boost = 1.5
                if result.type == ".plsql":
                    result.score *= (
                        filename_match_boost * 1.2
                    )  # Extra boost for API files
                elif result.type == ".views":
                    result.score *= filename_match_boost
                elif result.type == ".projection":
                    result.score *= filename_match_boost
                elif result.type == ".entity":
                    result.score *= (
                        filename_match_boost * 0.9
                    )  # Slightly less for entities

            # Module relevance boosting for core business modules
            if result.module and result.module.upper() in [
                "PERSON",
                "ORDER",
                "ACCRUL",
                "INVENT",
            ]:
                result.score *= 1.15

            # File size/complexity boosting - larger files often have more business logic
            if result.type == ".plsql" and result.line_count > 100:
                complexity_boost = min(1.5, 1.0 + (result.line_count / 1000))
                result.score *= complexity_boost

        return results

    def _apply_post_search_filters(
        self,
        results: List["SearchResult"],
        file_type: Optional[str],
        module: Optional[str],
        logical_unit: Optional[str],
        min_complexity: Optional[float],
        max_complexity: Optional[float],
    ) -> List["SearchResult"]:
        """Apply filters that couldn't be applied during search."""
        filtered_results = []

        for result in results:
            # File type filter
            if file_type and result.type:
                if not result.type.endswith(file_type):
                    continue
            elif file_type and not result.type:
                # Skip results without type if file type filter is specified
                continue

            # Module filter
            if module and result.module:
                if not result.module.lower().startswith(module.lower()):
                    continue
            elif module and not result.module:
                # Skip results without module if module filter is specified
                continue

            # Logical unit filter
            if logical_unit and result.logical_unit:
                if not result.logical_unit.lower().startswith(logical_unit.lower()):
                    continue
            elif logical_unit and not result.logical_unit:
                # Skip results without logical unit if logical unit filter is specified
                continue

            # Complexity filters
            if min_complexity is not None and result.complexity_score < min_complexity:
                continue
            if max_complexity is not None and result.complexity_score > max_complexity:
                continue

            filtered_results.append(result)

        return filtered_results
